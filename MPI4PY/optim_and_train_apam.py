from mpi4py import MPI
import numpy as np
import torch
from torch import Tensor
from typing import Iterable
Params = Iterable[Tensor]

### define the optimizer
class APAM:
    def __init__(self, params:Params, device, opt_name='apam', alpha: float=1e-4, amsgrad: bool=True, beta1: float=0.9, beta2: float=0.99, epsilon: float=1e-8, weight_decay=0.):
        
        # optimization variables
        self.params = list(params)
        self.device = device
        
        # variables' information which will used in pack and unpack
        self.num_set = len(self.params)
        self.set_size  = []
        self.set_shape = []
        for param in self.params:
            self.set_size.append(param.data.numel())
            self.set_shape.append(param.data.cpu().numpy().shape)
        self.num_param = sum(self.set_size)
       
        # hyper-parameters for the solver
        self.step_num = 0
        self.opt_name = opt_name
        self.alpha = alpha
        
        self.weight_decay = weight_decay
        
        if self.opt_name=='sgd':
            # no more hyper-parameters in sgd
            pass
        
        if self.opt_name=='apam':
            # hyper-parameters in apam
            self.amsgrad = amsgrad
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = np.empty(self.num_param)
            self.v = np.empty(self.num_param)
            if self.amsgrad:
                self.v_hat = np.empty(self.num_param)
            self.reset()
              
    def reset(self):
        self.step_num = 0
        self.m = np.zeros(self.num_param,dtype=np.float32)
        self.v = np.zeros(self.num_param,dtype=np.float32)
        if self.amsgrad:
            self.v_hat = np.zeros(self.num_param,dtype=np.float32)
             
    def unpack(self,w): # unpack from float array (w) to tensor (in the model)
        # used by the master and workers
        offset = 0
        for idx,param in enumerate(self.params):
            param.data.copy_(torch.tensor(w[offset:offset+self.set_size[idx]].reshape(self.set_shape[idx])).to(self.device))
            offset += self.set_size[idx]
#             print(param)

    def pack_w(self): # pack from tensor ( parameters in the model) to float array (change w)
        # used by the master
        w = np.concatenate([param.data.cpu().numpy().flatten() for param in self.params])
        if w.dtype != np.float32: w=w.astype(np.float32)
        return w

    def pack_g(self):  # pack from tensor ( gradient in the model) to float array (change g)
        # used by the workers.
        g = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.params])
        if g.dtype != np.float32: g=g.astype(np.float32)
        return g

    def zero_grad(self):
        # set the gradients to zero
        for idx,param in enumerate(self.params):
            if param.grad is None:
                continue
            param.grad.data.copy_(torch.tensor(np.zeros(self.set_shape[idx])))
    
    def step(self,w,g): # only used by the master
        ## weight decay, default weight_decay = 0
        g += self.weight_decay * w
        
        self.step_num +=1
        if self.opt_name == 'sgd':
            w -= self.alpha*g
            
        if self.opt_name == 'apam':
            step_size = self.alpha*np.sqrt(1 - np.power(self.beta2,self.step_num)) / (1 - np.power(self.beta1,self.step_num))
            self.m = self.m*self.beta1 + (1-self.beta1)*g
            self.v = self.v*self.beta2 + (1-self.beta2)*(g**2)
            if self.amsgrad:
                self.v_hat = np.maximum(self.v,self.v_hat)
                denom = np.sqrt(self.v_hat) + self.epsilon
            else:
                denom = np.sqrt(self.v) + self.epsilon
                
            w -= step_size*(self.m/denom)
        
        ##  -= is used to update w, which means w is updated in the same memory.
        ##  after call APAM.step(w,g),the value of input w is changed


ROOT = 0
DONE = 999999
NOT_DONE = 1

COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()

# loss function
CRITERION = torch.nn.CrossEntropyLoss()

# synchronous master, training for one epoch
def train_sync_master(num_iter_per_epoch, optimizer):

    peers = list(range(SIZE)); peers.remove(ROOT)

    w = np.zeros(optimizer.num_param,dtype=np.float32)
    ave_g = np.zeros(optimizer.num_param,dtype=np.float32)
    gs = np.empty((SIZE,optimizer.num_param),dtype=np.float32)
 
    num_iter = 0
    w = optimizer.pack_w()
    
    for i in range(num_iter_per_epoch):
        # get gradient to workers and send model to workers
        g = np.zeros(optimizer.num_param, dtype=np.float32)
        COMM.Gather(g, gs, root=0) # synchronous
        ave_g = gs[peers].mean(axis=0)
        optimizer.step(w, ave_g)
        COMM.Bcast(w, root=ROOT) # synchronous
         
    optimizer.unpack(w)
                
# synchronous worker, training for one epoch
def train_sync_worker(model, device, num_iter_per_epoch, train_loader,optimizer):
    
    w = np.empty(optimizer.num_param, dtype=np.float32)
    g = np.empty(optimizer.num_param, dtype=np.float32)
    gs = None
    dataiter = train_loader.__iter__()
    model.train()
    ## loop of data sample in one epoch
    for i in range(num_iter_per_epoch):
        # get the stochastic data sample
        try:
            data, target = dataiter.next()
        except StopIteration:
            del dataiter
            dataiter = train_loader.__iter__()
            data, target = dataiter.next()
            
        data, target = data.to(device), target.to(device)
        
        # get the stochastic gradient
        optimizer.zero_grad()
        output = model(data)
        loss = CRITERION(output, target)
        loss.backward()
        g = optimizer.pack_g()
        
        # send gradient to master and get model from master
        COMM.Gather(g, gs, root=0)
        COMM.Bcast(w, root=ROOT)
        optimizer.unpack(w)

# asynchronous master, training for one epoch
def train_async_master(num_iter_per_epoch, optimizer):

    peers = list(range(SIZE)); peers.remove(ROOT)
    N_peers = len(peers)
    w = np.empty(optimizer.num_param, dtype=np.float32)
    g = np.empty(optimizer.num_param, dtype=np.float32)
    
    
    ## gg is used to storage the gradnets from all workers
    ## requests is the asynchronous receive request from all workers
    gg =  np.empty((N_peers,optimizer.num_param), dtype=np.float32)
    requests  = [MPI.REQUEST_NULL for i in peers]
    for i in range(N_peers):
        requests[i] = COMM.Irecv(gg[i], source=peers[i])
    
    n_master_receive_each_epoch = 0
    w = optimizer.pack_w()
    num_active_workers = N_peers
    # do the loop in one epoch
    while  num_active_workers > 0:
        idx_of_received_list = MPI.Request.Waitsome(requests)
        for i in idx_of_received_list:
            ## update the model with the received gradient
            optimizer.step(w, gg[i])
            n_master_receive_each_epoch += 1
            if n_master_receive_each_epoch < num_iter_per_epoch:
                # not enough gradients are received for one epoch
                # the update should continue
                # send the current model to the worker with tag="Not_Done"
                COMM.Send(w, dest=peers[i], tag=NOT_DONE)
                requests[i] = COMM.Irecv(gg[i], source=peers[i])
            else:
                # enough gradients are received for one epoch
                # the update should stop
                # send the current model to the worker with tag="Done"
                COMM.Send(w, dest=peers[i], tag=DONE)
                num_active_workers -=1
 
    optimizer.unpack(w)

# asynchronous worker, training for one epoch
def train_async_worker(model, device, train_loader, optimizer):
    
    w = np.empty(optimizer.num_param, dtype=np.float32)
    g = np.empty(optimizer.num_param, dtype=np.float32)
    info = MPI.Status()
    info.tag = NOT_DONE
    dataiter = train_loader.__iter__()
    model.train()
    
    ## when the tag==Done, the epoch is finished, the loop is break
    ## otherwise, the worker get a new stochastic gradent and send it to the master.
    while info.tag==NOT_DONE:
        
        # get the stochastic data sample
        try:
            data, target = dataiter.next()
        except StopIteration:
            del dataiter
            dataiter = train_loader.__iter__()
            data, target = dataiter.next()
        data, target = data.to(device), target.to(device)
        
        # get the stochastic gradient
        optimizer.zero_grad()
        output = model(data)
        loss = CRITERION(output, target)
        loss.backward()
        g = optimizer.pack_g()
        
        # send gradient to master and get a new model from master
        COMM.Send(g, dest=ROOT)
        COMM.Recv(w, source=ROOT, tag=MPI.ANY_TAG, status=info)
        optimizer.unpack(w)

# test the model
def validate(model, device, val_loader, topk=(1,)):
    maxk = max(topk)
    test_loss = 0
    corrects = {}
    for k in topk:
        corrects[k] = 0
    
    total = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = CRITERION(output, target)

            test_loss += loss.item()
            
            total += target.size(0)
            _, pred = output.topk(maxk, 1, True, True) # top5
            pred = pred.t()
            correct_topk = pred.eq(target.view(1, -1).expand_as(pred))
             
            for k in topk:
                corrects[k] += correct_topk[:k].reshape(-1).float().sum(0, keepdim=True).item()

    results = {}
    results['loss'] = test_loss/len(val_loader)
    for k in topk:
        results['acc_top'+str(k)] = 100.*corrects[k]/total
    
    return results
    
