import numpy as  np
import torch

import time
 
## function for the newtwork architecture
import models
import torch.nn.functional as F

## function for reading the dataset
from read_datasets import read_datasets


## keep at most "max_delay" history model
## put them in the same memory,
## and indicate the position of current model by "pos_now".
def pos_next_fun(pos_now, max_delay):
    return (pos_now+1) % max_delay # take the mod

## APAM solver
from typing import Iterable
from torch import Tensor
Params = Iterable[Tensor]
class APAM:
    def __init__(self, params:Params, device, opt_name='apam', alpha: float=1e-4, amsgrad: bool=True, beta1: float=0.9, beta2: float=0.99, epsilon: float=1e-8):
        self.params = list(params)
        self.device = device
        self.num_set = len(self.params)
        self.set_size  = []
        self.set_shape = []
        for param in self.params:
            self.set_size.append(param.data.numel())
            self.set_shape.append(param.data.cpu().numpy().shape)
        self.num_param = sum(self.set_size)
        
        self.step_num = 0
        
        self.opt_name = opt_name
        self.alpha = alpha
        if self.opt_name=='sgd':
            pass
        
        if self.opt_name=='apam':
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
        offset = 0
        for idx,param in enumerate(self.params):
            param.data.copy_(torch.tensor(w[offset:offset+self.set_size[idx]].reshape(self.set_shape[idx])).to(self.device))
            offset += self.set_size[idx]
#             print(param)
          
    def pack_w(self): # pack from tensor ( parameters in the model) to float array (change w)
        w = np.concatenate([param.data.cpu().numpy().flatten() for param in self.params])
        if w.dtype != np.float32: w=w.astype(np.float32)
        return w
  
    def pack_g(self):  # pack from tensor ( gradient in the model) to float array (change g)
        g = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.params])
        if g.dtype != np.float32: g=g.astype(np.float32)
        return g
    
    def zero_grad(self):
        for idx,param in enumerate(self.params):
            if param.grad is None:
                continue
            param.grad.data.copy_(torch.tensor(np.zeros(self.set_shape[idx])))
    
    def step(self,w,g):
        self.step_num +=1
        
        if self.opt_name == 'sgd':
            w = w - self.alpha*g
            
        if self.opt_name == 'apam':
                
            step_size = self.alpha*np.sqrt(1 - np.power(self.beta2,self.step_num)) / (1 - np.power(self.beta1,self.step_num))
  
            self.m = self.m*self.beta1 + (1-self.beta1)*g
            self.v = self.v*self.beta2 + (1-self.beta2)*(g**2)
            if self.amsgrad:
                self.v_hat = np.maximum(self.v, self.v_hat)
                denom = np.sqrt(self.v_hat) + self.epsilon
            else:
                denom = np.sqrt(self.v) + self.epsilon
                
            w = w - step_size*(self.m/denom)

        # In optim_and_train_apam.py, "-=" is used to update the value of w, but the memory does not change
        # Here, w = w - ... is used to update w. w is allocated a new memory
        # we return the updated w at new memory, which will save at next position in ws.
        return w 


## train with the artifical maximum delay for one epoch
def train_max_delay(model, train_loader, num_iter_per_epoch, optimizer, device, ws, pos_now):
        
    max_delay = len(ws)
    dataiter = train_loader.__iter__()
    num_iter = 0
    model.train()
    # loop for the data samples in one epoch
    for i in range(num_iter_per_epoch):
        try:
            data, target = dataiter.next()
        except StopIteration:
            del dataiter
            dataiter = train_loader.__iter__()
            data, target = dataiter.next()
            
        data, target = data.to(device), target.to(device)
        
        # randomly select one model with the maximum delay
        # and put the selected delayed model to the solver
        optimizer.unpack(ws[np.random.randint(min(max_delay,optimizer.step_num+1))])
        
        # get the stochastic gradient at the delayed model
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        g = optimizer.pack_g()
        
        # update the current model with the delayed gradinet
        # and save the updated model to the next position in ws

        pos_next = pos_next_fun(pos_now,max_delay)
        ws[pos_next] = optimizer.step(ws[pos_now], g)
        pos_now = pos_next
        
    return pos_now
 
## test with the current model, return the test loss and accuracy
def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            test_loss += loss.item()  #note loss is sum in the above nll_loss, sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        
    return [test_loss, test_acc]
     
#### hyper-parameters
import argparse
parser = argparse.ArgumentParser()
## dataset
parser.add_argument('--data_dir', type=str, default='./data/cifar10')

## model
parser.add_argument('--model_name', type=str, default='AllCNN')

## optimizer
parser.add_argument('--opt_name', type=str, default='apam', choices=['sgd','apam'])
parser.add_argument('--alpha', type=float, default=0.0001)
parser.add_argument('--amsgrad', type=bool, default=True)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)

## loader
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--pred_train_batch_size', type=int, default=1000)

## epoch and epochs
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--log_per_epoch', type=int, default=1)
 
## max_delay
parser.add_argument('--max_delay', type=int, default=1)

parser.add_argument('--cuda', type=str, default='True', choices=('True','False'))
parser.add_argument('--gpu_id', type=int, default=0)

def main():
    args = parser.parse_args()
    # parameters for optimizer
    opt_name = args.opt_name
    alpha =  args.alpha
    amsgrad = args.amsgrad
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.eps
    
    # epochs
    num_epochs = args.epochs
    epoch = args.epoch
    
    # artifical maximun delay
    max_delay = args.max_delay
      
    # set randomness seed
    seed = 20211203
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ##CPU or GPU
    use_cuda = args.cuda=='True'
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")
  
    # load datasets
    train_dataset, test_dataset = read_datasets('cifar10', data_dir=args.data_dir)

    # parameters for loader
    train_batch_size = args.train_batch_size
    pred_train_batch_size = args.pred_train_batch_size
    test_batch_size = args.test_batch_size
    # define the loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and use_cuda) else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    pred_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=pred_train_batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False, **kwargs)
    
    # number of iterations for one epoch
    num_iter_per_epoch = np.int(np.ceil(len(train_dataset)/train_batch_size))
  
    # file name of the results
    filename = './results/results_cifar10_allcnn' + '_' + opt_name + '_epochs' + str(num_epochs) + '_bacthsize' + str(train_batch_size) + '_MaxDelay' + str(max_delay) + '.txt'
    f = open(filename, "w")
    # print the file name to the screen
    print(filename+ ' is computing ..... ', flush=True)
    
        
    # print the results titles in the file and screen
    f.write('epoch\t test_loss\t test_acc\t train_loss\t train_acc \t time_since_begin(without testing)\t time_since_begin(with tetsing)\n')
    print('epoch\t test_loss\t test_acc\t train_loss\t train_acc \t time_since_begin(without testing)\t time_since_begin(with testing)', flush=True)
     
    # define the neural network model
    model = getattr(models, args.model_name)().to(device)
    # define the solver APAM
    optimizer = APAM(model.parameters(), device=device, opt_name=opt_name, alpha=alpha, amsgrad=amsgrad, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    # ws is the memory for the "max_delay" memory
    # the initial model is put at the 0 position.
    ws = [None for i in range(max_delay)] # no delay for current model
    ws[0] = optimizer.pack_w()
    pos_now = 0
    total_test_time = 0.
    time_start = time.time()
    
    # loop about the epoch
    while epoch<num_epochs:
        pos_now = train_max_delay(model, train_loader, num_iter_per_epoch, optimizer, device, ws, pos_now)
        epoch += 1
        
        # for every args.log_per_epoch epochs, do the test and print the results to the file and screen
        # the test is based on current model
        optimizer.unpack(ws[pos_now])
        if (epoch%args.log_per_epoch==0):
            test_time0 = time.time()
            
            [test_loss, test_acc]  = test(model, device, test_loader)
            [train_loss, train_acc] = test(model, device, pred_train_loader)
            
            total_test_time += time.time()-test_time0
            # print the results to the file and screen
            print('{}\t {}\t {}\t {}\t {}\t{}\t{}'.format(epoch, test_loss, test_acc, train_loss, train_acc, time.time()-time_start-total_test_time, time.time()-time_start), flush=True)
            f.write('{}\t {}\t {}\t {}\t {}\t{}\t{}\n'.format(epoch, test_loss, test_acc, train_loss, train_acc, time.time()-time_start-total_test_time, time.time()-time_start))
           
    f.close()
        
if __name__ == '__main__':
    main()
