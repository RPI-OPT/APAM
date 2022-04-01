import argparse
import os
import time

from mpi4py import MPI
import numpy as  np
 
import torch 

import sys
sys.path.append("..")

## function for reading the dataset
from read_datasets import read_datasets

## function for the newtwork architecture
import models as models

## APAM solver and train functions
from optim_and_train_apam import *

## Communication setting
ROOT = 0
COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()

## Test the model for both training and testing dataset
## Note while training, the forwark and backward of models are on workers
## the master only update the model with the gradient.
## The test is done on the master.
## We test the model on the training dataset with model.train() in order to get the mean and variance in the batchnorm layers.
## We test the model on the testing dataset with model.test().
def test(model, device, pred_train_loader, test_loader, topk=(1,)):
    state = {}
    
    # test on the training dataset
    model.train()
    train_results = validate(model, device, pred_train_loader, topk=topk)
    for key, value in train_results.items():
        state['train_'+key]=value
        
    # test on the testing dataset
    model.eval()
    test_results  = validate(model, device, test_loader, topk=topk)
    for key, value in test_results.items():
        state['test_'+key]=value
    
    return state

#### hyper-parameters
parser = argparse.ArgumentParser()

## dataset and model
parser.add_argument('--data_name', type=str, default='CINIC-10')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--model_name', type=str, default='resnet18')

## optimizer
parser.add_argument('--opt_name', type=str, default='apam', choices = ['sgd','apam'])
parser.add_argument('--alpha', type=float, default=0.0001)
parser.add_argument('--amsgrad', type=bool, default=True)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=0)

## loader
parser.add_argument('--train_batch_size', type=int, default=40)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--pred_train_batch_size', type=int, default=100)

## epoch and epochs
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--log_per_epoch', type=int, default=10)
 
## communication
parser.add_argument('--communication', type=str, default='sync', choices = ['sync','async'])

##CPU or GPU
parser.add_argument('--cuda', type=str, default='True', choices=('True','False'))

def main():
    args = parser.parse_args()
    # parameters for optimizer
    opt_name = args.opt_name
    alpha =  args.alpha
    amsgrad = args.amsgrad
    beta1 =args.beta1
    beta2 =args.beta2
    epsilon = args.eps
    weight_decay = args.weight_decay
    
    # epochs
    num_epochs = args.epochs
    epoch = args.epoch
    
    # set the random seed
    seed = RANK*100+20220329
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ##CPU or GPU
    use_cuda = args.cuda=='True'
    device = torch.device("cuda:{}".format(RANK % torch.cuda.device_count()) if (torch.cuda.is_available() and use_cuda) else "cpu")
        
    # read the dateset
    train_dataset, test_dataset = read_datasets(args.data_name,args.data_dir,device=device)
    
    # set the model
    model = getattr(models, args.model_name)(num_classes=args.num_classes).to(device)
    
    if args.data_name=='imagenet32':
        topk = (1,5) ## if it is on imagenet32, we test the top1 and top5 accuracy
    else:
        topk = (1,) ## otherwise, we test only the top1 and top5.
 
    optimizer = APAM(model.parameters(), device=device, opt_name=opt_name, alpha=alpha, amsgrad=amsgrad, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
    
    # parameters for loader
    train_batch_size = args.train_batch_size
    pred_train_batch_size = args.pred_train_batch_size
    test_batch_size = args.test_batch_size
    # define the loader
    Nworkers = SIZE-1
    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and use_cuda) else {}
    if args.communication == 'sync':
        my_train_batch_size = train_batch_size//Nworkers
        if  RANK  <  train_batch_size%Nworkers and (RANK != ROOT):
            my_train_batch_size = my_train_batch_size+1
        train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=my_train_batch_size, shuffle=True, **kwargs)
    if args.communication == 'async':
        train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    
    # define the solver APAM
    pred_train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=pred_train_batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader( test_dataset,  batch_size=test_batch_size, shuffle=True, drop_last=True,  **kwargs)
    
    # number of iterations for one epoch
    num_iter_per_epoch = np.int(np.ceil(len(train_dataset)/train_batch_size))
 
    w = np.empty(optimizer.num_param,dtype=np.float32)

    if RANK == ROOT:
        # file name of the results
        filename = args.data_name + '_' + args.model_name +'_'+opt_name+'_comm'+'_'+args.communication+'_SIZE_'+str(SIZE)+'_epochs_'+str(num_epochs) +'_bacthsize'+str(train_batch_size)+'_alpha'+str(alpha)
        
        if (torch.cuda.is_available() and use_cuda):
            filename = 'GPU' + filename
        else:
            filename = 'CPU' + filename
            
        print(filename+ ' is computing ..... ',flush=True)
        f = open('./results/results_'+filename+'.txt' ,"w")
         
         
        ## Initialize the time record
        total_test_time = 0.
        time_start = time.time()
        test_state = {}
        test_time0 = time.time()
        
        ## Test with the initial model
        if use_cuda: test_state = test(model, device, pred_train_loader, test_loader, topk=topk)
        total_test_time += time.time()-test_time0

        # print the results titles to the file and screen
        print('epoch\t' + ''.join([str(key)+'\t' for key in test_state.keys()])+'time_since_begin(without testing)\t time_since_begin(with testing)',flush=True)
        f.write('epoch\t' + ''.join([str(key)+'\t' for key in test_state.keys()])+'time_since_begin(without testing)\t time_since_begin(with testing)\n')
        # print the initial results to the file and screen
        print('{}\t'.format(epoch) + ''.join(['{:.4f}\t'.format(value) for value in test_state.values()])+'{:.4f}\t {:.4f}'.format(time.time()-time_start-total_test_time, time.time()-time_start),flush=True)
        f.write('{}\t'.format(epoch) + ''.join(['{:.4f}\t'.format(value) for value in test_state.values()])+'{:.4f}\t {:.4f}\n'.format(time.time()-time_start-total_test_time, time.time()-time_start))
 
        w = optimizer.pack_w()
    
    # synch the model at the begining.
    COMM.Bcast(w, root=ROOT)
    optimizer.unpack(w)
    
    # a variable which be change to a small number while debugging
    #max_iter_per_epoch = 1000
    max_iter_per_epoch =  num_iter_per_epoch
    
    # loop about the epoch
    while epoch<num_epochs:
        epoch+=1 
 
        # do the synchronous training
        if args.communication == 'sync':
            if RANK==ROOT:
                train_sync_master(max_iter_per_epoch,optimizer)
            else:
                train_sync_worker(model, device, num_iter_per_epoch, train_loader,optimizer)
                 
        # do the asynchronous training
        if args.communication == 'async':
            if RANK==ROOT:
                train_async_master(max_iter_per_epoch,optimizer)
            else:
                train_async_worker(model,device,train_loader,optimizer)
             
        # do the testing
        if (RANK==ROOT) and (epoch%args.log_per_epoch==0 or epoch==1):
            test_time0 = time.time()
            if use_cuda: test_state = test(model, device, pred_train_loader, test_loader, topk=topk)
            total_test_time += time.time()-test_time0
            # print the results to the file and screen
            print('{}\t'.format(epoch) + ''.join(['{:.4f}\t'.format(value) for value in test_state.values()])+'{:.4f}\t {:.4f}'.format(time.time()-time_start-total_test_time, time.time()-time_start),flush=True)
            f.write('{}\t'.format(epoch) + ''.join(['{:.4f}\t'.format(value) for value in test_state.values()])+'{:.4f}\t {:.4f}\n'.format(time.time()-time_start-total_test_time, time.time()-time_start))
 
        if (RANK==ROOT):  w = optimizer.pack_w()
        # synchronous after every epoch
        COMM.Bcast(w, root=ROOT)
        optimizer.unpack(w)
        
    time_end = time.time()
    if RANK==ROOT:
        # print the total time to the file and screen
             
        print('training time {} seconds (with {} workers)\n'.format(time_end-time_start-total_test_time, Nworkers))
        print('total time {} seconds (with {} workers)\n'.format(time_end-time_start, Nworkers))

        f.write('Training time {} seconds (with {} workers)\n'.format(time_end-time_start-total_test_time, Nworkers))
        f.write('total time {} seconds (with {} workers)\n'.format(time_end-time_start, Nworkers))
         
        f.close()

if __name__ == '__main__':
    main()
