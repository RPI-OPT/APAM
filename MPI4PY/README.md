# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a Python-MP4PY implementation of APAM [(Xu et al. 2020)](#Xu2020).

The results in section 5.4 and section 5.5 are given in this repository.
  

## Usage
> - "run_all.sh" includes how to run the experiments. Running it would produce all the results in section 5.4 and section 5.5.

> - read_datasets.py includes the data reading functions. With the dataset name ('cifar10, CINIC-10, imagenet32'), it return the train_dataset and test_dataset.

> - The folder "models" includes the neural network models ('AllCNN, resnet18, wideresnet28_5') used in the two sections.

> - "cifar10_allcnn_artifical_maxdelay.py" implements the APAM with a given artificial maximum delay for training AllCNN network on the Cifar10 dataset.
> - "run_artifical_delay.sh" test APAM with different maximum artificial delays.
> - "plot_cifar10_maxdelay.py" plots the results with different  maximum artificial delays.


> - "main_apam.py" is the main function to training the neural networks on the given datasets. Its inputs include neural netwrok, datasets, apam or sgd, async or sync, use GPU or not, and other hyper-parameters. 
> - "optim_and_train_apam.py" includes functions called in main_apam.py, including the APAM solver, test function, and train function for one epoch on master and workers in async and sync communication.
> - "run_training_larger_datasets.sh" first tests the training of resnet18 on CINIC-10, then the training of wideresnet28_5 on imagenet32. 
> - "plot_CINIC_async_sync_cpu_time_compare.py" compares the running time on CPU by apam and sync AMSGrad for training resnet18 on CINIC-10.
> - "plot_CINIC_async_gpu_different_workers.py" plots the prediction accuracy for training resnet18 on CINIC-10 with different number of workers and GPUs.
> - "plot_imagenet32_async_sync_cpu_time_compare.py" compares the running time on CPU by apam and sync AMSGrad for training resnet18 on CINIC-10.
> - "plot_imagenet32_async_gpu_different_workers.py" plots the prediction accuracy for training wideresnet28_5 on imagenet32 with different number of workers and GPUs.


## Performance

On Ubuntu Linux 16.04, Dual Intel Xeon Gold 6130 3.7GHz, 32 CPU cores.
### Results with artificial delay 
<img src="./pictures/Artifical_delat_train_acc.png"  width="400"/> <img src="./pictures/Artifical_delat_test_acc.png"  width="400">

predication accuracy by APAM for training the AllCNN network on Cifar10 dataset with Python implementation and artificial delay.

### training the Resnet18 network on CINIC10 dataset  
<img src="./pictures/CINIC10_time_APAM_sync.png"  width="400">

running time (hour) on CPU by APAM and the sync-parallel AMSGrad with Python and MPI4PY implementation for one epoch

<img src="./pictures/CINIC10_train_acc.png"  width="400"/> <img src="./pictures/CINIC10_test_acc.png"  width="400" >

prediction accuracy by APAM and the sync-parallel AMSGrad with Python and MPI4PY implementation for training the Resnet18 network on CINIC10 dataset.

### training the WRN-28-5 on Imagenet32$\times$32 dataset  
<img src="./pictures/Imagenet32_time_APAM_sync.png"  width="400" >

running time (hour) on CPU by APAM and the sync-parallel AMSGrad with Python and MPI4PY implementation for one epoch

<img src="./pictures/Imagenet32_train_acc1.png"  width="400"  > <img src="./pictures/Imagenet32_test_acc1.png"  width="400">
<img src="./pictures/Imagenet32_train_acc5.png"  width="400"  > <img src="./pictures/Imagenet32_test_acc5.png"  width="400">

prediction accuracy by APAM and the sync-parallel AMSGrad with Python and MPI4PY implementation for training the WRN-28-5 network on Imagenet32$\times$32 dataset.


## Reference

- <a name="Xu2020"></a>Yangyang Xu, Yibo Xu, Yonggui Yan, Colin Sutcher-Shepard, Leopold Grinberg, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.
