The file includes the code for the added new numerical experiments (sections 5.4 and 5.5) in the paper 
"Parallel and distributed asynchronous adaptive stochastic gradient methods".

The code is in Python with MPI4PY for distributed communication.
The experiments were run on a Dell workstation with 32 CPU cores, 64 GB memory, and two Quadro RTX 500 GPUs.

Running "run_all.sh" would produce all the results in sections 5.4 and 5.5.

The folder "data" includes three datasets (cifar10, CINIC-10, imagenet32) and "about_data.txt" which includes more information about each dataset.

The folder "models" includes the neural network models ('AllCNN, resnet18, wideresnet28_5').
The folder "results" will include all the results after the test.
The folder "pictures" will include all the plotted pictures by the plot functions.

"read_datasets.py" includes the data reading functions. With the dataset name ('cifar10, CINIC-10, imagenet32'), it returns the train_dataset and test_dataset.

"cifar10_allcnn_artifical_maxdelay.py" implements the APAM with a given artificial maximum delay for training the AllCNN network on the Cifar10 dataset corresponding to section 5.4.
"run_artifical_delay.sh" tests APAM with different maximum artificial delays.
"plot_cifar10_maxdelay.py" plots the results with different maximum artificial delays (Figure 9).

"main_apam.py" is the main function to train the given neural networks on the given dataset corresponding to section 5.5. Its inputs include the name of the neural network, the name of the dataset, apam or sgd, async or sync, use GPU or not, and other hyper-parameters. 
"optim_and_train_apam.py" includes functions called in main_apam.py, including the APAM solver, test function, and train function for one epoch on the master and workers in async and sync communication modes.
"run_training_larger_datasets.sh" first tests the training of resnet18 on the CINIC-10, then the training of wideresnet28_5 on the imagenet32. 
"plot_CINIC_async_sync_cpu_time_compare.py" compares the running time with CPUs by APAM and sync AMSGrad for training resnet18 on CINIC-10 (Left subfigure in Figure 10).
"plot_CINIC_async_gpu_different_workers.py" plots the prediction accuracy by APAM for training resnet18 on CINIC-10 with different numbers of workers and GPUs (Figure 11).
"plot_imagenet32_async_sync_cpu_time_compare.py" compares the running time with CPUs by APAM and sync AMSGrad for training wideresnet28_5 on the imagenet32 (Right subfigure in Figure 10).
"plot_imagenet32_async_gpu_different_workers.py" plots the prediction accuracy by APAM for training wideresnet28_5 on imagenet32 with different numbers of workers and GPUs (Figure 12).

