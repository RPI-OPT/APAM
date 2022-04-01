The file include the code for the added new numerical experiments (section 5.4 and  section 5.5) in the paper 
"Parallel and distributed asynchronous adaptive stochastic gradient methods".

The code is in Python with MPI4PY for distributed communication.
The experiments were run on a Dell workstation with 32 CPU cores, 64 GB memory, and two Quadro RTX 500 GPUs.


"run_all.sh" includes how to run the experiments. Running it would produce all the results in section 5.4 and section 5.5.
read_datasets.py includes the data reading functions. With the dataset name ('cifar10, CINIC-10, imagenet32'), it return the train_dataset and test_dataset.

The folder "data" includes three subfolders, each subfolder is one datasets. 
-- The folder "cifar10" includes the cifar10 dataset, which is downloaded directed by "torchvision.datasets.CIFAR10()".
-- The folder "CINIC-10"  includes the CINIC10 dataset, which is downloaded and reorganized in the "train" and "test".
-- The folder "imagenet32" includes the imagenet32 dataset. We download the .npy format and then we transform back to images and save them in "train" and "test". "image_mean.npy" is the mean of all training images. 

For "CINIC-10" and "imagenet32", "train" and "test" include subfolders with class name as the fold name and each subfolders includes the images in that class.


The structure of the folder "data" is as fellow:
-data
----cifar10 
----CINIC-10
--------train
------------class1
----------------image1_in_class1
----------------image2_in_class1
----------------....
------------class2
----------------image1_in_class2
----------------image2_in_class2
----------------....
------------....
--------test
------------class1
----------------image1_in_class1
----------------image2_in_class1
----------------....
------------class2
----------------image1_in_class2
----------------image2_in_class2
----------------....
------------....
----imagenet32
--------image_mean.npy
--------train
------------class1
----------------image1_in_class1
----------------image2_in_class1
----------------....
------------class2
----------------image1_in_class2
----------------image2_in_class2
----------------....
------------....
--------test
------------class1
----------------image1_in_class1
----------------image2_in_class1
----------------....
------------class2
----------------image1_in_class2
----------------image2_in_class2
----------------....
------------....


The folder "models" includes the neural network models ('AllCNN, resnet18, wideresnet28_5') used in the two sections.
The folder "results" will include all the results after the test.
The folder "pictures" will include all the plotted pictures based on the results by plot functions.


"cifar10_allcnn_artifical_maxdelay.py" implements the APAM with a given artificial maximum delay for training AllCNN network on the Cifar10 dataset corresponding the section 5.4.
"run_artifical_delay.sh" test APAM with different maximum artificial delays.
"plot_cifar10_maxdelay.py" plots the results with different  maximum artificial delays (Figure 9).


"main_apam.py" is the main function to training the neural networks on the given datasets corresponding the section 5.5. Its inputs include neural netwrok, datasets, apam or sgd, async or sync, use GPU or not, and other hyper-parameters. 
"optim_and_train_apam.py" includes functions called in main_apam.py, including the APAM solver, test function, and train function for one epoch on master and workers in async and sync communication.
"run_training_larger_datasets.sh" first tests the training of resnet18 on CINIC-10, then the training of wideresnet28_5 on imagenet32. 
"plot_CINIC_async_sync_cpu_time_compare.py" compares the running time on CPU by apam and sync AMSGrad for training resnet18 on CINIC-10 (Left subfigure in Figure 10).
"plot_CINIC_async_gpu_different_workers.py" plots the prediction accuracy for training resnet18 on CINIC-10 with different number of workers and GPUs (Figure 11).
"plot_imagenet32_async_sync_cpu_time_compare.py" compares the running time on CPU by apam and sync AMSGrad for training resnet18 on CINIC-10 (Right subfigure in Figure 10).
"plot_imagenet32_async_gpu_different_workers.py" plots the prediction accuracy for training wideresnet28_5 on imagenet32 with different number of workers and GPUs (Figure 12).
