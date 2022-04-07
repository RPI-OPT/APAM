
# running this file will produce the results in section 5.4 and section 5.5

# run the results in section 5.4
# run APAM with the artificial delay for training AllCNN network on the Cifar10 dataset
./run_artifical_delay.sh
# we run for 400 epochs with one GPU
# we run with 8 different maximum delays and each one takes about 5 hours.
# If only with CPUs, it may run for a log time.

# run the results in section 5.5
# run APAM for training neural networks on larger datasets
# first train resnet18 on CINIC-10
# then train wideresnet28_5 on imagenet32
./run_training_larger_datasets.sh
## Train resnet18 on CINIC-10
## For running with CPUs for one epoch to compare the speed up between APAM and sync AMSgrad
##     it takes 1.79 hours with 1 workers,
##          and 0.39, 0.25, 0.17 hours with 5, 10, 20 workers by APMA
##          and 0.68, 0.77, 1.08 hours with 5, 10, 20 workers by sync AMSgrad
## For running with two GPUs for 200 epochs to see the effect of the delay
##     it takes 42.39, 24.12, 24.05, 24.54 hours with 1, 2, 5, 10 workers by APAM.
##     If only with CPUs, it may run for a log time.

## Train wideresnet28_5 on imagenet32
## For running with CPUs for one epoch to compare the speed up between APAM and sync AMSgrad
##     it takes 24.48 hours with 1 workers,
##          and 5.31, 3.27, 2.20 hours with 5, 10, 20 workers by APAM
##          and 6.57, 5.69, 6.20 hours with 5, 10, 20 workers by sync AMSgrad
## For running with two GPUs for 40 epochs to see the effect of the delay
##     it takes 51.73, 27.84, 25.39, 26.49 hours with 1, 2, 5, 10 workers by APAM.
##     If only with CPUs, it may run for a log time.
