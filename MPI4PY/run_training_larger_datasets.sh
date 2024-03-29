echo "Training neural networks on large datasets"
function_name=main_apam.py
echo

###################################################################################################
####  Training neural networks on large datasets
####  Datasets: CINIC-10
####  Model: resnet18
###################################################################################################
echo
echo "Training Resnet18 on the CINIC10 dataset"
 
data_name=CINIC-10
model_name=resnet18
num_classes=10
alpha=0.0001
train_batch_size=80
echo "running APAM and the sync AMSGrad with CPUs to see the speed up."
epochs=1
echo "APAM"
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
echo "sync AMSGrad"
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
###################################################################################################
echo "running APAM with GPUs to see the effect of the delay."
epochs=200
log_per_epoch=10
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 3  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
echo

###################################################################################################
####  Training neural networks on large datasets
####  Datasets: imagenet32*32
####  Model: WRN-28-5(wideresnet28_5)
###################################################################################################
echo "Training WRN-28-5 on the imagenet32*32 dataset"

data_name=imagenet32
model_name=wideresnet28_5
num_classes=1000
alpha=0.001
train_batch_size=100
echo "running APAM and the sync AMSGrad with CPUs to see the speed up."
epochs=1
echo "APAM"
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
echo "sync AMSGrad"
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
###################################################################################################
echo "running APAM with GPUs to see the effect of the delay."
epochs=40
log_per_epoch=2
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 3  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
echo
