
echo "Training neural networks on large datasets"
function_name=main_apam.py
echo

###################################################################################################
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
echo "running apam and the sync AMSGrad on CPU to see the speed up."
epochs=1
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
echo "plot the running time on CPU by APAM and sync AMSGrad"
python plot_CINIC_async_sync_cpu_time_compare.py --epochs=$epochs
###################################################################################################
echo "running APAM on GPU to see the effect of the delay."
epochs=200
log_per_epoch=10
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 3  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
echo "plot the prediction accuracy by APAM on GPU by apam and sync AMSGrad"
python plot_CINIC_async_gpu_different_workers.py --epochs=$epochs --log_per_epoch=$log_per_epoch
echo
echo

###################################################################################################
####  Datasets: imagenet32*32
####  Model: WRN-28-5(wideresnet28_5)
###################################################################################################
echo "Training WRN-28-5 on the CINIC10 dataset"
data_name=imagenet32
model_name=wideresnet28_5
num_classes=1000
alpha=0.001
train_batch_size=100
echo "running APAM and the sync AMSGrad on CPU to see the speed up."
epochs=1
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=False
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
mpirun -np 21 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=sync --cuda=False
echo "plot the running time on CPU by APAM and sync AMSGrad"
python plot_imagenet32_async_sync_cpu_time_compare.py --epochs=$epochs
###################################################################################################
echo "running APAM on GPU to see the effect of the delay."
epochs=40
log_per_epoch=2
mpirun -np 2  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 3  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 6  python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
mpirun -np 11 python $function_name --data_name=$data_name --model_name=$model_name --num_classes=$num_classes --epochs=$epochs --alpha=$alpha --train_batch_size=$train_batch_size --opt_name=apam --communication=async --cuda=True --log_per_epoch=$log_per_epoch
echo "plot the prediction accuracy by APAM on GPU by apam and sync AMSGrad"
python plot_imagenet32_async_gpu_different_workers.py --epochs=$epochs --log_per_epoch=$log_per_epoch
echo
echo
