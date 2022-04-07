
################################################################################################
####  Test APAM with the artifical delay
####  Datasets: Cifar10
####  Model: AllCNN
################################################################################################
function_name=cifar10_allcnn_artifical_maxdelay.py

echo
echo "running APAM on cifar10 with artifical maximum delay"
model_name=AllCNN
alpha=0.0001
train_batch_size=100
epochs=400
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=1
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=6
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=11
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=21
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=51
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=101
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=201
python $function_name --model_name=$model_name --alpha=$alpha --train_batch_size=$train_batch_size --epochs=$epochs --opt_name=apam --max_delay=401
echo "plot the predication accuracy about the artifical delay"
python plot_cifar10_maxdelay.py --epochs=$epochs
echo
echo
