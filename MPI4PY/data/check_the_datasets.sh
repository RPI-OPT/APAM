data_name=CINIC-10
#data_name=food-101
#data_name=fruits-360
#data_name=tiny-imagenet-200
#   data_name=imagenet32
echo $data_name

echo "size"
du -sh $data_name 
du -sh $data_name/train 
du -sh $data_name/test

echo '# classes'
ls $data_name/train | wc -l
ls $data_name/test | wc -l

echo '# train images'
find $data_name/train -type d -or -type f -printf '.' | wc -c

echo '# test images'
find $data_name/test -type d -or -type f -printf '.' | wc -c



