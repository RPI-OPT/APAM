# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a C++-OpenMP implementation of APAM [(Xu et al. 2020)](#Xu2020).


## Usage

Under the code directory, compile code. 

```sh
make 
```

Run the code. Example by 8 threads (1 master and 7 workers), mini-batch = 64, and maxepoch = 100 for solving logistic regression problem on rcv1 data, under bash:

```sh
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
rcv1_train.binary.bz2
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
rcv1_test.binary.bz2
bzip2 -d *.bz2
./LR_APAM.exe rcv1_train.binary rcv1_test.binary 64 100 8
```

## Performance

On Ubuntu Linux 16.04, Dual Intel Xeon Gold 6130 3.7GHz, 32 CPU cores

|# threads | train loss | test accuracy | train time (sec) | speedup|
| :-------: | :---------: | :-----------: | :--------------: | :-----: |
| 1           | 0.00271   | 0.9990       | 320.99            | 1        |
| 2           | 0.00267   | 0.9990       | 168.15            | 1.91   |
| 4           | 0.00239   | 0.9992       | 88.31              | 3.63   |
| 8           | 0.00268   | 0.9990       | 48.27              | 6.65   |
| 16         | 0.00285   | 0.9991       | 27.60              | 11.63 |
| 32         | 0.00271   | 0.9990       | 19.71              | 16.29 |

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.
