# APAM: Asynchronous Parallel Adaptive stochastic gradient Method

This repository contains a C++-MPI implementation of APAM [(Xu et al. 2020)](#Xu2020).

## Usage

Install MPI as needed. Example on Ubuntu Linux with OpenMPI:

```sh
sudo apt-get install openmpi-bin libopenmpi-dev
```

Install libtorch as needed. Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and get the correct link to download. Example on Linux without GPU support:

```sh
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
```

Under the code directory, compile code. The `/absolute/path/to/libtorch` below is where you unzip in the last step, concatenated with the folder name.

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

Run the code. Example by using 5 mpi processes (1 master and 4 workers), each of which uses only 1 thread, under bash:

```sh
OMP_NUM_THREADS=1 mpirun -np 5 APAM_LeNet5
```

## Performance

On Ubuntu Linux 16.04, Dual Intel Xeon Gold 6130 3.7GHz, 32 CPU cores

| # Workers | Train accuracy | Test accuracy | Train time (sec) | Speedup |
| :-------: | :-------: | :-----------: | :--------------: | :-----: |
| 1         | 100%    | 98.95%        | 1661.89           | 1.00    |
| 2         | 100%    | 99.05%        | 821.95           | 2.02    |
| 5         | 100%    | 99.01%        | 369.88           | 4.49    |
| 10         | 100%    | 99.02%        | 233.73           | 7.11    |
| 20         | 100%    | 99.08%        | 145.51           | 11.42    |

## Reference

- <a name="Xu2020"></a>Yangyang Xu, Colin Sutcher-Shepard, Yibo Xu, and Jie Chen. [Asynchronous parallel adaptive stochastic gradient methods](https://arxiv.org/abs/2002.09095). Preprint arXiv:2002.09095, 2020.

