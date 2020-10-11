// This code implements the APAM optimizer with MPI parallelization.
// It is run on the LeNet-5 neural network with FashionMNIST data set.
//
// For C++, we manipulate primitive data types (e.g., float), because
// mpi can only handle these types; but for python, we can manipulate
// pytorch data types (e.g., torch::Tensor), because pytorch has an
// mpi interface handling such.

#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <torch/torch.h>

#include <string>
#include <iostream>
// global constant
const int ROOT = 0;       // this is the master
const int DONE = 999999;  // the "done" flag. choose some special number
const int NOT_DONE = 1;   // some flag different from "done"

//-----------------------------------------------------------------------------

// define the network architecture
struct Net : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));
        conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(2));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(2));
        fc1 = register_module("fc1", torch::nn::Linear(256, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, 10));
    }
    
    // LeNet-5
    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = torch::relu(x);
        x = pool1->forward(x);
        x = conv2->forward(x);
        x = torch::relu(x);
        x = pool2->forward(x);
        x = torch::flatten(x, 1);
        x = fc1->forward(x);
        x = torch::relu(x);
        x = fc2->forward(x);
        x = torch::relu(x);
        x = fc3->forward(x);
        x = torch::log_softmax(x, 1);
        return x;  // output
    }
};

//-----------------------------------------------------------------------------

// define the optimizer. the official APAM is based on AMSGRAD, but it
// should also work on the base of ADAM. AMSGRAD differs from ADAM in
// only one line
class APAM {
    
public:
    APAM(Net &model,
         double alpha = 1e-3,
         bool amsgrad = true,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-8);
    ~APAM();
    
    // the meat
    void unpack_w(float *w);       // unpack from float array to tensor
    void pack_w(float *w);         // pack from tensor to float array
    void pack_g(float *g);         // pack from tensor to float array
    void zero_grad(void);          // called by worker. conform with pytorch
    void step(float *g, float *w); // called by master. conform with pytorch
    
    // utility functions
    int get_num_param(void);
    
private:
    Net& _model;
    double _alpha;
    double _beta1;
    double _beta2;
    double _epsilon;
    bool _amsgrad;
    int num_set;   // as in "number of weight matrices"
    int *set_size; // as in "number of elements in a weight matrix"
    int num_param; // = sum(set_size)
    float *m;      // storage
    float *v;      // storage
    float *v_hat;  // storage
    int t;         // iteration number
    
    // not sure if this function should be made public or not
    void reset(void);
};

APAM::
APAM(Net &model,
     double alpha,
     bool amsgrad,
     double beta1,
     double beta2,
     double epsilon) : _model(model) {
    
    _alpha = alpha;
    _beta1 = beta1;
    _beta2 = beta2;
    _epsilon = epsilon;
    _amsgrad = amsgrad;
    num_set = _model.parameters().size();
    set_size = new int [num_set];
    num_param = 0;
    for (int i = 0; i < num_set; i++) {
        set_size[i] = _model.parameters()[i].numel();
        num_param += set_size[i];
    }
    m = new float [num_param];
    v = new float [num_param];
    if (_amsgrad) {
        v_hat = new float [num_param];
    }
    else {
        v_hat = NULL;
    }
    reset();
}

APAM::
~APAM() {
    delete [] set_size;
    delete [] m;
    delete [] v;
    if (_amsgrad) {
        delete [] v_hat;
    }
}

void APAM::
reset(void) {
    t = 0;
    memset(m, 0, num_param*sizeof(float));
    memset(v, 0, num_param*sizeof(float));
    if (_amsgrad) {
        memset(v_hat, 0, num_param*sizeof(float));
    }
}

void APAM::
unpack_w(float *w) {
    float *tw = NULL;
    int offset = 0;
    for (int i = 0; i < num_set; i++) {
        tw = static_cast<float*>(_model.parameters()[i].storage().data());
        memcpy(tw, w+offset, set_size[i]*sizeof(float));
        offset += set_size[i];
    }
}

void APAM::
pack_w(float *w) {
    float *tw = NULL;
    int offset = 0;
    for (int i = 0; i < num_set; i++) {
        tw = static_cast<float*>(_model.parameters()[i].storage().data());
        memcpy(w+offset, tw, set_size[i]*sizeof(float));
        offset += set_size[i];
    }
}

void APAM::
pack_g(float *g) {
    float *tg = NULL;
    int offset = 0;
    for (int i = 0; i < num_set; i++) {
        tg = static_cast<float*>(_model.parameters()[i].grad().storage().data());
        memcpy(g+offset, tg, set_size[i]*sizeof(float));
        offset += set_size[i];
    }
}

void APAM::
zero_grad(void) {
    float *tg = NULL;
    for (int i = 0; i < num_set; i++) {
        if (!_model.parameters()[i].grad().defined()) {
            continue;
        }
        tg = static_cast<float*>(_model.parameters()[i].grad().storage().data());
        memset(tg, 0, set_size[i]*sizeof(float));
    }
}

void APAM::
step(float *g, float *w) {
    
    // note two differences from the APAM paper: (1) the calculation of
    // lr; (2) the use of "+ _epsilon" in the denominator
    t++;
    double lr = _alpha * sqrt(1 - pow(_beta2,t)) / (1 - pow(_beta1,t));
    double one_minus_beta1 = 1 - _beta1;
    double one_minus_beta2 = 1 - _beta2;
    for (int i = 0; i < num_param; i++) {
        m[i] = _beta1 * m[i] + one_minus_beta1 * g[i];
        v[i] = _beta2 * v[i] + one_minus_beta2 * g[i] * g[i];
        if (_amsgrad) {
            v_hat[i] = std::max(v_hat[i], v[i]);
            w[i] -= lr * m[i] / (sqrt(v_hat[i]) + _epsilon);
        }
        else {
            w[i] -= lr * m[i] / (sqrt(v[i]) + _epsilon);
        }
    }
}

int APAM::
get_num_param(void) {
    return num_param;
}

//-----------------------------------------------------------------------------

template <typename DataLoader>
void train_master(Net& model,
                  torch::Device device,
                  DataLoader& dataloader,
                  size_t dataset_size,
                  APAM& optimizer,
                  float **gg,
                  float *w,
                  int N,
                  int maxiter,
                  int num_iter_per_epoch,
                  bool debug_comm,
                  bool debug_time,
                  bool debug_grad,
                  bool check_progress) {
    
    // mpi context
    int myrank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    // debug timing (will output timing information to file)
    clock_t time1, time2;
    int *elapse = NULL;
    int elapse_count = 0;
    if (debug_time) {
        time1 = clock();
        elapse = new int [(maxiter+nranks)*3];
    }
    
    
    // initiate nonblocking receives from all workers
    MPI_Request *recv_request = NULL;
    recv_request = new MPI_Request [nranks];
    for (int i = 0; i < nranks; i++) {
        if (i == ROOT) {
            recv_request[i] = MPI_REQUEST_NULL;
        }
        else {
            if (MPI_Irecv(gg[i], N, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD,
                          &recv_request[i]) != MPI_SUCCESS) {
                printf("master %d: Error in MPI_Irecv!", myrank); exit(1);
            }
        }
    }
    
    // print training set loss and accuracy (time consuming)
    if (check_progress) {
        printf("process %d: Train epoch 0 [0/%d]\n", myrank, maxiter);
        test(model, device, dataloader, dataset_size, "train set");
    }
    
    // debug gradient history (will output gradient history to file)
    int *which_worker = NULL;
    int *num_grad_this_wait = NULL;
    int num_wait = 0;
    if (debug_grad) {
        which_worker = new int [maxiter+nranks];
        num_grad_this_wait = new int [maxiter+nranks];
    }
    
    // prepare for while loop
    int num_active_workers = nranks - 1;
    int counter = 0;
    int nreceived;
    int *idx_of_received = NULL;
    idx_of_received = new int [nranks];
    MPI_Status *status_of_received = NULL;
    status_of_received = new MPI_Status [nranks];
    
    // while loop
    while (num_active_workers > 0) {
        
        // wait for new g
        if (MPI_Waitsome(nranks, recv_request, &nreceived, idx_of_received,
                         status_of_received) != MPI_SUCCESS) {
            printf("master %d: Error in MPI_Waitsome!", myrank); exit(1);
        }
        
        // debug timing
        if (debug_time) {
            time2 = clock();
            elapse[elapse_count++] = time2 - time1;
            time1 = time2;
        }
        
        // debug communication
        if (debug_comm) {
            printf("master %d: received %d new g from ranks [ ", myrank, nreceived);
            for (int i = 0; i < nreceived; i++) {
                printf("%d ", idx_of_received[i]);
            }
            printf("]\n");
            if (nreceived <= 0) {
                nreceived = 0;
            }
        }
        
        // debug gradient history
        if (debug_grad) {
            num_grad_this_wait[num_wait++] = nreceived;
            for (int i = 0; i < nreceived; i++) {
                which_worker[counter+i] = idx_of_received[i];
            }
        }
        
        // compute new w based on received g
        for (int i = 0; i < nreceived; i++) {
            
            // compute new w based on received g
            optimizer.step(gg[idx_of_received[i]], w);
            ++counter; // bookkeep the number of digested g in total
            
            // print training set loss and accuracy (time consuming)
            if (check_progress && counter % num_iter_per_epoch == 0) {
                optimizer.unpack_w(w);
                int epoch_number = counter / num_iter_per_epoch;
                printf("process %d: Train epoch %d [%d/%d]\n",
                       myrank, epoch_number, counter, maxiter);
                test(model, device, dataloader, dataset_size, "train set");
            }
        }
        
        // debug communication
        if (debug_comm) {
            printf("master %d: total number of digested g = %d\n", myrank, counter);
        }
        
        // debug timing
        if (debug_time) {
            time2 = clock();
            elapse[elapse_count++] = time2 - time1;
            time1 = time2;
        }
        
        // send new w to select workers and post new nonblocking receives
        for (int i = 0; i < nreceived; i++) {
            if (counter < maxiter) { // if not done
                
                // send new w to worker
                if (MPI_Send(w, N, MPI_FLOAT, idx_of_received[i], NOT_DONE,
                             MPI_COMM_WORLD) != MPI_SUCCESS) {
                    printf("master %d: Error in MPI_Send!", myrank); exit(1);
                }
                
                // post new nonblocking receive from worker
                if (MPI_Irecv(gg[idx_of_received[i]], N, MPI_FLOAT,
                              idx_of_received[i], MPI_ANY_TAG, MPI_COMM_WORLD,
                              &recv_request[idx_of_received[i]]) != MPI_SUCCESS) {
                    printf("master %d: Error in MPI_Irecv!", myrank); exit(1);
                }
            }
            else { // if done
                
                // send new w to worker. also signal termination
                if (MPI_Send(w, N, MPI_FLOAT, idx_of_received[i], DONE,
                             MPI_COMM_WORLD) != MPI_SUCCESS) {
                    printf("master %d: Error in MPI_Send!", myrank); exit(1);
                }
                
                // this worker is done
                num_active_workers--;
            }
        }
        
        // debug timing
        if (debug_time) {
            time2 = clock();
            elapse[elapse_count++] = time2 - time1;
            time1 = time2;
        }
    }
    
    // copy final w in the communication buffer to model
    optimizer.unpack_w(w);
    
    // debug timing: output timing information to file
    FILE *fp = NULL;
    if (debug_time) {
        fp = fopen("debug_time_outfile.txt", "w");
        for (int i = 0; i < elapse_count; i++) {
            fprintf(fp, "%d ", elapse[i]);
            if ((i+1)%3 == 0) {
                fprintf(fp, "\n");
            }
        }
        fclose(fp);
    }
    
    // debug gradient history: output gradient history to file
    if (debug_grad) {
        fp = fopen("debug_grad_outfile.txt", "w");
        int *which_worker_ptr = which_worker;
        for (int i = 0; i < num_wait; i++) {
            for (int j = 0; j < num_grad_this_wait[i]; j++) {
                fprintf(fp, "%d ", *which_worker_ptr++);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    // clean up
    if (debug_time) {
        delete [] elapse;
    }
    if (debug_grad) {
        delete [] which_worker;
        delete [] num_grad_this_wait;
    }
    delete [] recv_request;
    delete [] idx_of_received;
    delete [] status_of_received;
}

//-----------------------------------------------------------------------------

template <typename DataLoader>
void train_worker(Net& model,
                  torch::Device device,
                  DataLoader& dataloader,
                  size_t dataset_size,
                  APAM& optimizer,
                  float *g,
                  float *w,
                  int N,
                  bool debug_comm,
                  bool check_progress) {
    
    MPI_Status status;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // set the mode to train (affects dropout, batchnorm, etc)
    model.train();
    
    // while loop
    while (1) {
        
        // compute new g based on the current w
        for (auto& batch : dataloader) {
            
            // move data to device
            auto data = batch.data.to(device), target = batch.target.to(device);
            
            // use w in the communication buffer to set parameters in model
            optimizer.unpack_w(w);
            
            // reset gradient
            optimizer.zero_grad();
            
            // forward
            auto output = model.forward(data);
            auto loss = torch::nll_loss(output, target);
            
            // backward (compute gradient)
            loss.backward();
            
            // extract gradient to communication buffer g
            optimizer.pack_g(g);
            
            // ensure that the loop is iterated only once, because we need
            // only one random batch
            break;
        }
        
        // send new g to master
        if (MPI_Send(g, N, MPI_FLOAT, ROOT, NOT_DONE, MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            printf("worker %d: Error in MPI_Send!", myrank); exit(1);
        }
        
        // receive new w from master
        if (MPI_Recv(w, N, MPI_FLOAT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status)
            != MPI_SUCCESS) {
            printf("worker %d: Error in MPI_Recv!", myrank); exit(1);
        }
        
        // debug communication
        if (debug_comm) {
            printf("worker %d: received w from master. "
                   "status.MPI_SOURCE = %d, status.MPI_TAG = %d\n",
                   myrank, status.MPI_SOURCE, status.MPI_TAG);
        }
        
        // terminate iteration
        if (status.MPI_TAG == DONE) {
            break;
        }
    }
}

//-----------------------------------------------------------------------------

struct loss_acc{
    double loss, acc;
};

typedef struct  loss_acc Struct;


template <typename DataLoader>
Struct test(Net& model,
          torch::Device device,
          DataLoader& dataloader,
          size_t dataset_size,
          std::string dataset_name) {
    
    int myrank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    torch::NoGradGuard no_grad; // no gradient calculation needed
    model.eval(); // set the mode to eval (affects dropout, batchnorm, etc)
    double loss = 0;
    int64_t correct = 0;
    for (auto& batch : dataloader) {
        auto data = batch.data.to(device), target = batch.target.to(device);
        auto output = model.forward(data);
        loss += torch::nll_loss(output, target).template item<float>()
        * batch.data.size(0);
        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().template item<int64_t>();
    }
    loss /= dataset_size;
    double accuracy = 100. * correct / dataset_size;
   
    Struct l_a;
    l_a.loss = loss;
    l_a.acc = accuracy;

    return l_a;
}
//-----------------------------------------------------------------------------

int main(int argc, char **argv) {
    
    // mpi initialization
    MPI_Init(&argc, &argv);
    int myrank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // parameters
    double lr = 1e-4;
    size_t train_batch_size = 40;
    
    bool use_amsgrad = true;
    size_t test_batch_size = 1000;
    size_t pred_train_batch_size = 1000;
    size_t num_epochs = 200;
    size_t epoch = 0;
    bool debug_comm = false; // if true, debugging info will print to screen
    bool debug_time = false; // if true, debugging info will print to file
    
    bool debug_grad = false; // if true, debugging info will print to file
    
    bool timing = true; // if false, will compute training error each epoch
    
    // pytorch setup
    size_t seed = myrank * 100 + 20190830;  // seed
    torch::manual_seed(seed);
    torch::DeviceType device_type =         // device
    torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    Net model;                              // move net to device
    model.to(device);
    APAM optimizer(model, lr, use_amsgrad); // set optimizer
    
    // training dataset and data loader (each process has a copy)
    //
    // torch::data::datasets::FashionMNIST is not supported but we can
    // reuse torch::data::datasets::MNIST
    auto train_dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // preprocessing
    .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
    torch::data::make_data_loader<torch::data::samplers::RandomSampler>
    (std::move(train_dataset), train_batch_size); // RandomSampler will shuffle
    
    
    auto pred_train_dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // preprocessing
    .map(torch::data::transforms::Stack<>());
    const size_t pred_train_dataset_size = pred_train_dataset.size().value();
    auto pred_train_loader =
    torch::data::make_data_loader(std::move(pred_train_dataset), pred_train_batch_size);
    

    // testing data and data loader
    auto test_dataset = torch::data::datasets::
    MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
    torch::data::make_data_loader(std::move(test_dataset), test_batch_size);
    
    // other parameters and storage
    int num_iter_per_epoch = (int)ceil((double)train_dataset_size /
                                       train_batch_size);
    bool check_progress = timing ? false : true;
    int N = optimizer.get_num_param();
    float *g = NULL;   // gradient (one for each worker)
    float **gg = NULL; // gradient (nranks for the master)
    float *w = NULL;   // parameter array
    if (myrank != ROOT) {
        g = new float [N];
    }
    else {
        gg = new float* [nranks];
        for (int i = 0; i < nranks; i++) {
            gg[i] = new float [N];
        }
    }
    w = new float [N];
    
    FILE *fp = NULL;
    
    if (myrank == ROOT){
		
		std::ostringstream streamObj;
        streamObj << lr;
               
        printf("nworkers = %d\n", nranks - 1);
        std::cout<< "learning_reating="<< lr<<std::endl;
        std::cout<< "num_epochs="<< num_epochs<<std::endl;

        printf(     "epoch \t test_loss \t test_acc \t train_loss \t train_acc \t time_since_begin \n");
    }
    
    
    if (myrank == ROOT){
        // master and workers should use the same initialization
        // Pytorch default initialization
        optimizer.pack_w(w);
    }
    
    MPI_Bcast(w, N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    
    double total_test_time = .0;
    
    // training
    clock_t time_start = clock();
    while(epoch < num_epochs){
        if (myrank == ROOT) {
            train_master(model, device, *train_loader, train_dataset_size, optimizer,
                         gg, w, N, num_iter_per_epoch, num_iter_per_epoch, debug_comm, debug_time,
                         debug_grad,
                         check_progress);
        }
        else {
            train_worker(model, device, *train_loader, train_dataset_size, optimizer,
                         g, w, N, debug_comm, check_progress);
        }
        
        epoch++;
        
        // testing (done by only the master)
        if (myrank == ROOT && epoch%5 == 0){
            clock_t test_time0 = clock();
            
            Struct test_l_a  = test(model, device, *test_loader, test_dataset_size,  "test set");
               
            Struct train_l_a = test(model, device, *pred_train_loader, pred_train_dataset_size, "train set");
                
            printf(     "%d \t %g \t %g \t %g \t %g \t %g\n", epoch, test_l_a.loss, test_l_a.acc, train_l_a.loss, train_l_a.acc, (double)(clock() - time_start) / CLOCKS_PER_SEC );
            
            total_test_time += (double)(clock() - test_time0) / CLOCKS_PER_SEC;
            
        }
        
        MPI_Bcast(w, N, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    }
    
    clock_t time_end = clock();
    
    // print timing (done by only the master)
    if (timing && myrank == ROOT) {
        double time_elapsed = (double)(time_end - time_start) / CLOCKS_PER_SEC - total_test_time;
        printf("process %d: Training time %g seconds (with %d workers)\n\n",
               myrank, time_elapsed, nranks-1);
        fprintf(fp, "process %d: Training time %g seconds (with %d workers)\n\n",
        myrank, time_elapsed, nranks-1);
    }
    
    
    // clean up and return
    if (myrank != ROOT) {
        delete [] g;
    }
    else {
        fclose(fp);
        for (int i = 0; i < nranks; i++) {
            delete [] gg[i];
        }
        delete [] gg;
    }
    delete [] w;
    MPI_Finalize();
    return 0;
}

