#include <iostream>
#include <fstream>
#include "matrices.h"
#include "util.h"
#include<cstdlib>
#include <random>
#include <thread>
#include "randsample.h"
#include <omp.h>
using namespace std;

void run_iteration_single_master(int* is_new_grad);

void run_iteration_multiple_master(int* is_new_grad);

void do_all_check();

int m, n, epoch = 0, maxepoch, n_test, epoch_report = 0;

// report all results every maxepoch_report
int maxepoch_report = 1;

int batch_sz, nthreads, nworkers, nmasters, total_iter = 0;
int total_grad_by_worker = 0, total_grad_digested = 0;

SpMat X, Xt, Xtest, Xtest_t;
Vector y, w, ytest;

Vector all_grad; 

int main(int argc, char *argv[]) {
    
    if(argc < 6 || argc > 7){
        cout << "Usage: *.exe train_data test_data batch_sz, maxepoch, nthreads, nmasters" << endl;
        return 1;
    }
    
    std::default_random_engine generator;
    generator.seed(20190830);
    
    std::normal_distribution<double> gdis(0,1); //mean=0, std=1
    
    string train_data_file_name, test_data_file_name;
    
    train_data_file_name = std::string( argv[1] );
    
    test_data_file_name = std::string( argv[2] );
    
    batch_sz = atoi( argv[3] );
    
    maxepoch = atoi( argv[4] );
    
    nthreads = atoi( argv[5] );
    
    nmasters = 1;
    
    if(argc == 7)
        nmasters = atoi( argv[6] );
    
    nworkers = nthreads - nmasters;
    
    
    loadLibSVM(X, y, train_data_file_name);
    
    loadLibSVM(Xtest, ytest, test_data_file_name);
    
    m = X.rows(); // number of features
    n = X.cols(); // number of train samples
    n_test = Xtest.cols(); // number of test samples
        
    Xt = X.transpose();
    Xtest_t = Xtest.transpose();
    
    int i;
    
    double *w_init = new double[m];
    
    for(int i = 0; i < m; i++)
        w_init[i] = gdis(generator);
    
    
    w.assign(w_init, w_init+m);
    
    delete []w_init;
    
    if(nworkers > 0){
        // assign two grad buffer to each worker
        double *all_grad_init = new double[2*m*nworkers];
        for(i = 0; i < 2*m*nworkers; i++)
            all_grad_init[i] = 0.0;
        all_grad.assign(all_grad_init, all_grad_init+2*m*nworkers);
        delete []all_grad_init;
    }
    
    int is_new_grad[2*nworkers];
    
    for(i = 0; i < 2*nworkers; i++)
        is_new_grad[i] = 0;
    
    double total_check_time = 0.;
    
    cout << "maxepoch = " << maxepoch << endl;
    
    printf("epoch \t loss \t train_acc \t test_acc\n");
    
    do_all_check();
    
    double start_time = get_wall_time();
    
    while(epoch < maxepoch){
    
        epoch_report = 0;
        
        for(i = 0; i < 2*nworkers; i++)
            is_new_grad[i] = 0;
    
# pragma omp parallel num_threads(nworkers+1)
    {
        if(nmasters == 1)
            run_iteration_single_master(is_new_grad);
        else
            run_iteration_multiple_master(is_new_grad);
    }
        
        epoch += maxepoch_report;
        
        double start_check_time = get_wall_time();
        
        do_all_check();
        
        total_check_time += get_wall_time() - start_check_time;
    }
    
    double end_time = get_wall_time();
    cout << "Total running time is " << end_time - start_time - total_check_time << endl;
    
    return 0;
    
} // end of main

//=====================================================================

void run_iteration_single_master(int* is_new_grad){
    
    int thread_id = omp_get_thread_num();
    
    std::default_random_engine generator;
    generator.seed(20190830 + 1000*rand()*thread_id);
    
    std::uniform_int_distribution<int> udis(0,n-1);
    
    std::normal_distribution<double> gdis(0,1); //mean=0, std=1
    
    int i, loc;
    
    double z_sample, coeff;
    
    double label_val, tmp, local_obj, Zero = .00000000001;
    
    int ngroups = n / batch_sz, smp_extra = n % batch_sz, group_id, st, ed;
    
    Vector local_grad(m, 0.), local_grad_col(m, 0.);
    
    if(thread_id == nworkers){
        
        int num_grad = 0, thread_search, found_new_grad = 0;
        
        Vector u(m, 0.), v(m, 0.), vhat(m, 0.);
        
        double alpha0 = 10, beta1 = 0.9, beta2 = 0.999, alpha, alpha_max = 0.01;
        
        while(epoch_report < maxepoch_report){
            
            total_iter++;
            
            alpha = alpha0/sqrt(total_iter);
            if(alpha > alpha_max)
                alpha = alpha_max;
            
            for(thread_search = 0; thread_search < 2*nworkers; thread_search++){
                found_new_grad = is_new_grad[thread_search];
                if(found_new_grad != 0){
                    total_grad_digested++;
                    break;
                }
            }
            
            if(found_new_grad != 0){
                // found a new sample grad
                
                for(i = 0; i < m; i++){
                    loc = thread_search*m;
                    local_grad[i] = all_grad[i + loc];
                }
                
                is_new_grad[thread_search] = 0;
            }
            
            else{
                // otherwise, compute a sample gradient
                
                for(i = 0; i < m; i++)
                    local_grad[i] = 0.0;
                
                num_grad++;
                group_id = rand() % ngroups;
                if(group_id < smp_extra){
                    st = group_id*(batch_sz+1);
                    ed = st + (batch_sz+1);
                }
                else{
                    st = group_id*batch_sz + smp_extra;
                    ed = st + batch_sz;
                }
                
                for(i = st; i < ed; i++){
                    z_sample = dot(Xt, w, i);
                    label_val = y[i];
                    tmp = z_sample * label_val;
                    if(tmp >= 0.){
                        tmp = exp(-tmp);
                        coeff = -(tmp/(1+tmp))*label_val;
                    }
                    else{
                        tmp = exp(tmp);
                        coeff = -label_val/(1+tmp);
                    }
                    trans_multiply_row(Xt, coeff, local_grad_col, i);
                    add(local_grad, local_grad_col);
                }
            }
            
            // update u, v, vhat, and w
            
            for(i = 0; i < m; i++){
                u[i] = beta1*u[i] + (1-beta1)*local_grad[i];
                v[i] = beta2*v[i] + (1-beta2)*local_grad[i]*local_grad[i];
                if(vhat[i] < v[i])
                    vhat[i] = v[i];
                
                if(vhat[i] > Zero)
                    w[i] -= alpha*u[i]/sqrt(vhat[i]);
            }
            
            if(total_iter%(n/batch_sz) == 0)
                epoch_report++;
                
        } // end of while in master part
    }
    
    
    else{
        
        while(epoch_report < maxepoch_report){ // worker nodes compute sample gradients
            
            group_id = rand()%ngroups;
            if(group_id < smp_extra){
                st = group_id*(batch_sz+1);
                ed = st + (batch_sz+1);
            }
            else{
                st = group_id*batch_sz + smp_extra;
                ed = st + batch_sz;
            }
            
            for(i = 0; i < m; i++){
                local_grad[i] = 0.0;
            }
            
            for(i = st; i < ed; i++){
                z_sample = dot(Xt, w, i);
                label_val = y[i];
                tmp = z_sample * label_val;
                if(tmp >= 0.){
                    tmp = exp(-tmp);
                    coeff = -(tmp/(1+tmp))*label_val;
                }
                else{
                    tmp = exp(tmp);
                    coeff = -label_val/(1+tmp);
                }
                
                trans_multiply_row(Xt, coeff, local_grad_col, i);
                add(local_grad, local_grad_col);
                
            }
            
            total_grad_by_worker++;
            
            if(is_new_grad[thread_id] == 0){
                loc = thread_id*m;
                for(i = 0; i < m; i++)
                    all_grad[loc+i] = local_grad[i];
                is_new_grad[thread_id] = 1;
                
            }
            else{
                loc = (thread_id+nworkers)*m;
                for(i = 0; i < m; i++)
                    all_grad[loc+i] = local_grad[i];
                is_new_grad[thread_id+nworkers] = 1;
            }
        }// end of while-loop in worker part
        
    }// end of if-else
    
} // end of run_iteration_single_master

//================================================================
//
void run_iteration_multiple_master(int* is_new_grad){
    
    int thread_id = omp_get_thread_num();
    
    std::default_random_engine generator;
    generator.seed(20190830 + 1000*rand()*thread_id);
    
    std::uniform_int_distribution<int> udis(0,n-1);
    
    std::normal_distribution<double> gdis(0,1); //mean=0, std=1
    
    int i, loc;
    
    double z_sample, coeff;
    
    double label_val, tmp, local_obj, Zero = .00000000001;
    
    int ngroups = n/batch_sz, smp_extra = n%batch_sz, group_id, st, ed;
    
    if(thread_id == nworkers){
        
        int num_grad = 0, thread_search, found_new_grad = 0;
        omp_set_nested(1);
        
        double alpha0 = 10., beta1 = 0.9, beta2 = 0.999, alpha, alpha_max = 0.01;
        
        Vector local_grad(m, 0.), u(m, 0.), v(m, 0.), vhat(m, 0.);
        
# pragma omp parallel num_threads(nmasters)
        {
            
            int master_id = omp_get_thread_num();
            
            int num_feature_per_master = m / nmasters, num_feature_extra = m % nmasters;
            int local_num_feature, local_feature_st, local_feature_ed;
            int local_num_smp, num_smp_extra, local_smp_st, local_smp_ed;
            int local_i;
            
            double nested_z_sample, nested_tmp, nested_label_val, nested_coeff;
            Vector nested_grad(m, 0.), nested_grad_col(m,0.);
            
            if(master_id < num_feature_extra){
                local_num_feature = num_feature_per_master + 1;
                local_feature_st = master_id*local_num_feature;
                local_feature_ed = local_feature_st + local_num_feature;
            }
            else{
                local_num_feature = num_feature_per_master;
                local_feature_st = master_id*local_num_feature + num_feature_extra;
                local_feature_ed = local_feature_st + local_num_feature;
            }
            
            while(epoch_report < maxepoch_report){
                
                if(master_id == 0){
                    total_iter++;
                    
                    alpha = alpha0/sqrt(total_iter);
                    if(alpha > alpha_max)
                        alpha = alpha_max;
                    
                    for(thread_search = 0; thread_search < 2*nworkers; thread_search++){
                        found_new_grad = is_new_grad[thread_search];
                        if(found_new_grad != 0){
                            total_grad_digested++;
                            loc = thread_search*m;
                            break;
                        }
                    }
                }
                
# pragma omp barrier
                
                if(found_new_grad != 0){
                    // found a new sample grad
                    
                    for(local_i = local_feature_st; local_i < local_feature_ed; local_i++)
                        local_grad[local_i] = all_grad[local_i + loc];
                    
                    if(master_id == 0)
                        is_new_grad[thread_search] = 0;
                }
                
                else{
                    
                    // otherwise, compute a sample gradient
                    
                    for(local_i = local_feature_st; local_i < local_feature_ed; local_i++)
                        local_grad[local_i] = 0.0;
                    
                    if(master_id == 0){
                        num_grad++;
                        group_id = rand()%ngroups;
                        if(group_id < smp_extra){
                            st = group_id*(batch_sz+1);
                            ed = st + (batch_sz+1);
                        }
                        else{
                            st = group_id*batch_sz + smp_extra;
                            ed = st + batch_sz;
                        }
                    }
                    
# pragma omp barrier
                    
                    local_num_smp = (ed - st) / nmasters;
                    num_smp_extra = (ed - st) % nmasters;
                    
                    if(master_id < num_smp_extra){
                        local_num_smp++;
                        local_smp_st = st + master_id*local_num_smp;
                        local_smp_ed = local_smp_st + local_num_smp;
                    }
                    else{
                        local_smp_st = st+ master_id*local_num_smp + num_smp_extra;
                        local_smp_ed = local_smp_st + local_num_smp;
                    }
                    
                    for(local_i = 0; local_i < m; local_i++)
                        nested_grad[local_i] = 0.0;
                    
                    for(local_i = local_smp_st; local_i < local_smp_ed; local_i++){
                        nested_z_sample = dot(Xt, w, local_i);
                        nested_label_val = y[local_i];
                        nested_tmp = nested_z_sample * nested_label_val;
                        if(nested_tmp >= 0.){
                            nested_tmp = exp(-nested_tmp);
                            nested_coeff = -(nested_tmp/(1+nested_tmp))*nested_label_val;
                        }
                        else{
                            nested_tmp = exp(nested_tmp);
                            nested_coeff = -nested_label_val/(1+nested_tmp);
                        }
                        
                        trans_multiply_row(Xt, nested_coeff, nested_grad_col, local_i);
                        add(nested_grad, nested_grad_col);
                    }
                    
# pragma omp critical
                    add(local_grad, nested_grad);
                    
                }
                
                
# pragma omp barrier
                
                // update u, v, vhat, and w
                
                for(local_i = local_feature_st; local_i < local_feature_ed; local_i++){
                    u[local_i] = beta1*u[local_i] + (1-beta1)*local_grad[local_i];
                    v[local_i] = beta2*v[local_i] + (1-beta2)*local_grad[local_i]*local_grad[local_i];
                    if(vhat[local_i] < v[local_i])
                        vhat[local_i] = v[local_i];
                    
                    if(vhat[local_i] > Zero)
                        w[local_i] -= alpha*u[local_i]/sqrt(vhat[local_i]);
                }
                
                if(master_id == 0){
                    
                    if(total_iter%(n/batch_sz) == 0)
                        epoch_report++;
                    
                } // end of iter++ region
# pragma omp barrier
            } // end of while
            
        } // end of nested parallel region
        
    }
    
    
    else{
        Vector local_grad(m, 0.), local_grad_col(m, 0.);
        
        while(epoch_report < maxepoch_report){ // worker nodes compute sample gradients
            
            group_id = rand()%ngroups;
            if(group_id < smp_extra){
                st = group_id*(batch_sz+1);
                ed = st + (batch_sz+1);
            }
            else{
                st = group_id*batch_sz + smp_extra;
                ed = st + batch_sz;
            }
            
            for(i = 0; i < m; i++){
                local_grad[i] = 0.0;
            }
            
            for(i = st; i < ed; i++){
                z_sample = dot(Xt, w, i);
                label_val = y[i];
                tmp = z_sample * label_val;
                if(tmp >= 0.){
                    tmp = exp(-tmp);
                    coeff = -(tmp/(1+tmp))*label_val;
                }
                else{
                    tmp = exp(tmp);
                    coeff = -label_val/(1+tmp);
                }
                
                trans_multiply_row(Xt, coeff, local_grad_col, i);
                add(local_grad, local_grad_col);
            }
            
            total_grad_by_worker++;
            
            if(is_new_grad[thread_id] == 0){
                loc = thread_id*m;
                for(i = 0; i < m; i++)
                    all_grad[loc+i] = local_grad[i];
                is_new_grad[thread_id] = 1;
                
            }
            else{
                loc = (thread_id+nworkers)*m;
                for(i = 0; i < m; i++)
                    all_grad[loc+i] = local_grad[i];
                is_new_grad[thread_id+nworkers] = 1;
            }
        }
        
    }// end of if-else
} // end of run_iteration_multiple_master

void do_all_check(){
    
    double obj = 0., tmp;
    int correct_num = 0, correct_num_test = 0, i;
    Vector z(n, 0.), z_test(n_test, 0.);

    multiply(Xt, w, z); // z = X'*w;
    
    for(i = 0; i < n; i++){
        tmp = y[i]*z[i];
        if(tmp >= 0.){
            correct_num++;
            obj += log(1+exp(-tmp));
        }
        else{
            obj += log(1+exp(tmp)) - tmp;
        }
    }
    
    obj /= n;
    
    multiply(Xtest_t, w, z_test);
    for(i = 0; i < n_test; i++){
        tmp = ytest[i]*z_test[i];
        if(tmp >= 0.)
            correct_num_test++;
    }
    
    // print out results
    
    cout << epoch << "\t" << obj << "\t" << (correct_num * 1.0)/n << "\t" << (correct_num_test * 1.0)/n_test << endl;
    
}
