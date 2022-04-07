import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':30}) 

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)  
            
data_label=['epoch', 'train_loss', 'train_acc_top1', 'train_acc_top5', 'test_loss', 'test_acc_top1', 'test_acc_top5', 'time_since_begin(without testing)', 'time_since_begin(with testing)']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--log_per_epoch', type=int, default=2)

def main():
    args = parser.parse_args()
    Epoch = args.epochs
    epoch_recode = args.log_per_epoch

    dir0 = './results/results_GPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_2_epochs_'+str(Epoch)+'_bacthsize100_alpha0.001.txt'
    dir1 = './results/results_GPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_3_epochs_'+str(Epoch)+'_bacthsize100_alpha0.001.txt'
    dir2 = './results/results_GPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_6_epochs_'+str(Epoch)+'_bacthsize100_alpha0.001.txt'
    dir3 = './results/results_GPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_11_epochs_'+str(Epoch)+'_bacthsize100_alpha0.001.txt'
    max_rows = 2 + Epoch//epoch_recode

    results_0 = np.loadtxt(dir0,skiprows=1, max_rows=max_rows)
    
    results_1 = np.loadtxt(dir1,skiprows=1, max_rows=max_rows)
    results_2 = np.loadtxt(dir2,skiprows=1, max_rows=max_rows)
    results_3 = np.loadtxt(dir3,skiprows=1, max_rows=max_rows)

    linewidth = 1.5
    markevery1 = 1
    markevery2 = 1
    markevery3 = 1
    markevery4 = 1

    plt.figure(figsize=(6, 4))
    plt.plot(results_0[:,0], results_0[:,2]/100, 'k-^', markevery=markevery1, linewidth=linewidth, label='APAM1')
    plt.plot(results_1[:,0], results_1[:,2]/100, 'r-*', markevery=markevery2, linewidth=linewidth, label='APAM2')
    plt.plot(results_2[:,0], results_2[:,2]/100, 'r-s', markevery=markevery3, linewidth=linewidth, label='APAM5')
    plt.plot(results_3[:,0], results_3[:,2]/100, 'r-p', markevery=markevery4, linewidth=linewidth, label='APAM10')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('training accuracy', fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, Epoch)

    savename = 'Imagenet32_train_acc1'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(results_0[:,0], results_0[:,3]/100, 'k-^', markevery=markevery1, linewidth=linewidth, label='APAM1')
    plt.plot(results_1[:,0], results_1[:,3]/100, 'r-*', markevery=markevery2, linewidth=linewidth, label='APAM2')
    plt.plot(results_2[:,0], results_2[:,3]/100, 'r-s', markevery=markevery3, linewidth=linewidth, label='APAM5')
    plt.plot(results_3[:,0], results_3[:,3]/100, 'r-p', markevery=markevery4, linewidth=linewidth, label='APAM10')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('training top5 accuracy ', fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, Epoch)

    savename = 'Imagenet32_train_acc5'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()


    plt.figure(figsize=(6, 4))
    plt.plot(results_0[:,0], results_0[:,5]/100, 'k-^', markevery=markevery1, linewidth=linewidth, label='APAM1')
    plt.plot(results_1[:,0], results_1[:,5]/100, 'r-*', markevery=markevery2, linewidth=linewidth, label='APAM2')
    plt.plot(results_2[:,0], results_2[:,5]/100, 'r-s', markevery=markevery3, linewidth=linewidth, label='APAM5')
    plt.plot(results_3[:,0], results_3[:,5]/100, 'r-p', markevery=markevery4, linewidth=linewidth, label='APAM10')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('testing accuracy', fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, Epoch)
    savename = 'Imagenet32_test_acc1'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()

     
    plt.figure(figsize=(6, 4))
    plt.plot(results_0[:,0], results_0[:,6]/100, 'k-^', markevery=markevery1, linewidth=linewidth, label='APAM1')
    plt.plot(results_1[:,0], results_1[:,6]/100, 'r-*', markevery=markevery2, linewidth=linewidth, label='APAM2')
    plt.plot(results_2[:,0], results_2[:,6]/100, 'r-s', markevery=markevery3, linewidth=linewidth, label='APAM5')
    plt.plot(results_3[:,0], results_3[:,6]/100, 'r-p', markevery=markevery4, linewidth=linewidth, label='APAM10')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('testing top5 accuracy ', fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, Epoch)
    savename = 'Imagenet32_test_acc5'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()
    
if __name__ == '__main__':
    main()
