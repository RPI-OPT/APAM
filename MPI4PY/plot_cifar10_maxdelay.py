import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)

data_label=['epoch number', 'test_loss', 'test_acc', 'train_loss', 'train_acc', 'time_since_begin(without testing)', 'time_since_begin(with tetsing)',]

def read_results(dir, epochs=200, epoch_recode=10):
    max_rows = epochs//epoch_recode
    return np.loadtxt(dir,skiprows=1, max_rows=max_rows)
  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400)

def main():
    args = parser.parse_args()
    epochs = args.epochs

    maxdelay = [1, 6, 11, 21, 51, 101, 201, 401]
    
    filename0 = './results/results_cifar10_allcnn_apam_epochs'+str(epochs)+'_bacthsize100_MaxDelay' 
    filename1 = '.txt'

    results=[None for i in range(len(maxdelay))]
    for i in range(len(maxdelay)):
        results[i] = read_results(filename0+str(maxdelay[i])+filename1, epochs=epochs, epoch_recode=1)

    linewidth = 1
    plt.figure(figsize = (6, 4))
    # plt.subplot(121)
    plt.plot(results[0][:,0], results[0][:,4]/100, 'k-^', markevery=20, linewidth=linewidth, label='Max Delay=0')
    plt.plot(results[1][:,0], results[1][:,4]/100, '-p',  markevery=25, linewidth=linewidth, label='Max Delay=5')
    plt.plot(results[2][:,0], results[2][:,4]/100, '-*',  markevery=35, linewidth=linewidth, label='Max Delay=10')
    plt.plot(results[3][:,0], results[3][:,4]/100, '-s',  markevery=22, linewidth=linewidth, label='Max Delay=20')
    plt.plot(results[4][:,0], results[4][:,4]/100, '-o',  markevery=26, linewidth=linewidth, label='Max Delay=50')
    plt.plot(results[5][:,0], results[5][:,4]/100, '-h',  markevery=51, linewidth=linewidth, label='Max Delay=100')
    plt.plot(results[6][:,0], results[6][:,4]/100, '-X',  markevery=57, linewidth=linewidth, label='Max Delay=200')
    plt.plot(results[7][:,0], results[7][:,4]/100, '-8',  markevery=81, linewidth=linewidth, label='Max Delay=400')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('training accuracy', fontsize=18)
    plt.legend(fontsize=12)
    plt.ylim(0.20, 1.03)
    plt.xlim(0, epochs)
    savename = 'Artifical_delay_train_acc'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(results[0][:,0], results[0][:,2]/100, 'k-^', markevery=20, linewidth=linewidth, label='Max Delay=0')
    plt.plot(results[1][:,0], results[1][:,2]/100, '-p', markevery=25, linewidth=linewidth, label='Max Delay=5')
    plt.plot(results[2][:,0], results[2][:,2]/100, '-*', markevery=35, linewidth=linewidth, label='Max Delay=10')
    plt.plot(results[3][:,0], results[3][:,2]/100, '-s', markevery=22, linewidth=linewidth, label='Max Delay=20')
    plt.plot(results[4][:,0], results[4][:,2]/100, '-o', markevery=26, linewidth=linewidth, label='Max Delay=50')
    plt.plot(results[5][:,0], results[5][:,2]/100, '-h', markevery=51, linewidth=linewidth, label='Max Delay=100')
    plt.plot(results[6][:,0], results[6][:,2]/100, '-X', markevery=57, linewidth=linewidth, label='Max Delay=200')
    plt.plot(results[7][:,0], results[7][:,2]/100, '-8', markevery=81, linewidth=linewidth, label='Max Delay=400')
    plt.xlabel('epoch number', fontsize=18)
    plt.ylabel('testing accuracy', fontsize=18)
    plt.legend(fontsize=12)
    plt.ylim(0.20, 0.92)
    plt.xlim(0, epochs)
    savename = 'Artifical_delay_test_acc'
    plt.savefig('./pictures/'+savename+'.pdf', bbox_inches='tight', format='pdf')
#    plt.show()

if __name__ == '__main__':
    main()
