import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':30}) 

plt.rc('xtick', labelsize=18)   # fontsize of the tick labels
plt.rc('ytick', labelsize=18)  

def read_time(dir0):
    with open(dir0) as f:
        lines = f.readlines()
        
    elements = lines[-2].split(' ')
    
    return [int(elements[5]),float(elements[2])/3600]
    
data_label=['#workers', 'Training Time (hour)']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)

def main():
    args = parser.parse_args()
    epochs = args.epochs

    dir0 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_2_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    dir1 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_6_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    dir2 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_11_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    dir3 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_async_SIZE_21_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'

    dir11 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_sync_SIZE_6_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    dir21 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_sync_SIZE_11_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    dir31 = './results/results_CPUimagenet32_wideresnet28_5_apam_comm_sync_SIZE_21_epochs_'+str(epochs)+'_bacthsize100_alpha0.001.txt'
    
    numworkers_async  = []
    train_times_async = []
    for dir_ in [dir0, dir1, dir2, dir3]:
        result = read_time(dir_)
        numworkers_async.append(result[0])
        train_times_async.append(result[1])

    numworkers_sync  = []
    train_times_sync = []
    for dir_ in [dir0, dir11, dir21, dir31]:
        result = read_time(dir_)
        numworkers_sync.append(result[0])
        train_times_sync.append(result[1])

    fig = plt.figure(figsize=(7, 3))
    name_list = ['1','5','10','20']
    x = np.arange(len(name_list))
    total_width, n = 0.6, 2
    width = total_width / n
    label_font=18
    plt.bar(x[:]-width/2-0.02, train_times_async[:], width=width, label='APAM',fc='r')
    plt.bar(x,[0,0,0,0], tick_label=name_list, width=0.2)
    plt.bar(x[:]+width/2+0.02, train_times_sync[:], width=width, label='sync method',fc='b')
    plt.legend(fontsize=label_font)
    plt.ylabel('times (hour)',fontsize=label_font)
    plt.xlabel('#worker process',fontsize=label_font)
       
    savename = 'Imagenet32_time_APAM_sync'
    plt.savefig('./pictures/'+savename+'.pdf',bbox_inches='tight',format='pdf')

#    plt.show()

if __name__ == '__main__':
    main()




