import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def read_datasets(dataset_name, data_dir=None,device='cpu'):
    if data_dir==None:
        data_dir = './data/' + dataset_name + '/'
        
#    if dataset_name in ['MNIST']:
#        train_dataset = datasets.MNIST( data_dir, train=True, download=True,
#                             transform= transforms.Compose([
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.1307,), (0.3081,)) ]))
#
#        test_dataset =  datasets.MNIST( data_dir, train=False, download=True,
#                             transform= transforms.Compose([
#                                transforms.ToTensor(),
#                                transforms.Normalize( (0.1307,), (0.3081,)) ]))
#        return  train_dataset, test_dataset
 
    if dataset_name in ['cifar10']:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
                
        transform_test = transforms.Compose([transforms.ToTensor(), normalize,])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
 
        return train_dataset, test_dataset
    
    if dataset_name in ['CINIC-10']: # 32*32
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)
 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize
        ])
        valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])
      
    if dataset_name in ['imagenet32']:
    
        mean_image = np.load(data_dir+'image_mean.npy')
     
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            MoveTransformation(torch.tensor(mean_image,dtype=torch.float32)),
        ])
        
        valid_transform = transforms.Compose([transforms.ToTensor(), MoveTransformation(torch.tensor(mean_image,dtype=torch.float32)),])
        
    if dataset_name in ['CINIC-10', 'imagenet32']:
        # Data loading code
        train_dir = data_dir + 'train'
        test_dir = data_dir + 'test'
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        test_dataset = datasets.ImageFolder(test_dir, valid_transform)
     
        return train_dataset, test_dataset

class MoveTransformation(torch.nn.Module):
    ### only for imagenet32.
    """Move a tensor image with a mean_variabme computed
    offline.
    Given mean_variabme, will subtract mean_variabme from it.

    Args:
        mean_variabme (Tensor): tensor [D], D = C x H x W
    """

    def __init__(self, mean_variabme):
        super().__init__()
        self.mean_variabme = mean_variabme

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be whitened.

        Returns:
            Tensor: Moved image.
        """
        return tensor - self.mean_variabme


    def __repr__(self):
        format_string = self.__class__.__name__ + '(mean_variabme='
        format_string += ( str(self.mean_variabme.tolist()) + ')')
        return format_string
