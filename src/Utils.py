# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:07:34 2020

@author: 11028434
"""
import torch
import torchvision
import torchvision.transforms as transforms

def Get_Cifar():
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    
    return trainset,testset


def DataLoader(train_set,test_set,cuda_batch_size,SEED):
    
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

    return train_loader, test_loader

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device
