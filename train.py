import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from unet_model import Unet2D

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from dataloaders import *

import torch.optim as optim
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import time
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(X_batch, Y_batch, optimizer, model, loss_fn, plot=True):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    if plot:
        plt.imshow(X_batch[0].detach().cpu().permute([1,2,0]))
        plt.show()
        plt.imshow(Y_batch[0].detach().cpu())
        plt.show()
        plt.imshow(outputs[0][1].detach().cpu())
        plt.show()
    print(loss.item())
    loss.backward()
    optimizer.step()

def train(model, num_classes, train_dl, val_dl, loss_fn, optimizer, epochs):
    print("train")

    for epoch in range(epochs):
        print("EPOCH", epoch)
        model.train()
        plot=False
        if epoch == 1:
            plot=True


        for X_batch, Y_batch in train_dl:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            train_step(X_batch, Y_batch, optimizer, model, loss_fn, plot)
            pass

def main():
    
    #HYPERPARAMETERS
    LEARNING_RATE = 3e-4

    # TRAIN TRANSFORMS

    tf = A.Compose([
        A.Normalize(mean=[0,0,0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_input_path = os.path.join("dataset", "train", "input")
    train_gt_path = os.path.join("dataset", "train", "gt")
    val_input_path = os.path.join("dataset", "val", "input")
    val_gt_path = os.path.join("dataset", "val", "gt")

    train_loader = get_dataloader(8, train_input_path, train_gt_path, transforms=tf, shuffle=True)
    val_loader = get_dataloader(8, val_input_path, val_gt_path)

    unet = Unet2D(3,2)
    unet.to(device)




    opt = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    
    train(unet, 2, train_loader, val_loader, loss_fn, opt, 10)



if __name__ == '__main__':
    main()




