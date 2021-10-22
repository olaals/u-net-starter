import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from unet_model import Unet2D

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn

from dataloaders import *

import torch.optim as optim
import torch.nn.functional as F
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

def save_images(X_batch, Y_batch, outputs, epoch, save_dir):
    fig,ax = plt.subplots(1,4)
    ax[0].set_axis_off()
    ax[0].set_title("input image")
    ax[0].imshow(X_batch[0].detach().cpu().permute([1,2,0]))
    ax[1].set_axis_off()
    ax[1].set_title("ground truth")
    ax[1].imshow(Y_batch[0].detach().cpu())
    ax[2].set_axis_off()
    ax[2].set_title("BG pred")
    ax[2].imshow(outputs[0][0].detach().cpu()) # class 0: background pred
    ax[3].set_axis_off()
    ax[3].set_title("bunny pred")
    ax[3].imshow(outputs[0][1].detach().cpu()) # class 1: bunny pred
    fig.savefig(os.path.join(save_dir, "epoch_"+format(epoch, "02d")+".png"))

def train_step(X_batch, Y_batch, optimizer, model, loss_fn):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = loss_fn(outputs, Y_batch)
    #print(loss.item())
    loss.backward()
    optimizer.step()
    return outputs

def train(model, num_classes, train_dl, val_dl, loss_fn, optimizer, epochs):

    for epoch in range(epochs):
        print("EPOCH", epoch)
        model.train()
        savefig=True
        for X_batch, Y_batch in tqdm(train_dl):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = train_step(X_batch, Y_batch, optimizer, model, loss_fn)
            if savefig:
                save_images(X_batch, Y_batch, outputs, epoch, "plots")
                savefig=False


def main():
    
    #HYPERPARAMETERS
    LEARNING_RATE = 3e-4
    CR_ENTR_WEIGHTS = torch.tensor([0.3,0.7]).to(device)
    BATCH_SIZE = 8

    
    DS_LENGTH = 50 #set number of images that is loaded and used for training (max = 400)
    # if training on GPU, set to 400, else set lower since training on CPU is slow
    
    
    

    # TRAIN TRANSFORMS

    tf = A.Compose([
        A.Normalize(mean=[0,0,0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_input_path = os.path.join("dataset", "train", "input")
    train_gt_path = os.path.join("dataset", "train", "gt")
    val_input_path = os.path.join("dataset", "val", "input")
    val_gt_path = os.path.join("dataset", "val", "gt")

    train_loader = get_dataloader(BATCH_SIZE, train_input_path, train_gt_path, DS_LENGTH, transforms=tf, shuffle=True)
    #validation set is unused, but can be loaded with:
    val_loader = get_dataloader(BATCH_SIZE, val_input_path, val_gt_path) 

    unet = Unet2D(3,2)
    unet.to(device)




    opt = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=CR_ENTR_WEIGHTS)

    
    train(unet, 2, train_loader, val_loader, loss_fn, opt, 10)



if __name__ == '__main__':
    main()




