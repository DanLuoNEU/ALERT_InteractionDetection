# Train Script of Interaction Detection, for ALERT project
# Dan and Tim, 11/27/2019
########## IMPORT ##########
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms

from datasets.AVA import ava_dataset
from networks.pytorch_i3d import InceptionI3d
########## IMPORT END ##########

########## ARGUMENT PARSER ##########
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default='3', help=" GPU id to train")
parser.add_argument('-batch_size', type=int, default=4, help=" batch_size")
parser.add_argument('-num_workers', type=int, default=4, help=" dataset parallel")
parser.add_argument('-max_steps', type=int, default=64e1, help="Max steps for the training")
parser.add_argument('-mode', type=str, default='rgb', help="rgb, flow, pose, two, three")
parser.add_argument('-data_root', type=str, default='/data/truppr/AVA/', help="/path/to/dataset")
parser.add_argument('-save_dir', type=str, default='exp/', help="/path/to/save_model")
parser.add_argument('-save_int', type=int, default=20, help="Itervals for saving model")

args = parser.parse_args()
########## ARGUMENT PARSER END ##########

########## CONFIGURATION ##########
gpu = torch.device(f"cuda:{args.gpu}")

init_lr = 0.01
alpha, beta = 0.5, 0.5
num_steps_per_update = 4 # accum gradient
########## CONFIGURATION END ##########

def train(model,optimizer,lr_sched,train_DL,steps):
    """ Train step
    """
    model.train()

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    num_iter = 0
    optimizer.zero_grad()

    start =time.time()
    # Iterate over data.
    for inputs, labels in train_DL:
        # print(num_iter)
        num_iter += 1
        # Wrap inputs and labels in Variable
        # inputs, labels = data
        inputs = Variable(inputs.cuda(device=gpu))
        labels = Variable(labels.cuda(device=gpu))
        t = inputs.size(2)

        # Forward the model
        per_frame_logits = model(inputs)
        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
        # compute localization loss
        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        tot_loc_loss += loc_loss.item()
        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.item()
        # Combine loss
        loss = (alpha*loc_loss + beta*cls_loss)
        tot_loss += loss.item()
        
        loss.backward()

        if num_iter == num_steps_per_update:
            steps += 1
            # print(steps)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()
            # unit of step => batch_size*num_steps_per_update = 16 samples
            # save_int = unit of step*args.save_int = 320 samples
            if steps % args.save_int == 0:
                unit = args.save_int*num_steps_per_update
                print(f'Train| Loc Loss: {tot_loc_loss/unit:.4f}| Cls Loss: {tot_cls_loss/unit:.4f}| Tot Loss: {tot_loss/unit:.4f}|{(time.time()-start):.0f} s')
                # save model
                save_path = args.save_dir+args.mode
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_name = save_path+'/'+args.mode+str(steps).zfill(6)+'.pth'
                torch.save(model.state_dict(), save_name)
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
    
    return steps


def eval(model,optimizer,val_DL):
    """ Validation step
    """
    model.eval()

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0
    num_iter = 0
    optimizer.zero_grad()
            
    start = time.time()
    # Iterate over data.
    for inputs, labels in val_DL:
        num_iter += 1
        # wrap them in Variable
        # inputs, labels = data
        inputs = Variable(inputs.cuda(device=gpu))
        labels = Variable(labels.cuda(device=gpu))
        t = inputs.size(2)

        per_frame_logits = model(inputs).cuda(device=gpu)
        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
        # compute localization loss
        loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
        tot_loc_loss += loc_loss.item()
        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
        tot_cls_loss += cls_loss.item()
        # loss per update
        loss = alpha*loc_loss + beta*cls_loss
        tot_loss += loss.item()

    print(f'Val| Loc Loss: {tot_loc_loss/num_iter:.4f}| Cls Loss: {tot_cls_loss/num_iter:.4f}| Tot Loss: {tot_loss/num_iter:.4f}|{(time.time()-start):.0f} s')


def main():
    # Setups
    ## Set up Dataset
    dataset = ava_dataset(root_path=args.data_root, split='train', mode=args.mode, seq_len=64)
    dataloader = tud.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ava_dataset(root_path=args.data_root, split='valid', mode=args.mode, seq_len=64)
    val_dataloader = tud.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    ## Set up Model
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(dataset.num_classes)
    i3d.cuda(device=gpu)
    # i3d = nn.DataParallel(i3d)
    ## Set up Optimizer
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    
    since = time.time()
    print(time.asctime(time.localtime(since)))
    steps = 0
    # train it
    while steps < args.max_steps:#for epoch in range(num_epochs):
        print(f'Step {steps}/{int(args.max_steps)}')

        steps = train(i3d, optimizer, lr_sched, dataloaders['train'], steps)
        # eval interval=> unit of step*args.save_int = 320 samples
        eval(i3d, optimizer, dataloaders['val'])

if __name__ == '__main__':
    main()
