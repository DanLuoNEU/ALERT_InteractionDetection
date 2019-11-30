# Dataloader for AVA dataset
# Dan and Tim, 11/27/2019
########## IMPORT ##########
import os
from os import listdir
from os.path import isfile, join

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}, '../utils'))

import csv
import cv2
import sys
import json
import time
import pickle
import random
import numpy as np

import torch
import torch.utils.data as tud
import torchvision.transforms as transforms


########## IMPORT END ##########
def load_rgb_frames(image_dir, vid, start, num=64):
    """ Load Cropped RGB images from the image_dir
    !!! Suppose the rgb files have been preprocessed to 224 x 224 !!! 
    """
    frames = []

    for i in range(start, start+num):
        # Read RGB image
        # TODO: change the path to rgb files
        img = cv2.imread('/path/to/images')[:, :, [2, 1, 0]] # BGR->RGB
        w,h,c = img.shape
        ######
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        #######
        img = (img/255.)*2 - 1
        
        frames.append(img)

    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
    """ Load optical flow files from image_dir
    !!! Suppose the optical flow files are cropped to 224 x 224 !!!
    """
    frames = []
    for i in range(start, start+num):
        # TODO: change the path to the optical flow files
        with open('/path/to/optical_flow_file', 'rb') as f:
            of = pickle.load(f)
    
        c,w,h = of.shape
        
        imgx = of[0,:,:]
        imgy = of[1,:,:] 
        # TODO: the optical flow files need to be cropped and then resized
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])

        frames.append(img)
    
        return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=2):
    """ Use the split_file to read all annotations we need
    """
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < 66:
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
    
    return dataset


def actions(act):
    if act == "46":   return "push" # 122(person) + 56(object)
    elif act == "36": return "pickup" # 452 samples


class ava_dataset(tud.Dataset):
    """ Dataset of customed AVA dataset
    
    """
    
    def __init__(self, root_path, split, seq_len=64, mode):
        """ Dataset Initialization

        Arguments:
            root_path: /path/to/dataset_root
            split: 'train', 'valid'
            seq_len: T >=65
        """
        # TODO: 
        self.samples = make_dataset(split, root_path, mode)
        self.mode = mode
        self.root = root
        self.seq_len = seq_len

        if split == 'train':
            self.transform = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
        elif split == 'valid' or split = 'test':
            self.transform = transforms.Compose([videotransforms.RandomCrop(224)])
                                
        
        print(f">====== AVA {len(self.samples)} samples Loaded! ======<")


    def __getitem__(self, index):
        """ Get one item from the dataset, suppose the data has been preprocessed
        
        Arguments:
            index: id of one sample
        Return:
            data: C x T x H x W, Torch.FloatTensor 
                    from numpy.ndarray( T x H x W x C) to torch.FloatTensor ===== torch.from_numpy(pic.transpose([3,0,1,2]))
            labels: num_classes x num_frames
        """
        vid, label, _, nf = self.samples[index]
        start_f = random.randint(1, nf-self.seq_len-1)

        if self.mode == 'rgb':
            inputData = load_rgb_frames(self.root, vid, start_f, self.seq_len)
        else:
            inputData = load_flow_frames(self.root, vid, start_f, self.seq_len)
        
        ####### Data Augmentation
        # # Not sure if should use here
        # inputData = self.transforms(inputData)
        #######
        inputData = torch.from_numpy(inputData.transpose([3,0,1,2]))
        gt = torch.from_numpy(label[:, start_f:start_f+self.seq_len])

        return inputData, gt

    def __len__(self):
        """ Return number of samples
        """
        return len(self.samples)


def main():
    dataset = ava_dataset('/data/Dan/AVA/', 'train')
    print("Well Done!")

if __name__ == "__main__":
    main()