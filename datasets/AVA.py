# Dataloader for AVA dataset
# Dan and Tim, 11/27/2019
########## IMPORT ##########
import os
from os import listdir
from os.path import isfile, join

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/../utils")

import csv
import cv2
import json
import time
import pickle
import random
import numpy as np

import torch
import torch.utils.data as tud
import torchvision.transforms as transforms

import videotransforms
########## IMPORT END ##########


class ava_dataset(tud.Dataset):
    """ Dataset of customed AVA dataset
    
    """
    def __init__(self, root_path='/data/truppr/AVA', split='train', mode='rgb', seq_len=64):
        """ Dataset Initialization

        Arguments:
            root_path: /path/to/dataset_root
            split: 'train', 'valid', 'test'
            mode: input stream type
            seq_len: T 
        """
        ''' new action_id for customed actions from AVA 
        # 0 -> 36, "push"   # 122(person) + 56(object)
        # 1 -> 46, "pickup" # 452 samples
        # 2 -> 11, "sit"
        # 3 -> 12, "stand"
        # 4 -> 47, "put down"
        # 5 -> 65, "give"
        # 6 -> 78, "take"
        '''
        self.actions = ["36", "46", "11", "12", "47", "65", "78"]
        self.num_classes = len(self.actions)

        self.root = root_path
        self.split = split
        self.mode = mode
        self.seq_len = seq_len
        self.make_dataset()
        # Build up the transform for Data Augmentation
        if split == 'train':
            self.transform = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
        elif split == 'valid' or split == 'test':
            self.transform = transforms.Compose([videotransforms.RandomCrop(224)])
                                
        print(f">====== AVA {split}({len(self.dataset)}) Samples Loaded! ======<")


    def make_dataset(self):
        """ Use the split_file to read all annotations we need
        """
        dataset = []
        # CSV data: num_samples long list, 
        #           [vedio_id,middle_frame_timestamp,x0,y0,x1,y1,action_id,person_id], 
        #           person_box is normalized wrt frame size
        split_file = f"{self.root}/annotations/ava_random_{self.split}_truppr_v7class.csv"
        with open(split_file, 'r') as f:
            data = list(csv.reader(f))

        for vid in data:
            vid_path = f"{self.root}/streams/{vid[0]}_{vid[1]}_{vid[6]}_{vid[7]}"
            if not os.path.exists(vid_path):
                continue
            num_frames = len(os.listdir(vid_path+f'/{self.mode}'))
            if self.mode == 'flow':
                num_frames = num_frames-1
            # Action Length is at least 66
            if num_frames < 66:
                print(vid[0], vid[1], "is NOT long enough!!! ")
                continue

            label = np.zeros((self.num_classes, num_frames), np.float32)
            ''' # Normal way to deal with the data, 0 and 1 for the same action
            fps = num_frames/data[vid]['duration']
            for ann in data[vid]['actions']:
                for fr in range(0,num_frames,1):
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], fr] = 1 # binary classification
            '''
            label[self.actions.index(vid[6]), :] = 1

            dataset.append((vid, label, num_frames))
    
        self.dataset = dataset


    def load_rgb_frames(self, image_dir, vid, start, num=64):
        """ Load Cropped RGB images from the image_dir
        !!! Suppose the rgb files have been preprocessed to 224 x 224 !!! 
        rgb directory in form of '0000001.jpg', starts from 1
        """
        frames = []

        for i in range(start+1, start+num+1):
            # Read RGB image
            img = cv2.imread(image_dir+"/"+str(i).zfill(7)+'.jpg')[:, :, [2, 1, 0]] # BGR->RGB
            """ Test if the input is right
            # cv2.imshow('img',cv2.imread(image_dir+"/"+str(i).zfill(7)+'.jpg'))
            # k = cv2.waitKey(0)
            # if k ==27:
            #     cv2.destroyAllWindows()
            #     exit(0)
            """
            w,h,c = img.shape
            ######
            # if w < 226 or h < 226:
            #     d = 226.-min(w,h)
            #     sc = 1+d/min(w,h)
            #     img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            if w < 224 or h < 224:
                d = 224.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            #######
            img = (img/255.)*2 - 1
        
            frames.append(img)

        return np.asarray(frames, dtype=np.float32)


    def load_flow_frames(self, image_dir, vid, start, num=64):
        """ Load optical flow files from image_dir, crop and resize
        !!! Suppose the optical flow files are cropped to 224 x 224 !!!
        flow files in form of '0.pkl', starts from 0
        """
        frames = []
        for i in range(start, start+num):
            # TODO: change the path to the optical flow files
            with open(image_dir+"/"+str(i)+".pkl", 'rb') as f:
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


    def __getitem__(self, index):
        """ Get one item from the dataset, suppose the data has been preprocessed
        
        Arguments:
            index: id of one sample
        Return:
            data: C x T x H x W, Torch.FloatTensor 
                    from numpy.ndarray( T x H x W x C) to torch.FloatTensor ===== torch.from_numpy(pic.transpose([3,0,1,2]))
            labels: num_classes x num_frames
        """
        vid, label, nf = self.dataset[index]
        start_f = random.randint(0, nf-self.seq_len-1)

        if self.mode == 'rgb':
            input_dir = f"{self.root}/streams/{vid[0]}_{vid[1]}_{vid[6]}_{vid[7]}/rgb"
            inputData = self.load_rgb_frames(input_dir, vid, start_f)
            gt = torch.from_numpy(label[:, start_f:start_f+self.seq_len])
        elif self.mode == 'flow':
            input_dir = f"{self.root}/streams/{vid[0]}_{vid[1]}_{vid[6]}_{vid[7]}/flow"
            inputData = self.load_flow_frames(input_dir, vid, start_f)
            gt = torch.from_numpy(label[:, start_f:start_f+self.seq_len])
        elif self.mode == 'pose':
            print('NOT NOW!')
            exit(0)
            inputData = self.load_pose_frames(vid, start_f)
        elif self.mode == 'two':
            # TODO: Flow images index Need to ALIGN with the other images
            print('NOT NOW!')
            exit(0)
        elif self.mode == 'three':
            # TODO: Flow images index Need to ALIGN with the other images
            print('NOT NOW!')
            exit(0)
        
        ####### Data Augmentation ######
        # # Not sure if should use here
        # inputData = self.transforms(inputData)
        #######
        inputData = torch.from_numpy(inputData.transpose([3,0,1,2]))
        

        return inputData, gt

    def __len__(self):
        """ Return number of samples
        """
        return len(self.dataset)


def main():
    """ Test function for Dataloader
    '''bash    'python -m datasets.AVA' at root path of the folder
    """
    dataset = ava_dataset()
    dataloader = tud.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    for i, (inputs, labels) in enumerate(dataloader):
        print(i)
        print(inputs.shape, labels.shape)

    print("Well Done!")

if __name__ == "__main__":
    main()