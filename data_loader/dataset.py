# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
import random
import editdistance
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
random.seed(2021)

jpeg = TurboJPEG()
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','^','$']
    
    def __init__(self, phase):
        
        self.data = []
        self.phase = phase
        if(self.phase == 'train'):
            self.index_root = 'LRW1000_Public_pkl_jpeg/trn'
        if(self.phase == 'test'):
            self.index_root = 'LRW1000_Public_pkl_jpeg/tst'   
        if(self.phase == 'val'):
            self.index_root = 'LRW1000_Public_pkl_jpeg/val'                     
        
        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))
        if phase == "train":random.shuffle(self.data)
        #if phase == 'test':
        #    random.shuffle(self.data)
        #    self.data = self.data[:5000]
            

                                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])
        video = pkl.get('video')
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]        
        video = np.stack(video, 0)
        video = video[:,:,:,0]
       
        
        if(self.phase == 'train'):
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == 'val' or self.phase == 'test':
            video = CenterCrop(video, (88, 88))      
        
        #video = CenterCrop(video, (88, 88))  # without any augment when start
        
        pkl['vid'] = torch.FloatTensor(video)[:,None,...] / 255.0 
        
        anno = pkl.get('txt')  
        txt_len = anno.shape[0]
        pkl["txt"] = torch.LongTensor(self._padding(anno, 50))  
        pkl["txt_len"] = txt_len + 1
        pkl["vid_len"] = 40
                
        return pkl
        
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0) 
        
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr[:]:
            if n == 29: break
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip() 
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(n >= start):
                if n == 29:
                    break
                txt.append(MyDataset.letters[n - start])               

        return ''.join(txt).strip()

    '''
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                if MyDataset.letters[n - start] == '$':
                    break
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])               
            pre = n
        return ''.join(txt).strip()
    '''
    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
