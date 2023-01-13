#!/usr/bin/env python
import argparse
import json
import os
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from seresformer_ende.transformer import Transformer
from data_loader.dataset import MyDataset
import time
import warnings
warnings.filterwarnings('ignore')
opt = __import__('options') 
parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
parser.add_argument('--beam-size', default=2, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode-max-len', default=50, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_,:], start=1) for _ in range(1,y.shape[0])]

def dataset2dataloader(dataset, num_workers=0, shuffle=True):
    return DataLoader(dataset,
        batch_size = 1, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False,
        pin_memory=True)

letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','^','$']

def arr2txt_(arr, start):
    txt = []
    for n in arr[:]:
        if n == 29: break
        if n == 28: continue
        if(n >= start):
            txt.append(letters[n - start])     
    return ''.join(txt).strip()  


def recognize(args):
    char_list = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','^','$']
    model = Transformer().cuda()
    if True:
        weights = "LipNet_weights_LRW/_loss_0.602636_wer_0.463714_cer_0.219896.pt"
        pretrained_dict = torch.load(weights)
        model_dict = model.state_dict()
        pretrained_dict_update = {}
        for k,v in pretrained_dict.items():
            if k in model_dict.keys() and v.size() == model_dict[k].size():
                pretrained_dict_update[k] = v
            
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict_update.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict_update),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict_update)
        model.load_state_dict(model_dict)
        
    model.eval()
    dataset = MyDataset('test')
    num = len(dataset)
    print('共有{}个测试样本。'.format(len(dataset)))  
    loader = dataset2dataloader(dataset, shuffle=False)

    # decode each utterance
    wer = []
    cer = []
    i=0
    with torch.no_grad():
        start_time = time.time()
        for (i_iter, input) in enumerate(loader):  
            vid = input.get('vid').cuda(non_blocking=True)
            txt = input.get('txt')
            vid_len = input.get('vid_len').cuda(non_blocking=True)
            txt_len = input.get('txt_len')
            duration = input.get('duration').cuda(non_blocking=True).float()
            
            print("i_iter{}, video shape:".format(i_iter+1),vid.shape)
            nbest_hyps = model.recognize(vid, vid_len, char_list, duration, args)
            
            t = arr2txt_(nbest_hyps[0]['yseq'], 1)
            anno = arr2txt_(txt[0], 1)
            print("Truth :",[anno])
            print("Predit:", [t])
            w, c = dataset.wer([t], [anno]), dataset.cer([t], [anno])
            print(w,c)
            wer.append(w)
            cer.append(c)
            #end_time = time.time()
            
            if i_iter % 20 == 0: 
                end_time = time.time()
                print("Iter", i_iter, "wer cer",np.array(wer).mean(), np.array(cer).mean(),
                        "eta:",(end_time-start_time)/(i_iter+1)*(num-i_iter)/3600.)
        print("Total wer cer:",np.array(wer).mean(), np.array(cer).mean())
        print("Total time:",time.time()-start_time)



if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)
