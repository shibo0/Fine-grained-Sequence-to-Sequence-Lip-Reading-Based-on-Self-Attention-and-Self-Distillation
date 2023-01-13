import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random
import os
import sys
from data_loader.dataset import MyDataset
from Optim import ScheduledOptim
from torch.autograd import Variable
import numpy as np
import time
import torch.optim as optim
import re
import json
import torch.distributed as dist
#torch.autograd.set_detect_anomaly(True)  #检测loss出现nan位置

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if(__name__ == '__main__'):
    opt = __import__('options')  
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu 

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory=True)

def show_lr(optimizer):
    return optimizer.current_lr

def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_,:], start=1) for _ in range(y.shape[0])]
    
    
def test(model, net):

    with torch.no_grad():
        dataset = MyDataset("val")
            
        print('num_test_data:{}'.format(len(dataset))) 
        loader = dataset2dataloader(dataset, shuffle=False)
        ''' 
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt_output').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            y = net(vid, vid_len, txt)
            if i_iter == 30:
                break
        '''
        model.eval()
        loss_list = []
        wer = []
        cer = []
        CTC_criterion = nn.CTCLoss()
        CE_criterion = nn.CrossEntropyLoss(ignore_index=0)
        tic = time.time()
        for (i_iter, input) in enumerate(loader):            
            vid = input.get('vid').cuda(non_blocking=True)
            txt = input.get('txt').cuda(non_blocking=True)
            vid_len = input.get('vid_len').cuda(non_blocking=True)
            txt_len = input.get('txt_len').cuda(non_blocking=True)
            duration = input.get('duration').cuda(non_blocking=True).float()
            
            y, gold, ctc_output = net(vid, vid_len, txt, duration)
            #y = net(vid)
            loss_ctc = CTC_criterion(ctc_output.transpose(0, 1).log_softmax(-1), gold, vid_len.view(-1), txt_len.view(-1))
            loss_ce = CE_criterion(y.view(-1, y.size(2)), gold.view(-1))
            loss = loss_ctc*0.3 + loss_ce*0.7
            
            loss_list.append(loss.detach().cpu().numpy())
            pred_txt = ctc_decode(y.detach().cpu().numpy())
            
            txt = txt.detach().cpu().numpy()
            truth_txt = [MyDataset.arr2txt(txt[_,:], start=1) for _ in range(txt.shape[0])]
            wer.extend(MyDataset.wer(pred_txt, truth_txt)) 
            cer.extend(MyDataset.cer(pred_txt, truth_txt))              
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                
                print(''.join(101*'-'))                
                print('{:<40}|{:>40}'.format('predict', 'truth'))
                print(''.join(101*'-'))                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<40}|{:>40}'.format(predict, truth))                
                print(''.join(101 *'-'))
                print('test_iter={},eta={:.6f},wer={:.6f},cer={:.6f}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))                
                print(''.join(101 *'-'))
                
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())
    
def train(model, net):
    
    dataset = MyDataset("train")
        
    loader = dataset2dataloader(dataset) 
    optimizer = ScheduledOptim(
       optim.Adam( net.parameters(), #filter(lambda p: p.requires_grad, net.parameters()),  #net.parameters(),
           betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-5),
        0.96, 512, 20000)

    #params = [
    #        {"params": model.frontend3D.parameters(), "lr": opt.base_lr},
    #        {"params": model.resnet18.parameters(), "lr": opt.base_lr},
    #        {"params": model.gru1.parameters(), "lr": opt.base_lr*0.1 },
    #        {"params": model.gru2.parameters(), "lr": opt.base_lr*0.1 },
    #        {"params": model.FC.parameters(), "lr": opt.base_lr*0.1 },
    #   ]
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.6, betas=(0.9, 0.98), eps=1e-09)   
    total_video = len(dataset)
    print('num_train_data:{}'.format(total_video))   
    criterion_ctc = nn.CTCLoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    tic = time.time()
    train_wer = []
    train_cer = []
    for epoch in range(opt.max_epoch):
        print("Echo : {0} ...".format(epoch))
        for (i_iter, input) in enumerate(loader):
            print("  Iter {0}".format(i_iter))
            batch_time = time.time()
            model.train()
            #model.apply(fix_bn)
            vid = input.get('vid').cuda(non_blocking=True)
            txt = input.get('txt').cuda(non_blocking=True)
            vid_len = input.get('vid_len').cuda(non_blocking=True)
            txt_len = input.get('txt_len').cuda(non_blocking=True)
            duration = input.get('duration').cuda(non_blocking=True).float()

            #print("video", vid.shape)
            #print("txt", txt.shape,txt)
            #print("vid_len",vid_len)
            #print("txt_len",txt_len)
            #print("duration ",duration.shape,duration)
            
            optimizer.zero_grad()
            y, gold, ctc_output = net(vid, vid_len, txt, duration)

            #print("out",y.shape)
            #print("ctc out",ctc_output.shape)
            #print("gold",gold.shape)
            
            loss_ctc = criterion_ctc(ctc_output.transpose(0, 1).log_softmax(-1), gold, vid_len.view(-1), txt_len.view(-1))
            loss_ce = criterion_ce(y.view(-1, y.size(2)), gold.view(-1))
            loss = loss_ctc*0.3 + loss_ce*0.7
            print("Loss:", loss)
      
            loss.backward()
            if(opt.is_optimize):
                #optimizer.step()
                optimizer.step_and_update_lr()
            tot_iter = i_iter + epoch*len(loader) + 1

            pred_txt = ctc_decode(y.detach().cpu().numpy())
            txt = txt.detach().cpu().numpy()
            truth_txt = [MyDataset.arr2txt(txt[_,:], start=1) for _ in range(txt.shape[0])]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))
            
            print("batch time :", time.time() - batch_time, (time.time()-tic)/tot_iter,"current lr:",show_lr(optimizer))
            if(tot_iter % opt.display == 0): 
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (total_video/opt.batch_size-i_iter)*v/3600.0
       
                print(''.join(101*'-'))                
                print('{:<40}|{:>40}'.format('predict', 'truth'))                
                print(''.join(101*'-'))
                
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<40}|{:>40}'.format(predict, truth))
                print(''.join(101*'-'))                
                print('epoch={},tot_iter={},eta={:.7f},loss={:.6f},train_wer={:.6f},train_cer={:.6f}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean(),np.array(train_cer).mean()))
                print(''.join(101*'-'))
                train_wer = []
                train_cer = []
                
            if(tot_iter % opt.test_step == 0): 
                with torch.no_grad():
                    (loss, wer, cer) = test(model, net)
                    print('i_iter={},loss={:.6f},wer={:.6f},cer={:.6f},lr={}'
                        .format(tot_iter,loss,wer,cer, show_lr(optimizer)))
                   
                savename = 'LipNet_{}_loss_{:.6f}_wer_{:.6f}_cer_{:.6f}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()
            #if tot_iter == 20000:
            #    exit()

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if(__name__ == '__main__'):
    print("Loading options...")
    #from speech_ResNet18.transformer import Transformer
    #model = Transformer()

    #from video_withoutPain.video_cnn import VideoCNN
    #model = VideoCNN()

    from seresformer_ende.transformer import Transformer
    model = Transformer()
    
    #from models import LipNet
    #model = LipNet()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        print("Use weights file name:", opt.weights)
        pretrained_dict = torch.load(opt.weights)
        print(pretrained_dict.keys())
        print("*"*20)
        model_dict = model.state_dict()
        pretrained_dict_update = {}
        for k,v in pretrained_dict.items():
            if k in model_dict.keys() and v.size() == model_dict[k].size():
                pretrained_dict_update[k] = v
            if k[:8]+k[23:] in model_dict.keys() and v.size() == model_dict[k[:8]+k[23:]].size():
                pretrained_dict_update[k[:8]+k[23:]] = v  #backbone.Encoder_W_real.resnet18.la
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        print(pretrained_dict_update.keys())
        print("*"*20)
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict_update.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict_update),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict_update)
        model.load_state_dict(model_dict)
       
    setup_seed(opt.random_seed)
    train(model, net)

        
