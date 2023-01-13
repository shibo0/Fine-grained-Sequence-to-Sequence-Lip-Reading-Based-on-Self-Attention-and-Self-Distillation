dataset = "LRW-1000"
gpu = '0,1'
random_seed = 0
images_path = "/media/data3/lip/LRW_Data/lipread_mp4_crop/"
vid_padding = 29
txt_padding = 50
batch_size = 64
base_lr = 2e-3
num_workers = 16
max_epoch = 10000
display = 100
test_step = 4000
save_prefix = 'weights_LRW/'
is_optimize = True
weights = "LipNet_weights_LRW/_loss_0.602165_wer_0.464181_cer_0.219984.pt"  

