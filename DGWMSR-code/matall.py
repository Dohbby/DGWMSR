###################---------Import---------###################
import os
import argparse
import torch
import time
import datetime
import random
import cv2
import glob
import requests
import utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
import skimage.color as sc
from dataset import H5Dataset


from collections import OrderedDict
from importlib import import_module
from tqdm import tqdm
# from torchsummary import summary
# from ptflops import get_model_complexity_info
from data import DIV2K_train, DIV2K_valid, Set5_val
from torch.utils.data import DataLoader
import torch.optim as optim
from util_calculate_psnr_ssim import bgr2ycbcr, calculate_psnr, calculate_ssim, calculate_psnrb

# from model.swin2sr import Swin2SR as net   #patch_size=192
# from model import hat
from model import cfat
from skimage import metrics
# from pytorch_msssim import ssim

# from Model import MFSR

torch.backends.cudnn.benchmark = True

###################---------Arguments---------###################
# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=50, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=[225, 350, 400, 450], help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5, help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loading")
parser.add_argument("--root", type=str, default="./Datasets/DIV2K/", help='dataset directory')
parser.add_argument("--n_train", type=int, default=800, help="number of training set")
parser.add_argument("--n_val", type=int, default=5, help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4, help="super-resolution scale")
# parser.add_argument("--patch_size", type=int, default=256, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=1, help="number of color channels to use")
parser.add_argument("--in_channels", type=int, default=72, help="number of channels for transformer")
parser.add_argument("--n_layers", type=int, default=3, help="number of FETB uits to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
# parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')
parser.add_argument("--channel_in_DE", type=int, default=3)
parser.add_argument("--channel_base_DE", type=int, default=8)
parser.add_argument("--channel_int", type=int, default=16)
# parser.add_argument("--output_dir", type=str, default="Put the adress of checkpoint directory here")
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--log_period", type=int, default=10)
parser.add_argument("--checkpoint_period", type=int, default=1)
# parser.add_argument("--eval_period", type=int, default=2)
# parser.add_argument('--folder_lq', type=str, default="./TestData_LR/", help='input low-quality test image folder')
# parser.add_argument('--folder_gt', type=str, default="./TestData_GT/", help='input ground-truth test image folder')
# parser.add_argument("--output_folder", type=str, default="./TestData_OUT/")
parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr')
# parser.add_argument('--save_img_only', default=False, action='store_true', help='save image and do not evaluate')
parser.add_argument('--tile', type=int, default=64,
                    help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=16, help='Overlapping of different tiles')
parser.add_argument("--psnr1text", type=str, default="/home/CFAT-yuanma/SRFormer/mat/x4-T1/PSNR-each.txt", help='text file to store validation results')
parser.add_argument("--psnr2text", type=str, default="/home/CFAT-yuanma/SRFormer/mat/x4-T1/PSNR.txt", help='text file to store validation results')
args = parser.parse_args()
print(args)

###################---------Random_Seed---------###################
if args.seed:
    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
else:
    seed_val = random.randint(1, 10000)
    print("Ramdom Seed: ", seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False

###################---------Environment---------###################
cuda = args.cuda
device = torch.device('cuda:0' if cuda else 'cpu')
gpus=[0,1]    #[0, 1, 2, 4] for batch size of 12
def ngpu(gpus):
    """count how many gpus used"""
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


###################---------Model---------###################
print("===> Building models")
torch.cuda.empty_cache()
args.is_train = True

##Model::ESRT
# model = net(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=8,
#                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
#                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
# model = hat.HAT()
from model import MBNSR
model=MBNSR.SRGAN()
# model = cfat.CFAT()

# from model import srfomer
# # model = cfat.CFAT()
# model = srfomer.SRFormer()

# from model import  fscwn
# model=fscwn.FSCWN()
###################---------Loss_Function---------###################
l1_criterion = nn.L1Loss()

###################---------Optimizer---------###################
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.step_size, gamma=args.gamma)

###################---------.to(Device)---------###################
print("===> Setting GPU")
if cuda:
    #print(device, gpus)
    #input()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    l1_criterion = l1_criterion.to(device)




###################---------Inference Time---------###################
# 测试推理时间
model.eval()
with torch.no_grad():
    # 创建两个随机输入张量，形状为 (batch_size=1, channels=72, height=192, width=320)
    test_input_left = torch.randn(1, 1, 192, 320).to(device)
    test_input_right = torch.randn(1, 1, 192, 320).to(device)
    # 记录推理开始时间
    start_time = time.time()

    # 模型推理
    try:
        output = model( test_input_right,test_input_right,test_input_right)  # 假设模型需要两个输入
    except Exception as e:
        print(f"Error during inference: {e}")
        output = None

    # 记录推理结束时间
    end_time = time.time()

    # 计算推理时间
    if output is not None:
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time * 1000:.4f} ms")
    else:
        print("Inference failed, cannot calculate inference time")

###################---------Load_model_to_resume_training---------###################
begin_epoch = 1
train_loss = {}

# if not os.path.exists(save_dir):
#    os.makedirs(save_dir)

border = args.scale
window_size = 16
print("param1:", (sum(param.numel() for param in model.parameters()) / (10 ** 6)))

def measure_inference_time(model, input_tensor, device='cuda', repetitions=100):
    # Warm-up (避免冷启动影响)
    for _ in range(10):
        _ = model(input_tensor)

    # 正式测量
    start_time = time.time()
    for _ in range(repetitions):
        _ = model(input_tensor)
    torch.cuda.synchronize()  # 确保 CUDA 操作完成（如果使用 GPU）
    total_time = time.time() - start_time

    avg_time = total_time / repetitions * 1000  # 转换为毫秒
    print(f"Average inference time: {avg_time:.2f} ms (over {repetitions} runs)")
    return avg_time

for epoch in range(31,32):
    checkpoint_file =  r'/home/CFAT-yuanma/SRFormer/model/x4/checkpoint_' + str(epoch) + '.pth'
    # checkpoint_file =  r'/home/code/CFAT/home/CFAT/pred-model-3x/checkpoint_' + str(epoch) + '.pth'
    # checkpoint_file =  r'/home/code/CFAT/home/CFAT/pred-model-4-2-2/checkpoint_' + str(epoch) + '.pth'
    print(checkpoint_file)

    # checkpoint=torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint['state_dict'])
    if os.path.exists(checkpoint_file):
        #   print('')
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        print('loading state dict')
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded state dict')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    model.eval()
    with torch.no_grad():
        PSNR = []
        for mat_num in range(1,2):
            preds = []
            mse_list = []
            path=r"/home/data/dataset3090/test-k21-4x-h5/"+ 'k21-4x-' + str(mat_num) + '.h5'

            # path = r"/home/data/dataset3090/test-3x-h5/" + 'k21-3x-' + str(mat_num) + '.h5'
            # path = r"/home/data/dataset3090/test-T1-4x-h5/" + 'T1-4x-' + str(mat_num) + '.h5'
            # path = r"/home/data/dataset3090/test-adult-4x-h5/" + 'adult-4x-' + str(mat_num) + '.h5'

            print(path)
            test_dataset = H5Dataset(path)  # 棰勫鐞嗛獙璇侀泦
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1) # 256
            for data in test_dataloader:
                lr, left, right, gt= data
                # left = left.to(device)
                # right = right.to(device)
                # gt = gt.to(device)
                # lr = lr.to(device)
                # #

                left = left.float().to(device)
                right = right.float().to(device)
                gt = gt.float().to(device)
                lr = lr.float().to(device)

                _, _, h_old, w_old = lr.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad] # [1, 1, 192, 320]

                left = torch.cat([left, torch.flip(left, [2])], 2)[:, :, :h_old + h_pad, :]
                left = torch.cat([left, torch.flip(left, [3])], 3)[:, :, :, :w_old + w_pad]  # [1, 1, 192, 320]

                right = torch.cat([right, torch.flip(right, [2])], 2)[:, :, :h_old + h_pad, :]
                right = torch.cat([right, torch.flip(right, [3])], 3)[:, :, :, :w_old + w_pad]  # [1, 1, 192, 320]

                # imgs=torch.cat((lr,left,right),dim=1)
                # print('111111',imgs.shape)
                # measure_inference_time(model, lr)

                output = model(lr) # [1, 1, 192, 320]
                output= output[:, :, :h_old, :w_old] # [1, 1, 170, 256]
                # print(output.shape)

                pred = np.squeeze(output)
                gt = np.squeeze(gt)
                a = pred.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                psnr = metrics.peak_signal_noise_ratio(gt, a, data_range=1)
                mse = metrics.mean_squared_error(gt, a)
                mse_list.append(mse)


                # print('PSNR:', psnr)
                preds.append(a)



            global_mse = np.mean(mse_list)  # 璁＄畻鍏ㄥ眬 MSE
            global_psnr = 10 * np.log10(1.0 / global_mse)  # 鍋囪 data_range=1
            print("Epoch:{}  第{}个头   PSNR:{}".format(epoch, mat_num, global_psnr))
            PSNR.append(global_psnr)
            txt_write = open(args.psnr1text, 'a')
            print("Epoch:{}  第{}个头    PSNR:{}".format(epoch,mat_num,global_psnr), file=txt_write)
            preds = np.array(preds)
            print(preds.shape)
            preds = np.array(preds).transpose(1, 2, 0)

            from scipy.io import savemat

            new = {'data': preds}

            savemat('/home/CFAT-yuanma/SRFormer/mat/x4-T1/pred{}-{}.mat'.format(epoch, mat_num), new)
            # savemat('/home/code/CFAT/pred-adult/pred{}-{}.mat'.format(epoch, mat_num), new)
            print('pred{}-{}.mat'.format(epoch, mat_num))
        print("epoch:{} mean----PSNR:{}".format(epoch, sum(PSNR) / len(PSNR)))
        txt_write = open(args.psnr2text, 'a')
        print("Epoch:{}  PSNR:{}".format(epoch, sum(PSNR) / len(PSNR)), file=txt_write)




print(f'=========================Done=========================')











