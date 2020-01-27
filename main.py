import os
import torch
import torchvision
import torch.nn as nn
from model.unet_model import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model.data_transforms as data_transforms
import torch.utils.data as Data
import math
import paramiko


# Hyper Parameters
IMG_SIZE = 256
BATCH_SIZE = 2
DEVICE = 'cuda'
LR = 1e-3
MOMENTUM = 0.9
EPOCH = 100
OUT_FEATURES = 17
L2_REGULATION = 1e-5

# Connect to SSH Server
print('Connecting to ssh server...')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='120.27.241.203', port=22,
            username='prp', password='00000000')
print('Connected to server.')

# Tensor to storage the images and masks.
image_list = torch.empty([0, 3, IMG_SIZE, IMG_SIZE])
mask_list = torch.empty([0, IMG_SIZE, IMG_SIZE])

# Load the data.
print('Loading Images...')
for root, dirs, files in os.walk('./img_voc/JPEGImages'):
    for f in files:
        # Open the files.
        file_name_no_extension = f.split('.')[0]
        img = Image.open('./img_voc/JPEGImages/' +
                         file_name_no_extension + '.jpg').convert('RGB')
        label = Image.open('./img_voc/SegmentationClassPNG/' +
                           file_name_no_extension + '.png').convert('P')

        img_ls, label_ls = data_transforms.transform_to_list(img, label, 10)
        image_list = torch.cat((image_list, img_ls), 0)
        mask_list = torch.cat((mask_list, label_ls), 0)

image_list = image_list.float()
mask_list = mask_list.long()

# Set the dataloader.
train_loader = Data.DataLoader(Data.TensorDataset(
    image_list, mask_list), batch_size=BATCH_SIZE, shuffle=True)

print('Data successfully loaded.')

# Build the net
net = UNet(3, OUT_FEATURES).cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=L2_REGULATION)
loss_func = nn.CrossEntropyLoss().to(DEVICE)

try:
    net.load_state_dict(torch.load('./net_param.pkl'))
    print('Loaded previous model.')
except Exception as err:
    pass

print('Net build finished.')


def SSH_Write(s, mode=0):
    global ssh
    try:
        if mode == 0:
            ssh.exec_command(f'echo \'{s}\' >> log.txt')
        else:
            ssh.exec_command(f'echo \'{s}\' > log.txt')
    except Exception:
        print('Connection failed.')
        try:
            ssh.close()
            ssh.connect(hostname='120.27.241.203', port=22,
                        username='prp', password='00000000')
        except Exception:
            pass


# Start training.
# SSH_Write('Connected to server.', 1)
# SSH_Write('Start Execution...')
TOTAL_STEP = math.ceil(image_list.shape[0] / BATCH_SIZE)
print(
    f'Start training the net. {EPOCH} epoches in total. {TOTAL_STEP} steps each epoch.')
net.train()
for epoch in range(EPOCH):
    total_loss = 0.0
    for step, (batchx, batchy) in enumerate(train_loader):
        batchx, batchy = batchx.to(DEVICE), batchy.to(DEVICE)
        pred: torch.Tensor = net(batchx)
        loss = loss_func(pred, batchy)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.subplot(1, 3, 1)
        plt.title('origin')
        plt.imshow(data_transforms.transform_to_img(batchx[0].to('cpu')))
        plt.subplot(1, 3, 2)
        plt.title('label')
        plt.imshow(batchy[0].to('cpu').numpy())
        plt.subplot(1, 3, 3)
        plt.title('pred')
        plt.imshow(torch.max(pred[0], 0)[1].to('cpu').numpy())
        plt.show()

        print(
            f'EPOCH:{epoch},STEP:{step},LOSS:{loss.item()},EPOCH PROGRESS:{int(step/TOTAL_STEP*100)}%')
        # SSH_Write(
        #     f'EPOCH:{epoch},STEP:{step},LOSS:{loss.item()},EPOCH PROGRESS:{int(step/TOTAL_STEP*100)}%')

    torch.save(net.state_dict(), 'net_param.pkl')
    print(
        f'Model saved! The total loss of epoch{epoch} is {total_loss}. TOTAL PROGRESS:{int(epoch/EPOCH*100)}%')
    # SSH_Write(
    #     f'EPOCH:{epoch},STEP:{step},LOSS:{loss.item()},EPOCH PROGRESS:{int(step/TOTAL_STEP*100)}%')

ssh.close()
