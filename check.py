import torch
import torch.nn as nn
from model.unet_model import *
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Hyper Parameters
DEVICE = 'cuda'
IMG_SIZE = 256
OUT_FEATURES = 4

# Set the input image.
raw_img: Image.Image = Image.open('./img/ljcd2.jpg')

# Transform.
transformer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
input_img = transformer(raw_img).to(DEVICE).unsqueeze(0)

# Load the net.
net = UNet(3, OUT_FEATURES).to(DEVICE)
net.load_state_dict(torch.load('./net_param.pkl'))

print('Loaded the image and the net successfully!')

# Start UNet.
net.eval()
with torch.no_grad():
    pred = net(input_img).squeeze()
    pred_img = torch.max(pred, 0)[1].to(
        'cpu').detach().unsqueeze(0).type(torch.uint8)
    print('Finished UNet.')

    # Show the result.
    transform_to_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((raw_img.height, raw_img.width), Image.NEAREST)
    ])
    pred_img = transform_to_img(pred_img)

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(raw_img)
    plt.subplot(1, 2, 2)
    plt.title('pred')
    plt.imshow(pred_img)
    plt.show()
