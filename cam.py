import torch.nn as nn
import torch 
from torch.autograd import Variable
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os

class VGG16(nn.Module):
    def __init__(self):  ##  输入224x224
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

##  读入待预测的图片
def img_processing(img_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ##  Normalize参数是因为ImageNet数据集，我们使用的权重是这个数据集训练得到的
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    return img

##  comp_class_vec作用：
##  因为最后vgg16输出的是各个类别的概率，我们只取其中最大的那个结果，然后其他的赋0
def comp_class_vec(output_vec, index=None):
    ##  计算类向量
    if not index:
        index = np.argmax(output_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis] ## np.newaxis作用是在这个位置增加一个一维
    index = torch.from_numpy(index)
    ##  下面这段代码意思是产生[1,1000]数值为0的tensor，并将[1][index]处数值改为1
    one_hot = torch.zeros(1, 1000).scatter_(1, index, 1) ## 加_和不加的区别在于加_是在原tensor上直接修改
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * (output.cpu())) 
    return class_vec

def gen_cam(feature_map, grads):
    ##  根据梯度和特征图，生成cam
    ##  feature_map: [C,H,W] , grads: [C,H,W] , return: [H,W]
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32) ## [H,W]
    weights = np.mean(grads, axis=(1, 2))
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))  
    cam -= np.min(cam)
    cam /= np.max(cam)
    
    return cam


def backwawrd_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

##  VGG16模型载入
vgg16 = VGG16().cuda()
vgg16.eval()
pretrain_path = 'vgg16-397923af.pth'  # 权重文件
checkpoint = torch.load(pretrain_path)
vgg16.load_state_dict(checkpoint)

##  class
fmap_block = list()
grad_block = list()
txt_path = 'label.txt'
labels = []
with open(txt_path, 'r', encoding='utf-8') as f: ## 因为要读入中文，所以要加上encoding='utf-8'
    for lines in f:
        labels.append(lines[:-1])       

##  图片读取
img_path = '5.jpg'
ori_img = Image.open(img_path)
img = img_processing(img_path)

##  注册hook, 来获得该层的梯度
##  因为pytorch计算梯度只保留最后的梯度，不保存中间层梯度和中间结果，所以要用hook(钩子)
vgg16.features[-3].register_forward_hook(farward_hook) ##  forward获得该层的输出结果
vgg16.features[-3].register_backward_hook(backwawrd_hook)  ##  backward获得梯度

##  forward（得到conv5层的输出结果）
output = vgg16(img)
idx = np.argmax(output.cpu().data.numpy())
print("class:{}".format(idx))
print("name:{}".format(labels[idx]))

##  backward，获得梯度(目的是得到conv5后每个特征图的权重)
vgg16.zero_grad()
class_loss = comp_class_vec(output)
class_loss.backward()

##  生成CAM
grads_val = grad_block[0].cpu().data.numpy().squeeze()
fmap = fmap_block[0].cpu().data.numpy().squeeze() 
cam = gen_cam(fmap, grads_val)

##  生成热力图
heatmap = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
heatmap = heatmap / np.max(heatmap)
heatmap = heatmap[:,:,::-1] ## cv2生成的热力图是GBR形式，转换成Image的RGB形式

##  输出结果
img_show = ori_img.resize((224,224))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_show)
plt.subplot(1,2,2)
plt.imshow(heatmap)
plt.show()


