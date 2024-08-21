import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.utils.data as Data
import PIL.ImageOps
import torch.nn as nn
# from torch import optim
import torch.nn.functional as F
# plt.switch_backend('agg')
import os
import logging
import xlwt
import csv


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 数据集的设置*****************************************************************************************************************


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')

def getcsvvalue(file_path):
    dictlist = {}
    with open(file_path) as file:
        datareader = csv.reader(file)
        csvlist = list(datareader)
        #keylist = csvlist[0]
        for value in range(0,len(csvlist)):
            csvdict = []
            for item in range(1, 13):
                csvdict.append(float(csvlist[value][item]))
                dictlist[csvlist[value][0]] = csvdict
    return dictlist

# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, original_img, template_img, transform=None, target_transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.original_img = Image.fromarray(original_img)
        self.template_img = Image.fromarray(template_img)

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        # img1 = self.loader(self.original_img_path)  # 按照路径读取图片
        if self.transform is not None:
            img1 = self.transform(self.original_img.convert("RGB"))  
            img2 = self.transform(self.template_img.convert("RGB"))  # 数据标签转换为Tensor
        return img1, img2  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return 1


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()



class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = models.resnet101(progress=False)

        self.fc1 = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(1000, 10))

    def forward_once(self, x):
        output = self.cnn1(x)
        # output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def main(original_img, template_img):
    net = SiameseNetwork()
    net.load_state_dict(torch.load('./resnet101_params.pkl', map_location='cuda:0'))
    net.eval()
    test_data = MyDataset(original_img, template_img, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            img0, img1 = data
            output1, output2 = net(Variable(img0), Variable(img1))
            euclidean_distance = F.pairwise_distance(output1, output2)
            output3 = torch.abs(10 - euclidean_distance).reshape(-1, 1)
            output = torch.cat((output1 - output2, output3), -1)
            # print(output)
    return output.numpy()[0].tolist()

if __name__ == '__main__':
    original_img_path = './input/08.png'
    template_img_path = './input/000768.png'
    x = main(original_img_path, template_img_path)
    print(x)
