import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image
import os
import cv2


path = './dataset/SegmentationClass'
img_list = os.listdir(path)  # 图片名称列表
# 通道转换
def change_image_channels(image):
    # 3通道转单通道
    if image.mode == 'RGB':
        r, g, b = image.split()
        return r
    else:
        return 0


for imgs in img_list:
    img = Image.open(os.path.join(path, imgs))  # 读取图片
    # img = change_image_channels(img)
    img = img.convert('P')
    img.save(os.path.join(path, imgs))  # 保存（默认路径为原图片位置+原图片名[也就是相当于覆盖]）
