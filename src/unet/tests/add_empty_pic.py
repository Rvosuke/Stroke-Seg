import os
import shutil


pic_list = ['4', '3', '2', '1', '5', '6']
pic_dir = './dataview/stoke_mask'

for dir in os.listdir(pic_dir):
    # for pic_one in os.listdir(os.path.join(pic_dir, dir)):
    pic_one_dir = os.listdir(os.path.join(pic_dir, dir))
    # print(pic_one_dir)
    for i in range(len(pic_one_dir)):  # change the form of pic.png to pic
        pic_one_dir[i] = pic_one_dir[i].split('.')[0]
    # print(pic_one_dir)
    for pic_num in pic_list:
        if pic_num not in pic_one_dir:
            shutil.copyfile('1.png', os.path.join(pic_dir, dir, pic_num+'.png'))
            print(os.path.join(pic_dir, dir, pic_num+'.png'))