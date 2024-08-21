import os


# pic_dir = './dataset/SegmentationClass'
pic_dir = './dataview/stroke_mask'

for dir in os.listdir(pic_dir):
    for img_ori in os.listdir(os.path.join(pic_dir, dir)):
        img = int(img_ori.split('.png')[0])
        if int(img) < 10:
            os.rename(os.path.join(pic_dir, dir, img_ori), os.path.join(pic_dir, dir, '0' + str(img) + '.png'))
            print(os.path.join(pic_dir, dir, '0' + str(img) + '.png'))

