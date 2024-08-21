import os
import cv2

pic_dir = './dataset/JPEGImages'
cut_dir = './dataview/v2_mask'

for dir in os.listdir(cut_dir):
    # print(os.path.join(pic_dir,dir+'.png'))
    pic_ori = cv2.imread(os.path.join(pic_dir,dir+'.png'))
    # cv2.imshow("Image",pic_ori)
    # cv2.waitKey()
    for pic_cut_dir in os.listdir(os.path.join(cut_dir, dir)):
        print(os.path.join(cut_dir, dir,pic_cut_dir))
        pic_cut = cv2.imread(os.path.join(cut_dir, dir,pic_cut_dir))
        pic_cut = cv2.resize(pic_cut, (400, 400))
        pic_ori = cv2.resize(pic_ori, (400, 400))
        pic_new = pic_cut - pic_ori
        # cv2.imshow("Image",pic_new)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(cut_dir, dir,pic_cut_dir),pic_new)