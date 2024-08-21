import os
import cv2

pic_dir = './dataview/stoke-out-pic-dir/out-6-pic-dir'
save_dir = './dataview/stoke-out-pic/out-6-pic'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for dir in os.listdir(pic_dir):
    # print(os.path.join(pic_dir,dir+'.png'))

    pic_ori = cv2.imread(os.path.join(pic_dir, dir, os.listdir(os.path.join(pic_dir, dir))[0]))
    pic_ori = cv2.resize(pic_ori, (400, 400))
    # cv2.imshow("Image",pic_ori)
    # cv2.waitKey()
    for pic in os.listdir(os.path.join(pic_dir, dir))[1:]:
        print(os.path.join(pic_dir, dir))
        pic_cut = cv2.imread(os.path.join(pic_dir, dir, pic))
        pic_cut = cv2.resize(pic_cut, (400, 400))
        pic_ori = pic_cut + pic_ori
        # cv2.imshow("Image", pic_ori)
        # cv2.waitKey()
    cv2.imwrite(os.path.join(save_dir, dir+'.png'), pic_ori)