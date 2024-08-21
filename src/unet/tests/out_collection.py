import os
import shutil
import re

inputdir = './dataview/stoke-out-pic'
outputdir = './dataview/stoke_mask'

for pic_dir in os.listdir(inputdir):
    for pic in os.listdir(os.path.join(inputdir, pic_dir)):

        newdir = os.path.join(outputdir, pic.split('.png')[0])
        if not os.path.exists(newdir):
            os.makedirs(newdir)

        oldname = os.path.join(inputdir, pic_dir, pic)

        # 设置新文件名
        newname = newdir + os.sep + pic_dir.split('-')[1] + '.png'
        # newname = outputdir + os.sep + dir.split('_')[0] + os.sep + dir.split('_')[0] + '.png'

        shutil.copyfile(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)
