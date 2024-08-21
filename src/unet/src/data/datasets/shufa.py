from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import cv2
import csv

class shufaSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 # base_dir=Path.db_root_dir('shufa'),
                 base_dir='./dataset',
                 # base_dir='/mnt/D4FA828F299D817A/gxy/deeplabv3_001/dataset',
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        info_csv = {}
        with open("./dataset/info.csv", "r", errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                info_csv[row[0]] = row[1:]
            f.close()

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        # _splits_dir = os.path.join('/mnt/D4FA828F299D817A/gxy/deeplabv3_001/dataset', 'ImageSets', 'Segmentation')
        _splits_dir = os.path.join('./dataset', 'ImageSets', 'Segmentation')
        # print(_splits_dir)

        self.im_ids = []
        self.images = []
        self.categories = []
        self.info = []

        for splt in self.split:
            # with open(os.path.join(os.path.join('/mnt/D4FA828F299D817A/gxy/deeplabv3_001/dataset/ImageSets/Segmentation/', splt + '.txt')), "r") as f:
            with open(os.path.join(os.path.join('./dataset/ImageSets/Segmentation/', splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # print(self._image_dir, self._cat_dir)
                _image = os.path.join(self._image_dir, line + ".png")
                _cat = os.path.join(self._cat_dir, line)
                assert os.path.isfile(_image)
                self.im_ids.append(line)
                self.info.append(info_csv[line])
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        # _img = Image.open(self.images[index]).convert('RGB')
        # _img = _img.resize((400, 400))
        # _img.size
        _img = cv2.imread(os.path.join(self.images[index]))
        _img = cv2.resize(_img, (400, 400))
        # xxx = int(self.info[index])
        _img[389:_img.shape[0], 0:int(_img.shape[1] / 2), :] = int(self.info[index][0]) / 26 * 255
        _img[389:_img.shape[0], 201:230, :] = int(self.info[index][1]) / 10 * 255
        _img[389:_img.shape[0], 231:260, :] = int(self.info[index][2]) / 10 * 255
        _img[389:_img.shape[0], 261:290, :] = int(self.info[index][3]) / 10 * 255
        _img[389:_img.shape[0], 291:320, :] = int(self.info[index][4]) / 10 * 255
        _img[389:_img.shape[0], 321:350, :] = int(self.info[index][5]) / 10 * 255
        _img[389:_img.shape[0], 351:380, :] = int(self.info[index][6]) / 10 * 255

        # for pic_dir in os.listdir(self.categories[index]):
        x = []
        for pic_dir in os.listdir(self.categories[index]):
            pic = cv2.imread(os.path.join(self.categories[index], pic_dir))
            # print(pic_dir, pic)
            pic = cv2.resize(pic, (400, 400))
            img = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('imshow',img)
            # cv2.waitKey(0)
            _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
            binary = binary / 255
            binary = binary[np.newaxis, :]
            # print(img.shape)
            x.append(binary)
        # print((len(x)))
        _target = np.r_[x[0], x[1], x[2], x[3], x[4], x[5]]
        # print(_target, _target.shape)    # --------(16, 400, 400)
        # = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = shufaSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

