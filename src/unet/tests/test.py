import argparse
import os
import numpy as np
import time
import cv2
import csv
import torch
import torch.nn as nn

from modeling import UNet
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default='./dataset/test',
                        help='image to test')
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    # parser.add_argument('--ckpt', type=str, default='./run/shufa/mobilenet_n2_lr0.002/model_best.pth.tar',
    parser.add_argument('--ckpt', type=str, default='./run/shufa/v1_lr0.002/experiment_5/checkpoint.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='shufa',
                        choices=['pascal', 'coco', 'cityscapes', 'invoice', 'shufa'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=400,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    save_dir = './dataset/test_result_lr0.002'

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()

    model = UNet(n_channels=3, n_classes=6, bilinear=False)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))

    info_csv = {}
    with open("./dataset/test_info.csv", "r", errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            info_csv[row[0]] = row[1:]
        f.close()

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    sigmoid = nn.Sigmoid()
    for name in os.listdir(args.in_path):
        s_time = time.time()
        # image = Image.open(args.in_path + "/" + name).convert('RGB')
        # image = np.array(Image.open(args.in_path + "/" + name).convert('RGB'))
        image = cv2.imread(args.in_path + "/" + name)
        image = cv2.resize(image, (400, 400))
        xxx = info_csv[name.split('.png')[0]]
        # image[389:image.shape[0], 0:int(image.shape[1] / 2), 2] = int(info_csv[name.split('.png')[0]]) / 26 * 255
        image[389:image.shape[0], 0:int(image.shape[1] / 2), :] = int(info_csv[name.split('.png')[0]][0]) / 26 * 255
        image[389:image.shape[0], 201:230, :] = int(info_csv[name.split('.png')[0]][1]) / 10 * 255
        image[389:image.shape[0], 231:260, :] = int(info_csv[name.split('.png')[0]][2]) / 10 * 255
        image[389:image.shape[0], 261:290, :] = int(info_csv[name.split('.png')[0]][3]) / 10 * 255
        image[389:image.shape[0], 291:320, :] = int(info_csv[name.split('.png')[0]][4]) / 10 * 255
        image[389:image.shape[0], 321:350, :] = int(info_csv[name.split('.png')[0]][5]) / 10 * 255
        image[389:image.shape[0], 351:380, :] = int(info_csv[name.split('.png')[0]][6]) / 10 * 255
        # print(image)
        # for x in range(image.shape[0]):
        #     for y in range(image.shape[1]):
        #         if (image[x, y, 0] <= 128 or image[x, y,1] <= 128 or image[x, y, 2] <= 128):  # 判断该点的值
        #             image[x, y, :] = 0
        #         else:
        #             image[x, y, :] = 255
        # image = Image.fromarray(image.astype('uint8')).convert('RGB')

        # image = Image.open(args.in_path).convert('RGB')
        target = Image.open(args.in_path + "/" + name).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
            output = sigmoid(output)
        # print(output.shape)

        if not os.path.exists(save_dir + "/" + name):
            os.makedirs(save_dir + "/" + name)
        for output_n2 in range(output.shape[1]):
            temp_pic = output[0, output_n2, :, :]
            temp_pic = temp_pic.data.cpu().numpy()
            # temp_pic = 1 * (temp_pic > 0.5) + \
            #            0 * (temp_pic < 0.5)
            # grid_image = make_grid(decode_seg_map_sequence(torch.max(temp_pic[:3], 1)[1].detach().cpu().numpy()),
            #                        3, normalize=False, range=(0, 255))
            grid_image = temp_pic * 255
            cv2.imwrite(os.path.join(save_dir, name, "{}_".format(name[0:-4])) + str(output_n2) + ".png", grid_image)

        u_time = time.time()
        img_time = u_time - s_time
        print("image:{} time: {} ".format(name, img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")


if __name__ == "__main__":
    main()

# python demo.py --in-path your_file --out-path your_dst_file
