#######################
import cv2
import numpy as np
from skimage import morphology
from functools import reduce
from PIL import Image
import math
import xml.etree.ElementTree as ET
import csv
import os
from tqdm import tqdm
import matplotlib.image as mping
import matplotlib.pyplot as plt
####################################################################################
def main(original_img_path, template_img_path):



    img1 = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    ###################################################################################
    # 一、图片预处理部分（去除墨块、主体提取、高精度配准）
    width = 400
    height1 = int(width * img1.shape[0] / img1.shape[1])
    height2 = int(width * img2.shape[0] / img2.shape[1])
    img1 = cv2.resize(img1, (width, height1), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height2), interpolation=cv2.INTER_AREA)


    # 二值化与去除墨块
    def num2true(img):
        img = np.array(img).astype(bool)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    img[i][j] = bool(True)
                else:
                    img[i][j] = bool(False)
        return img


    def true2num(img):
        img.dtype = 'uint8'
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 1:
                    img[i][j] = 0
                else:
                    img[i][j] = 255
        # img = np.array(img).astype(int)
        return img


    _, res1 = cv2.threshold(img1, 90, 255, cv2.THRESH_BINARY)
    _, res2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)
    res1 = num2true(res1)
    res2 = num2true(res2)
    res1 = morphology.remove_small_objects(res1, min_size=400, connectivity=1)
    res2 = morphology.remove_small_objects(res2, min_size=400, connectivity=1)
    res1 = true2num(res1)
    res2 = true2num(res2)
    kernel = np.ones((5, 5), np.uint8)
    res2 = cv2.erode(res2, kernel)
    kernel = np.ones((5, 5), np.uint8)
    res2 = cv2.dilate(res2, kernel)
    image1 = res1.copy()
    image2 = res2.copy()


    # cv2.imshow('PART 1 presentation', image1)
    # cv2.imshow('PART 2 presentation', image2)
    # cv2.waitKey(0)
    # 取中心点函数
    def center(location):
        lenth = location.shape[0]
        sumloc = 0
        for i in range(lenth):
            sumloc = sumloc + location[i, 0, 0]
        avrlenth = sumloc / lenth
        sumlocw = 0
        for i in range(lenth):
            sumlocw = sumlocw + location[i, 0, 1]
        avrwidth = sumlocw / lenth
        return avrlenth, avrwidth


    # 将不连通的图提取外接矩形
    def trans2whole(image):
        # image = 255 - image
        minx, miny, maxx, maxy = width, width, 0, 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 0:
                    if i <= minx:
                        minx = i
                    if i >= maxx:
                        maxx = i
                    if j <= miny:
                        miny = j
                    if j >= maxy:
                        maxy = j
        ptLeftTop = (miny, minx)
        ptRightBottom = (maxy, maxx)
        # point_color = (0, 0, 0)  # BGR
        # thickness = 1
        # lineType = 4
        # cv2.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        return ptLeftTop, ptRightBottom


    # 提取字体最小拟合矩形
    def minrect(image):
        ptLeftTop, ptRightBottom = trans2whole(image)
        roi = image[ptLeftTop[1]:ptRightBottom[1], ptLeftTop[0]:ptRightBottom[0]]
        return roi


    newimage = np.zeros((400, 400), np.uint8)
    newimage.fill(255)


    # 图像平移配准
    def imgmatch(newimage, src, cx, cy):
        cx1 = cy1 = int(newimage.shape[0] / 2)
        deltax = cx1 - cx
        deltay = cy1 - cy
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                if i + deltax < newimage.shape[0] and j + deltay < newimage.shape[0]:
                    newimage[i + deltax][j + deltay] = src[i][j]
        return newimage


    # 新建图像重新匹配
    def newimage(model, img, size):
        newimage1 = np.zeros((size, size), np.uint8)
        newimage2 = np.zeros((size, size), np.uint8)
        newimage1.fill(255)
        newimage2.fill(255)
        roi1 = minrect(model)
        roi2 = minrect(img)
        M1 = cv2.moments(255 - roi1)
        M2 = cv2.moments(255 - roi2)
        num_model = int(M1["m00"]) / 255
        num_roi = int(M2["m00"]) / 255
        scale_percent = math.sqrt(num_model / num_roi)
        roi2 = cv2.resize(roi2, (int(roi2.shape[1] * scale_percent), int(roi2.shape[0] * scale_percent)),
                          interpolation=cv2.INTER_AREA)
        M2 = cv2.moments(255 - roi2)
        # 重心计算
        cx1 = int(M1['m10'] / M1['m00'])
        cy1 = int(M1['m01'] / M1['m00'])
        cx2 = int(M2['m10'] / M1['m00'])
        cy2 = int(M2['m01'] / M1['m00'])
        # cv2.circle(roi1, (cx1,cy1), 1, (0, 0, 255), 4)
        # cv2.circle(roi2, (cx2, cy2), 2, (0, 0, 255), 4)
        newimage1 = imgmatch(newimage1, roi1, cy1, cx1)
        newimage2 = imgmatch(newimage2, roi2, cy2, cx2)
        return newimage1, newimage2, M1, M2

    newimage1, newimage2, comment1_cx, comment2_cx = newimage(image1, image2, 400)

    strumodel1 = 255 - newimage1.copy()
    strumodel2 = 255 - newimage1.copy()
    strumodel3 = 255 - newimage1.copy()
    strumodel4 = 255 - newimage1.copy()
    strustudent1 = 255 - newimage2.copy()
    strustudent2 = 255 - newimage2.copy()
    strustudent3 = 255 - newimage2.copy()
    strustudent4 = 255 - newimage2.copy()


    #########################################################################################
    # 二、字形笔画IOU评价
    # 计算IOU
    def get_iou(target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score


    iou1 = get_iou(255 - newimage1, 255 - newimage2)
    score1 = iou1 * 10 * (2 - iou1)
    imageshow2 = newimage2.copy()
    imageshow2 = Image.fromarray(imageshow2).convert("RGB")
    imageshow2 = np.array(imageshow2)
    contours, hier = cv2.findContours(255 - newimage1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    imageshow2 = 255 - imageshow2
    cv2.drawContours(imageshow2, contours, -1, (0, 0, 255), 3)

    #########################################################################################
    # 三、框架结构IOU评价
    imagestr = imageshow2.copy()
    thresh1 = 255 - newimage1
    thresh2 = 255 - newimage2

    comment = res2.copy()
    comment = 255 - comment

    contours1, hierarchy1 = cv2.findContours(thresh1, 3, 2)  # 模仿
    contours2, hierarchy2 = cv2.findContours(thresh2, 3, 2)  # 原图

    contours_comment, hierarchy_comment = cv2.findContours(comment, 3, 2)

    centerdot1 = np.zeros(shape=(len(contours1), 1, 2), dtype=np.int)
    centerdot2 = np.zeros(shape=(len(contours2), 1, 2), dtype=np.int)
    centerdot_comment = np.zeros(shape=(len(contours_comment), 1, 2), dtype=np.int)

    for i in range(len(contours1)):
        # 1.先找到轮廓
        cnt = contours1[i]
        # 2.寻找凸包，得到凸包的角点
        hull = cv2.convexHull(cnt)
        # 3.绘制凸包
        # print(hull)
        cv2.polylines(res1, [hull], True, (0, 0, 0), 2)
        centerdot1[i, 0, 0] = int(center(hull)[0])
        centerdot1[i, 0, 1] = int(center(hull)[1])
        # print(centerdot1[i, 0, 0])

    for i in range(len(contours2)):
        # 1.先找到轮廓
        cnt = contours2[i]
        # 2.寻找凸包，得到凸包的角点
        hull = cv2.convexHull(cnt)
        # 3.绘制凸包
        # print(hull)
        cv2.polylines(res2, [hull], True, (0, 0, 0), 2)
        centerdot2[i, 0, 0] = int(center(hull)[0])
        centerdot2[i, 0, 1] = int(center(hull)[1])
        # print(centerdot2[i, 0, 0])

    for i in range(len(contours_comment)):
        # 1.先找到轮廓
        cnt = contours_comment[i]
        # 2.寻找凸包，得到凸包的角点
        hull = cv2.convexHull(cnt)
        # 3.绘制凸包
        # print(hull)
        cv2.polylines(res2, [hull], True, (0, 0, 0), 2)
        centerdot_comment[i, 0, 0] = int(center(hull)[0])
        centerdot_comment[i, 0, 1] = int(center(hull)[1])

    cv2.polylines(thresh1, [centerdot1], True, (255, 255, 255), 2)
    cv2.polylines(thresh2, [centerdot2], True, (255, 255, 255), 2)
    cv2.polylines(comment, [centerdot_comment], True, (255, 255, 255), 2)

    # 画外部轮廓_, thresh3 = cv2.threshold(image1, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours3, hierarchy3 = cv2.findContours(thresh1, 3, 2)
    contours4, hierarchy4 = cv2.findContours(thresh2, 3, 2)
    contours_comment, hierarchy_comment = cv2.findContours(comment, 3, 2)

    # 1.先找到轮廓
    cnt3 = contours3[0]
    cnt4 = contours4[0]
    cnt_comment = contours_comment[0]
    # 2.寻找凸包，得到凸包的角点
    hull3 = cv2.convexHull(cnt3)
    hull4 = cv2.convexHull(cnt4)
    hull_comment = cv2.convexHull(cnt_comment)
    # 3.绘制凸包
    # print(hull)
    cv2.polylines(thresh1, [hull3], True, (255, 255, 255), 1)
    cv2.polylines(imagestr, [hull3], True, (0, 0, 255), 6)  # 可视化
    cv2.polylines(imagestr, [hull4], True, (255, 0, 0), 3)  # 可视化
    cv2.polylines(thresh2, [hull4], True, (255, 255, 255), 1)
    cv2.fillPoly(thresh1, [hull3], (255, 255, 255))  # 填充内部
    cv2.fillPoly(thresh2, [hull4], (255, 255, 255))  # 填充内部

    # cv2.polylines(imagestr, [hull_comment], True, (255, 0, 0), 3)  可视化
    cv2.polylines(comment, [hull_comment], True, (255, 255, 255), 1)
    cv2.fillPoly(comment, [hull_comment], (255, 255, 255))  # 填充内部

    #-----------------commernt_m--------------------------
    M1 = cv2.moments(thresh1)
    M2 = cv2.moments(comment)
    # num_model = int(M1["m00"]) / 255
    # num_roi = int(M2["m00"]) / 255
    size_z = math.sqrt(M1["m00"] / M2["m00"])
    # 重心计算
    cx1 = int(M1['m10'] / M1['m00'])
    cy1 = int(M1['m01'] / M1['m00'])
    cx2 = int(M2['m10'] / M1['m00'])
    cy2 = int(M2['m01'] / M1['m00'])
    position = [cx1, cy1, cx2, cy2]

    cx_value = (comment1_cx['m00'] / M1["m00"]) / (comment2_cx['m00'] / M2["m00"])

    if size_z > 1.15:
        size_comment = 0
        # print('字写的小了')
    elif size_z > 1.04:
        size_comment = 1
        # print('字写的有点小了')
    elif size_z < 0.85:
        size_comment = 2
        # print('字写的大了')
    elif size_z < 0.96:
        size_comment = 3
        # print('字写的有点大了')
    else:
        size_comment = 4
        # print('字的大小不错哟')

    judge_point = 35

    if position[0] - position[2] > judge_point and abs(position[1] - position[3]) < judge_point:
        position_comment = 0
        # print('字写的偏左了')
    elif position[0] - position[2] < -judge_point and abs(position[1] - position[3]) < judge_point:
        position_comment = 1
        # print('字写的偏右了')
    elif position[1] - position[3] > judge_point and abs(position[0] - position[2]) < judge_point:
        position_comment = 2
        # print('字写的偏上了')
    elif position[1] - position[3] < -judge_point and abs(position[0] - position[2]) < judge_point:
        position_comment = 3
        # print('字写的偏下了')
    elif position[0] - position[2] > judge_point and position[1] - position[3] > judge_point:
        position_comment = 4
        # print('字写的偏左上方了')
    elif position[0] - position[2] < -judge_point and position[1] - position[3] > judge_point:
        position_comment = 5
        # print('字写的偏右上了')
    elif position[0] - position[2] > judge_point and position[1] - position[3] < -judge_point:
        position_comment = 6
        # print('字写的偏左下方了')
    elif position[0] - position[2] < -judge_point and position[1] - position[3] < -judge_point:
        position_comment = 7
        # print('字写的偏右下了')
    else:
        position_comment = 8
        # print('字的位置写的刚好')

    if cx_value > 1.5:
        cx_comment = 0
        # print('字写的粗了')
    elif cx_value > 1.2:
        cx_comment = 1
        # print('字写的有点粗了')
    elif cx_value < 0.5:
        cx_comment = 2
        # print('字写的细了')
    elif cx_value < 0.8:
        cx_comment = 3
        # print('字写的有点细了')
    else:
        cx_comment = 4
        # print('字的粗细写的不错!')
    #-----------------commernt_m--------------------------

    # 计算IOU
    iou2 = get_iou(thresh1, thresh2)
    ###########################################################################
    def rotate_bound(image, angle):
        # 获取图像的尺寸
        # 旋转中心
        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        # 设置旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像旋转后的新边界
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        return cv2.warpAffine(image, M, (nW, nH))


    # 计算投影直方图
    def statisticalhis(img):
        h = img.shape[0]
        w = img.shape[1]

        # 从列看
        a = np.zeros([w, 1], np.float32)

        for j in range(0, w):  # 遍历每一行
            for i in range(0, h):  # 遍历每一列
                if img[i, j] > 0:  # 判断该点是否为黑点，0代表黑点
                    a[j, 0] += 1
                    img[i, j] = 0  # 将其改为白点，即等于255
        for j in range(0, w):  # 遍历每一行
            for i in range(0, int(a[j, 0])):  # 从该行应该变黑的最左边的点开始向最右边的点设置黑点
                img[h - i - 1, j] = 255  # 设置黑点
        return img, a


    # 0°
    strumodel1, amodle1 = statisticalhis(strumodel1)
    strustudent1, astudent1 = statisticalhis(strustudent1)
    iou41 = get_iou(strumodel1, strustudent1)
    match12 = cv2.compareHist(amodle1, astudent1, cv2.HISTCMP_CORREL)
    # print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match11, match12, match13))
    # cv2.imshow("bone model2", strumodel1)
    # cv2.imshow("bone student2", strustudent1)
    # cv2.waitKey(0)
    # -45°
    strumodel2, amodle2 = statisticalhis(rotate_bound(strumodel2, -45))
    strustudent2, astudent2 = statisticalhis(rotate_bound(strustudent2, -45))
    iou42 = get_iou(strumodel2, strustudent2)
    match22 = cv2.compareHist(amodle2, astudent2, cv2.HISTCMP_CORREL)
    # print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match21, match22, match23))
    # cv2.imshow("bone model1", strumodel2)
    # cv2.imshow("bone student1", strustudent2)
    # cv2.waitKey(0)
    # 45°
    strumodel3, amodle3 = statisticalhis(rotate_bound(strumodel3, 45))
    strustudent3, astudent3 = statisticalhis(rotate_bound(strustudent3, 45))
    iou43 = get_iou(strumodel3, strustudent3)
    match32 = cv2.compareHist(amodle3, astudent3, cv2.HISTCMP_CORREL)
    # print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match31, match32, match33))
    # cv2.imshow("bone model2", strumodel3)
    # cv2.imshow("bone student2", strustudent3)
    # cv2.waitKey(0)
    # 90°
    strumodel4, amodle4 = statisticalhis(rotate_bound(strumodel4, 90))
    strustudent4, astudent4 = statisticalhis(rotate_bound(strustudent4, 90))
    iou44 = get_iou(strumodel4, strustudent4)
    match42 = cv2.compareHist(amodle4, astudent4, cv2.HISTCMP_CORREL)
    ############################################################################

    tradition = [iou1, iou2, score1, iou41, match12, iou42, match22, iou43,
                                 match32, iou44, match42]

    # print(tradition)
    return tradition, res1, res2, newimage1, newimage2, imageshow2, imagestr, size_comment, position_comment, cx_comment

if __name__ == '__main__':
    original_img_path = './input/ori.png'
    template_img_path = './input/temp.png'
    x, _, _, _, _ = main(original_img_path, template_img_path)
    print(x)
