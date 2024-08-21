import pika
import json
import time
import cv2
import numpy as np
from skimage import morphology
from functools import reduce
from PIL import Image
import math
import os
import sys
import datetime
#图像评测核心算法
def makedir(pathd):
    if not os.path.exists(pathd):
        os.mkdir(pathd)


def graphyscore(modelname,studentname,id,word,type):
    #foldername = studentname.split(".")[0].split("/")[-1]
    timea = datetime.datetime.now()
    timefilename =  str(timea.year) + str(timea.month) + str(timea.day) + str(timea.hour) + str(timea.minute) + str(
        timea.second)
    #pathd =  foldername
    pathd1 = "/usr/local/CalligraphyData/"
    pathd3 = pathd1 + "User"
    pathd4 = pathd3 +"/" + str(id)
    pathd5 = pathd4 +"/" + word+"_"+type
    pathd6 = pathd5 + "/"+ timefilename
    pathd7 = pathd6 + "/" + "resultImg"
    makedir(pathd1)
    makedir(pathd3)
    makedir(pathd4)
    makedir(pathd5)
    makedir(pathd6)
    makedir(pathd7)

    print(pathd7)
    ######################################################################
    img1 = cv2.imread(modelname, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(studentname, cv2.IMREAD_GRAYSCALE)

    ###################################################################################
    width = 400
    height1 = int(width * img1.shape[0] / img1.shape[1])
    height2 = int(width * img2.shape[0] / img2.shape[1])

    img1 = cv2.resize(img1, (width, height1), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height2), interpolation=cv2.INTER_AREA)

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
    _, res2 = cv2.threshold(img2, 90, 255, cv2.THRESH_BINARY)
    res1 = num2true(res1)
    res2 = num2true(res2)
    res1 = morphology.remove_small_objects(res1, min_size=400, connectivity=1)
    res2 = morphology.remove_small_objects(res2, min_size=400, connectivity=1)
    res1 = true2num(res1)
    res2 = true2num(res2)
    image1 = res1.copy()
    image2 = res2.copy()

    # cv2.imshow('PART 1 presentation', image1)
    # cv2.imshow('PART 2 presentation', image2)
    # cv2.waitKey(0)
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
        point_color = (0, 0, 0)  # BGR
        # thickness = 1
        # lineType = 4
        # cv2.rectangle(image, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        return ptLeftTop, ptRightBottom

    def minrect(image):
        ptLeftTop, ptRightBottom = trans2whole(image)
        roi = image[ptLeftTop[1]:ptRightBottom[1], ptLeftTop[0]:ptRightBottom[0]]
        return roi

    newimage = np.zeros((400, 400), np.uint8)
    newimage.fill(255)

    def imgmatch(newimage, src, cx, cy):
        cx1 = cy1 = int(newimage.shape[0] / 2)
        deltax = cx1 - cx
        deltay = cy1 - cy
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                if i + deltax < newimage.shape[0] and j + deltay < newimage.shape[0]:
                    newimage[i + deltax][j + deltay] = src[i][j]
        return newimage

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
        return newimage1, newimage2

    newimage1, newimage2 = newimage(image1, image2, 400)


    strumodel1 = 255 - newimage1.copy()
    strumodel2 = 255 - newimage1.copy()
    strumodel3 = 255 - newimage1.copy()
    strumodel4 = 255 - newimage1.copy()
    strustudent1 = 255 - newimage2.copy()
    strustudent2 = 255 - newimage2.copy()
    strustudent3 = 255 - newimage2.copy()
    strustudent4 = 255 - newimage2.copy()
    path1 = pathd7 + "/premodel.png"
    path2 = pathd7 + "/prestudent.png"
    cv2.imwrite(path1, newimage1)
    cv2.imwrite(path2, newimage2)
    # cv2.imshow('pretreat model', newimage1)
    # cv2.imshow('pretreat student', newimage2)
    # cv2.waitKey(0)

    #########################################################################################
    def get_iou(target, prediction):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    iou1 = get_iou(255 - newimage1, 255 - newimage2)
    score1 = iou1 * 10 * (2 - iou1)
    imageshow1 = newimage1.copy()
    imageshow2 = newimage2.copy()
    imageshow2 = Image.fromarray(imageshow2).convert("RGB")
    imageshow2 = np.array(imageshow2)
    contours, hier = cv2.findContours(255 - newimage1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours1111, hier11111 = cv2.findContours(255 - newimage2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    imageshow2 = 255 - imageshow2
    cv2.drawContours(imageshow2, contours, -1, (0, 0, 255), 3)
    print("--------------PART 1---------------", score1)
    cv2.drawContours(imageshow2, contours, -1, (0, 0, 255), 3)
    # cv2.imshow('PART 1 presentation', imageshow2)
    # cv2.waitKey(0)
    path3 = pathd7 + "/part1.png"
    cv2.imwrite(path3, imageshow2)

    #########################################################################################
    imagestr = imageshow2.copy()
    thresh1 = 255 - newimage1
    thresh2 = 255 - newimage2

    contours1, hierarchy1 = cv2.findContours(thresh1, 3, 2)
    contours2, hierarchy2 = cv2.findContours(thresh2, 3, 2)
    centerdot1 = np.zeros(shape=(len(contours1), 1, 2), dtype=np.int)
    centerdot2 = np.zeros(shape=(len(contours2), 1, 2), dtype=np.int)

    for i in range(len(contours1)):
        cnt = contours1[i]
        hull = cv2.convexHull(cnt)
        # print(hull)
        cv2.polylines(res1, [hull], True, (0, 0, 0), 2)
        centerdot1[i, 0, 0] = int(center(hull)[0])
        centerdot1[i, 0, 1] = int(center(hull)[1])
        # print(centerdot1[i, 0, 0])

    for i in range(len(contours2)):
        cnt = contours2[i]
        hull = cv2.convexHull(cnt)
        # print(hull)
        cv2.polylines(res2, [hull], True, (0, 0, 0), 2)
        centerdot2[i, 0, 0] = int(center(hull)[0])
        centerdot2[i, 0, 1] = int(center(hull)[1])
        # print(centerdot2[i, 0, 0])

    cv2.polylines(thresh1, [centerdot1], True, (255, 255, 255), 2)
    cv2.polylines(thresh2, [centerdot2], True, (255, 255, 255), 2)

    contours3, hierarchy3 = cv2.findContours(thresh1, 3, 2)
    contours4, hierarchy4 = cv2.findContours(thresh2, 3, 2)

    cnt3 = contours3[0]
    cnt4 = contours4[0]
    hull3 = cv2.convexHull(cnt3)
    hull4 = cv2.convexHull(cnt4)
    cv2.polylines(thresh1, [hull3], True, (255, 255, 255), 1)
    cv2.polylines(imagestr, [hull3], True, (0, 0, 255), 6)
    cv2.polylines(imagestr, [hull4], True, (255, 0, 0), 3)
    cv2.polylines(thresh2, [hull4], True, (255, 255, 255), 1)
    cv2.fillPoly(thresh1, [hull3], (255, 255, 255))
    cv2.fillPoly(thresh2, [hull4], (255, 255, 255))

    iou2 = get_iou(thresh1, thresh2)
    score2 = iou2 * 10
    print("--------------PART 2---------------", iou2)
    # cv2.imshow('PART 2 presentation', imagestr)
    # cv2.waitKey(0)
    path4 = pathd7 + "/part2.png"
    cv2.imwrite(path4, imagestr)

    ######################################################################################################

    ###########################################################################
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        return cv2.warpAffine(image, M, (nW, nH))

    def statisticalhis(img):
        h = img.shape[0]
        w = img.shape[1]

        a = np.zeros([w, 1], np.float32)

        for j in range(0, w):
            for i in range(0, h):
                if img[i, j] > 0:
                    a[j, 0] += 1
                    img[i, j] = 0
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
    strumodel1 = 255 - strumodel1
    strustudent1 = 255 - strustudent1
    strumodel1 = Image.fromarray(strumodel1).convert("RGB")
    strumodel1 = np.array(strumodel1)
    strustudent1 = Image.fromarray(strustudent1).convert("RGB")
    strustudent1 = np.array(strustudent1)
    cv2.drawContours(strumodel1, contours, -1, (0, 0, 255), 3)
    cv2.drawContours(strustudent1, contours1111, -1, (255, 0, 0), 3)
    path4 = pathd7 + "/part3bone1.png"
    path5 = pathd7 + "/part3bone2.png"
    cv2.imwrite(path4, strumodel1)
    cv2.imwrite(path5, strustudent1)
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
    score3 = 10*(iou41 + iou42 + iou43 + iou44) / 4

    print("--------------PART 3---------------", score3
          )
    # 计算总分
    # total = 1.94064354*iou + 6.51982931*iou2 -0.96069589*similar1+ 4.23679642*similar2-2.637720298361252

    total = 2.88932039 * iou1 + 0.11458806 * iou2 + 2.44846675 * iou41 - 0.72886397 * match12 + 8.74964042 * iou42 - 4.9087531 * match22 + \
            10.0476079 * iou43 - 8.3394425 * match32 + 3.85927264 * iou44 - 2.05769276 * match42 + 1.4081326083885415
    if total >= 10:
        total = 10

    ########################################################################
    # total = 1.94064354*iou + 6.51982931*iou2 -0.96069589*similar1+ 4.23679642*similar2-2.637720298361252
    print("--------------PART 4---------------", total)
    return score1,score2,score3,total,pathd7
#生产
def product(user,key,score1,score2,score3,total,pathd,uid,id):
    print(sys.path)
    credentials = pika.PlainCredentials(user, key)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', port=5672, virtual_host='/', credentials=credentials))
    channel = connection.channel()
    result = channel.queue_declare(queue='Python2Java')
    dict = {}
    path = ""
    id = {"id": id}
    uid = {"uid": uid}
    score1 = {"score1":score1}
    score2 = {"score2": score2}
    score3 = {"score3": score3}
    total = {"total_score": total}
    result_imgFile_path = {"result_imgFile_path": pathd+ "/" }
    #prestudent_path = {"prestudent_path": path+user+"\\"+pathd +  "\\" + str(id1)+"prestudent.png"}
    # part1_path = {"part1_path": path+user+"\\"+pathd + "\\" + str(id1)+ "part1.png"}
    # part2_path = {"part2_path": path+user+"\\"+pathd +  "\\" + str(id1)+"part2.png"}
    # part31_path = {"part31_path": path+user+"\\"+pathd +  "\\" + str(id1)+"part3bone1.png"}
    # part32_path = {"part32_path": path+user+"\\"+pathd +  "\\" + str(id1)+"part3bone2.png"}
    dict.update(id)
    dict.update(uid)
    dict.update(score1)
    dict.update(score2)
    dict.update(score3)
    dict.update(total)
    dict.update(result_imgFile_path)
    # dict.update(prestudent_path)
    # dict.update(part1_path)
    # dict.update(part2_path)
    # dict.update(part31_path)
    # dict.update(part32_path)

    message = json.dumps(dict)
    channel.basic_publish(exchange='', routing_key='Python2Java', body=message)
    print(message)
    connection.close()


credentials = pika.PlainCredentials('SmwDsb', 'ietc`123')
connection = pika.BlockingConnection(pika.ConnectionParameters(host = 'localhost',port = 5672,virtual_host = '/',credentials = credentials))
channel = connection.channel()
# 申明消息队列，消息在这个队列传递，如果不存在，则创建队列
channel.queue_declare(queue = 'Java2Python', durable = False)
# 定义一个回调函数来处理消息队列中的消息，这里是打印出来
def callback(ch, method, properties, body):
    ch.basic_ack(delivery_tag = method.delivery_tag)
    body_json=json.loads(body)
    print(body_json)
    #print(body_json["characters_img_path"])
    #(body_json["user_img_path"])
    score1,score2,score3,total,pathd=graphyscore(body_json["production"]["characters_img_path"], body_json["production"]["user_img_path"],body_json["production"]["uid"],body_json["production"]["word"],body_json["production"]["typeface"])
    score1 = '%.2f' % (score1)
    score2 = '%.2f' % (score2)
    score3 = '%.2f' % (score3)
    total = '%.2f' % (total)
    product("SmwDsb","ietc`123", score1, score2, score3, total, pathd,body_json["production"]["uid"],body_json["production"]["id"])


channel.basic_consume('Java2Python',callback)

# 开始接收信息，并进入阻塞状态，
channel.start_consuming()

