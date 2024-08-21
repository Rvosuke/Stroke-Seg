import tradition as trad
import deeplearning as deep
import joblib
import pika
import json
import time
import cv2
import numpy as np
import os
import datetime
import sys

def makedir(pathd):
    if not os.path.exists(pathd):
        os.mkdir(pathd)

def mkdir(id, word,type):
    timea = datetime.datetime.now()
    timefilename = str(timea.year) + str(timea.month) + str(timea.day) + str(timea.hour) + str(timea.minute) + str(
        timea.second)
    # pathd =  foldername
    pathd1 = "/usr/local/CalligraphyData/"
    pathd3 = pathd1 + "User"
    pathd4 = pathd3 + "/" + str(id)
    pathd5 = pathd4 + "/" + word + "_" + type
    pathd6 = pathd5 + "/" + timefilename
    pathd7 = pathd6 + "/" + "resultImg"
    makedir(pathd1)
    makedir(pathd3)
    makedir(pathd4)
    makedir(pathd5)
    makedir(pathd6)
    makedir(pathd7)

    return pathd7


def main(modelname,studentname,id,word,type):
    pathd7 = mkdir(id, word,type)
    t1 = time.time()
    original_img_path = modelname
    template_img_path = studentname

    feature2, img1, img2, newimage1, newimage2, imageshow2, imagestr, \
    size_comment, position_comment, cx_comment = trad.main(original_img_path, template_img_path)

    path1 = pathd7 + "/premodel.png"
    path2 = pathd7 + "/prestudent.png"
    cv2.imwrite(path1, newimage1)
    cv2.imwrite(path2, newimage2)

    path3 = pathd7 + "/part1.png"
    cv2.imwrite(path3, imageshow2)

    path4 = pathd7 + "/part2.png"
    cv2.imwrite(path4, imagestr)


    # print(img)
    feature1 = deep.main(img1, img2)

    # print(feature1,feature2)

    # 读取模型
    model = joblib.load('./fusion.pkl')

    test = np.array(feature1 + feature2)
    # print(test.shape)
    t2 = time.time()
    result = model.predict(test.reshape((1,22)))
    print(t2-t1, feature2)
    return result, feature1, feature2, pathd7, size_comment, position_comment, cx_comment


def product(user,key,score1,score2,score3,total,pathd,uid,id, size_comment, position_comment, cx_comment):
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
    size_comment = {"evolute1": size_comment}
    position_comment = {"evolute2": position_comment}
    cx_comment = {"evolute3": cx_comment}
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
    dict.update(size_comment)
    dict.update(position_comment)
    dict.update(cx_comment)
    dict.update(result_imgFile_path)

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

def mapping(score):
    return score * (40.0 - score) / 30.0


def callback(ch, method, properties, body):
    ch.basic_ack(delivery_tag = method.delivery_tag)
    body_json=json.loads(body)
    print(body_json)
    #print(body_json["characters_img_path"])
    #(body_json["user_img_path"])
    result, feature1, feature2, pathd7, size_comment, position_comment, cx_comment = main(body_json["production"]["characters_img_path"], body_json["production"]["user_img_path"],body_json["production"]["uid"],body_json["production"]["word"],body_json["production"]["typeface"])
    result = mapping(result)
    score1 = '%.2f' % (feature2[2])
    score2 = '%.2f' % (feature2[1]*10)
    score3 = '%.2f' % (feature1[-1])
    total = '%.2f' % (result)
    print(score1, score2, score3, total, pathd7)
    product("SmwDsb","ietc`123", score1, score2, score3, total, pathd7, body_json["production"]["uid"],body_json["production"]["id"], size_comment, position_comment, cx_comment)


channel.basic_consume('Java2Python',callback)

# 开始接收信息，并进入阻塞状态，
channel.start_consuming()
