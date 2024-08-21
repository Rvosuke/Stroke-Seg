import sys
import os
import cv2
import numpy as np
from random import choice
import time
import json
import base64

from math import cos, sin, pi, fabs, radians

label = []


# 读取json
def readJson(jsonfile):
    with open(jsonfile, encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData


# 转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


# 坐标旋转
def rotatePoint(jsonTemp):
    for key, value in jsonTemp.items():
        if key == 'shapes':
            new = []
            for item in value:
                # print(item)
                labelnum = item['label']
                if labelnum not in label:
                    label.append(labelnum)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


# 保存json
def writeToJson(filePath, data):
    fb = open(filePath, 'w')
    fb.write(json.dumps(data, cls=NpEncoder))  # ,encoding='utf-8'
    fb.close()


if __name__ == '__main__':

    path_json = './dataview/json_ori'
    json_write = './dataview/json3_new'

    for json_file in os.listdir(path_json):
        json_read = os.path.join(path_json, json_file)
        # print(json_read)
        jsonData = readJson(json_read)

        json_dir = os.path.join(json_write, json_file)

        rotatePoint(jsonData)

    print(label)
