from PIL import Image
import os
import glob
import json
import base64

# 数据集路径
path = ".\\data\\"
# 生成数据的保存路径
save_path = ".\\data\\"
# 当前数据集图片格式
file_format = ".jpg"
# 替换格式jpg -> json
replace_format = ".json"
# 左右翻转文件名附加字符
LR = "lr_"
# 上下翻转文件名附加字符
TB = "tb_"
# 获取数据集目录的图片数据集
img_list = glob.glob(path + "*" + file_format)

print("左右翻转-start")
# 1.遍历图片
for i in range(len(img_list)):
    # 图片路径
    img_path = img_list[i]
    # 对应json路径
    json_path = img_list[i].replace(file_format, replace_format)
    # 判断json文件是否存在
    is_exists = os.path.exists(json_path)
    if is_exists:
        # 打开json文件
        f = open(json_path, encoding='utf-8')
        # 读取json
        setting = json.load(f)
        # 获取当前图片尺寸
        width = setting['imageWidth']
        height = setting['imageHeight']
        # 获取中轴
        mid_width = width / 2
        mid_height = height / 2

        print("中轴：x-" + str(mid_width) + ",y-" + str(mid_height))
        # 2.遍历shapes
        for i2 in range(len(setting['shapes'])):
            # 3.遍历每个shapes的点
            for i3 in range(len(setting['shapes'][i2]['points'])):
                temp_x = setting['shapes'][i2]['points'][i3][0]
                temp_y = setting['shapes'][i2]['points'][i3][1]
                if temp_x > mid_width:
                    dis = temp_x - mid_width
                    new_x = mid_width - dis
                elif temp_x < mid_width:
                    dis = mid_width - temp_x
                    new_x = mid_width + dis
                else:
                    new_x = temp_x
                new_y = temp_y
                setting['shapes'][i2]['points'][i3][0] = new_x
                setting['shapes'][i2]['points'][i3][1] = new_y
        # 从json获取文件名
        file_name = setting['imagePath']
        # 修改json文件名
        setting['imagePath'] = LR + file_name
        full_path = save_path + LR + file_name
        full_path = full_path.replace(file_format, replace_format)
        # 图片转换
        pri_image = Image.open(img_path)
        # 左右镜面翻转FLIP_LEFT_RIGHT
        pri_image.transpose(Image.FLIP_LEFT_RIGHT).save(path + LR + file_name)
        # 将转换后的图片进行base64加密
        with open(path + LR + file_name, 'rb') as f:
            setting['imageData'] = base64.b64encode(f.read()).decode()
        string = json.dumps(setting)
        # 将修改后的json写入文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(string)
            f.close()
        print(img_path + "-------转换完成")
        setting = None
    else:
        print(json_path + "-------文件不存在")
print("左右翻转-end")
# 原理同上
print("上下翻转-start")
for i in range(len(img_list)):
    img_path = img_list[i]
    json_path = img_list[i].replace(file_format, replace_format)
    is_exists = os.path.exists(json_path)
    if is_exists:
        f = open(json_path, encoding='utf-8')
        setting = json.load(f)
        width = setting['imageWidth']
        height = setting['imageHeight']
        mid_width = width / 2
        mid_height = height / 2

        for i2 in range(len(setting['shapes'])):
            for i3 in range(len(setting['shapes'][i2]['points'])):
                temp_x = setting['shapes'][i2]['points'][i3][0]
                temp_y = setting['shapes'][i2]['points'][i3][1]
                if temp_y > mid_height:
                    dis = temp_y - mid_height
                    new_y = mid_height - dis
                elif temp_y < mid_height:
                    dis = mid_height - temp_y
                    new_y = mid_height + dis
                else:
                    new_y = temp_y
                new_x = temp_x
                setting['shapes'][i2]['points'][i3][0] = new_x
                setting['shapes'][i2]['points'][i3][1] = new_y

        file_name = setting['imagePath']
        setting['imagePath'] = TB + file_name
        full_path = save_path + TB + file_name
        full_path = full_path.replace(file_format, replace_format)
        pri_image = Image.open(img_path)
        # 上下镜面翻转FLIP_TOP_BOTTOM
        pri_image.transpose(Image.FLIP_TOP_BOTTOM).save(path + TB + file_name)
        with open(path + TB + file_name, 'rb') as f:
            setting['imageData'] = base64.b64encode(f.read()).decode()
        string = json.dumps(setting)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(string)
            f.close()
        print(img_path + "-------转换完成")
        setting = None
    else:
        print(json_path + "-------文件不存在")
print("上下翻转-end")

