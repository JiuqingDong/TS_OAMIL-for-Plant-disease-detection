import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

xml_path0 = '/Users/jiuqingdong/Desktop/test/labels/'     # 你的xml文件路径
img_path0 = '/Users/jiuqingdong/Desktop/test/images/'         # 图像路径
img_xml  = '/Users/jiuqingdong/Desktop/test/show_xml/'       # 显示标注框保存该文件的路径

for set in ['train']:
# for set in ['unlabeled']:
    small = 0
    middle = 0
    large = 0
    xml_path = xml_path0 + set +'_xml'
    img_path = img_path0 + set
    for name in tqdm(os.listdir(xml_path)):
        image_name = os.path.join(img_path, name.split('.')[0] + '.jpg')
        if os.path.exists(image_name):
            # 打开xml文档
            tree = ET.parse(os.path.join(xml_path,name))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 得到文档元素对象
            root = tree.getroot()
            size = root.find('size')
            allObjects = root.findall('object')

            if len(allObjects) == 0:
                print("len(allObjects) == 0", name)
                continue


            if size is None:
                print("size is none")
                continue
            width = int(size.find('width').text)
            height = int(size.find('height').text)


            for i in range(len(allObjects)):    # 遍历xml标签，画框并显示类别。
                object = allObjects[i]

                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)

                w = xmax-xmin
                h = ymax-ymin
                relative_w = w/width
                relative_h = h/height
                if relative_w <=0.1:
                    small+=1
                elif relative_w<=0.3:
                    middle+=1
                else:
                    large+=1

                if relative_h <=0.1:
                    small+=1
                elif relative_h<=0.3:
                    middle+=1
                else:
                    large+=1


print("small = {}".format(small))
print("middle = {}".format(middle))
print("large = {}".format(large))

'''
for name in tqdm(os.listdir(xml_path0)):
    image_name = os.path.join(img_path0, name.split('.')[0] + '.jpg')

    if os.path.exists(image_name):
        # 打开xml文档
        tree = ET.parse(os.path.join(xml_path0, name))
        img = cv2.imread(image_name)
        box_thickness = int((img.shape[0] + img.shape[1])/1000) + 1  # 标注框的一个参数。本人图像大小不一致，在不同大小的图像上展示不同粗细的bbox

        text_thickness = box_thickness
        text_size = float(text_thickness/2)   # 显示标注类别的参数。字体大小。这些不是重点。不想要可以删掉。
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 得到文档元素对象
        root = tree.getroot()
        allObjects = root.findall('object')
        if len(allObjects) == 0:
            print("1 :", name)
            _image_name = str(image_name)
            _xml_name = _image_name.replace('.jpg', '.xml')
            _xml_name = _xml_name.replace('train', 'train_xml')
            if os.path.exists(_image_name):
                os.remove(_image_name)
            else:
                print("=========", _image_name)
            if os.path.exists(_xml_name):
                os.remove(_xml_name)
            else:
                print("=========", _xml_name)
            continue

        for i in range(len(allObjects)):    # 遍历xml标签，画框并显示类别。
            object = allObjects[i]
            objectName = object.find('name').text

            xmin = int(object.find('bndbox').find('xmin').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymax = int(object.find('bndbox').find('ymax').text)
            cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,0,0), text_thickness)
            cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)

            if len(allObjects) == 0:
                print("error")

        name = name.replace('xml', 'jpg')
        img_save_path = os.path.join(img_xml, name)
        cv2.imwrite(img_save_path, img)
'''