import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id          #
PRE_DEFINE_CATEGORIES = {"blossom_end_rot":1, "graymold":2, "powdery_mildew":3, "spider_mite":4,"spotting_disease":5, 'background':6, 'healthy': 7, 'unknown': 8}

# all_class = ["blossom_end_rot", "graymold", "powdery_mildew", "spider_mite","spotting_disease", 'background', 'healthy', 'unknown']

def get(root, name):
    vars = root.findall(name)
    return vars



def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        end_with = filename.split('.')[-1]
        if end_with == 'xml':
            end_with = 'jpg'
        filename = os.path.splitext(filename)[0]
        filename = filename.split('\\')
        filename = filename[-1]
        filename = filename+'.'+end_with
        if 'blossom_end_rot' in filename:
            id = filename.split('.')[0]
            id = id.replace('blossom_end_rot_', '1')
        elif 'graymold' in filename:
            id = filename.split('.')[0]
            id = id.replace('graymold_', '2')
        elif 'powdery_mildew' in filename:
            id = filename.split('.')[0]
            id = id.replace('powdery_mildew_', '3')
        elif 'spider_mite' in filename:
            id = filename.split('.')[0]
            id = id.replace('spider_mite_', '4')
        elif 'spotting_disease' in filename:
            id = filename.split('.')[0]
            id = id.replace('spotting_disease_', '5')
        else:
            print("there is something wrong!")
        ID = int(id)
        # print("filename is {}, ID is {}".format(filename, ID))
        return filename, ID
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        # print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number

        filename, image_id = get_filename_as_int(line)

        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)

        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text

            # print("category is ",category)
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax >= xmin)
            assert(ymax >= ymin)

            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':


    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_control_class/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    #for i in range(50):
    #    print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_control_class/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_control_class.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    # for i in range(50):
    #     print("=======================================================================================================")
    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_control_class/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)
'''
    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise5/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise5.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise5/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise5.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise5/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise5.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise10/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise10.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise10/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise10.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise10/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise10.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise15/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise15.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise15/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise15.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise15/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise15.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise20/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise20.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise20/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise20.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise20/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise20.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise25/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise25.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise25/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise25.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise25/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise25.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise30/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise30.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise30/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise30.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise30/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise30.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise35/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise35.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise35/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise35.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise35/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise35.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise40/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise40.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise40/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise40.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise40/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise40.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise45/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise45.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise45/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise45.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise45/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise45.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    val_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt"
    val_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise50/val_xml"
    val_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_val_position_noise50.json"
    convert(val_XML_LIST, val_XML_DIR, val_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    train_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt"
    train_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise50/train_xml"
    train_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_train_position_noise50.json"
    convert(train_XML_LIST, train_XML_DIR, train_OUTPU_JSON)

    for i in range(50):
        print("=======================================================================================================")

    test_XML_LIST = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt"
    test_XML_DIR = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/labels_position_noise50/test_xml"
    test_OUTPU_JSON = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/annotations/instances_test_position_noise50.json"
    convert(test_XML_LIST, test_XML_DIR, test_OUTPU_JSON)
'''
