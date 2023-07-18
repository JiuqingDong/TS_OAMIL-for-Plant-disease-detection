import xml.etree.ElementTree as ET
from lxml import etree


def get_new_value(object):
    category = str(object[0])
    xmin = int(object[1])
    ymin = int(object[2])
    xmax = int(object[3])
    ymax = int(object[4])
    score= object[5]
    if xmax <= xmin or ymax <= ymin:
        print("xmin,xmax,ymin,ymax", xmin, xmax, ymin, ymax)

    return category, xmin, ymin, xmax, ymax, score


def get_value(category, xmin, ymin, xmax, ymax):
    category = str(category.text)
    xmin = int(xmin.text)
    ymin = int(ymin.text)
    xmax = int(xmax.text)
    ymax = int(ymax.text)

    return category, xmin, ymin, xmax, ymax


def my_IOU(rec_1, rec_2):
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])  # 第一个bbox面积 = 长×宽
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])  # 第二个bbox面积 = 长×宽
    sum_s = s_rec1 + s_rec2  # 总面积
    left = max(rec_1[0], rec_2[0])  # 并集左上角顶点横坐标
    right = min(rec_1[2], rec_2[2])  # 并集右下角顶点横坐标
    bottom = max(rec_1[1], rec_2[1])  # 并集左上角顶点纵坐标
    top = min(rec_1[3], rec_2[3])  # 并集右下角顶点纵坐标
    if left >= right or top <= bottom:  # 不存在并集的情况
        return 0, s_rec1, s_rec2
    else:
        inter = (right - left) * (top - bottom)  # 求并集面积
        iou = (inter / (sum_s - inter)) * 1.0  # 计算IOU

    return iou, s_rec1, s_rec2


def selected_bbox(rec_1, rec_2):
    iou, s_rec1, s_rec2 = my_IOU(rec_1, rec_2)
    if iou == 0:
        return True
    elif 0<iou<0.25:
        if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
            return True
        elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
            return False
        else:
            return True
    elif 0.25<= iou <=0.7:
        if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
            return True
        elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
            return False
        elif s_rec1>s_rec2:
            return True
        else:
            return False
    elif iou>0.7:
        if s_rec1 >= s_rec2:
            return True
        else:
            return False
    else:
        raise EOFError


def selected_bbox_2(rec_1, rec_2, c1, c2, score_1, score_2):
    if score_1<0.99 and c1 in ["blossom_end_rot", "graymold", "spotting_disease", "powdery_mildew"]:
        return False
    if score_1<0.7 and c1 in ["spider_mite"]:
        return False
    iou, s_rec1, s_rec2 = my_IOU(rec_1, rec_2)

    if c1 == c2:
        if iou <= 0.20:
            return True
        # elif 0.1<iou<0.25:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     else:
        #         return True
        elif iou>0.20:
            if s_rec1 >= s_rec2:
                return True
            else:
                return False
        else:
            raise EOFError

    elif c1 != c2:
        if iou <0.5:
            return True
        # elif 0.25 <= iou <= 0.5:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     elif s_rec1 > s_rec2:
        #         return True
        #     else:
        #         return False
        elif iou > 0.5:
            if score_1 > score_2:
                return True
            elif score_1< score_2:
                return False
            else:
                if s_rec1 >= s_rec2:
                    return True
                else:
                    return False
        else:
            raise EOFError


def selected_bbox_3(rec_1, rec_2, c1, c2, score_1, score_2):
    iou, s_rec1, s_rec2 = my_IOU(rec_1, rec_2)
    if score_1<0.8 and c1 in ["wheat"]:
        return False
    if c1 == c2:
        if iou <= 0.60:
            return True
        # elif 0.1<iou<0.25:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     else:
        #         return True
        elif iou>0.6:
            if s_rec1 >= s_rec2:
                return True
            else:
                return False
        else:
            raise EOFError


def add_node(root, label, xmin, ymin, xmax, ymax):
    # deta_x = xmax - xmin
    # deta_y = ymax -ymin
    # xmin = int(xmin +0.02*deta_x)
    # ymin = int(ymin +0.02*deta_y)
    # xmax = int(xmax -0.02*deta_x)
    # ymax = int(ymax -0.03*deta_y)


    object = etree.Element("object")
    namen = etree.SubElement(object, "name")
    namen.text = label
    object.append(namen)
    pose = etree.SubElement(object, "pose")
    pose.text = str(0)
    object.append(pose)
    truncated = etree.SubElement(object, "truncated")
    truncated.text = str(0)
    object.append(truncated)
    difficult = etree.SubElement(object, "difficult")
    difficult.text = str(0)
    object.append(difficult)
    bndbox = etree.SubElement(object, "bndbox")
    xminn = etree.SubElement(bndbox, "xmin")
    xminn.text = str(xmin)
    bndbox.append(xminn)
    yminn = etree.SubElement(bndbox, "ymin")
    yminn.text = str(ymin)
    bndbox.append(yminn)
    xmaxn = etree.SubElement(bndbox, "xmax")
    xmaxn.text = str(xmax)
    bndbox.append(xmaxn)
    ymaxn = etree.SubElement(bndbox, "ymax")
    ymaxn.text = str(ymax)
    root.getroot().append(object)


def generate_xml(object_list, xml_file, target):
    target_file = xml_file.replace('labels', target)

    parser = etree.XMLParser(remove_blank_text=True)  #
    root = etree.parse(xml_file, parser)
    tree = etree.ElementTree(root.getroot())

   # old_all_objects = {"blossom_end_rot": [], "graymold": [], "spotting_disease": [], "spider_mite": [],
   #                "powdery_mildew": []}

    all_objects = {"blossom_end_rot": [], "graymold": [], "spotting_disease": [], "spider_mite": [],
                   "powdery_mildew": []}

    for object in tree.findall('object'):
        category = object.find('name')
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        xmin = bndbox.find('xmin')  # type(xmin) = int
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
#
        old_category, old_xmin, old_ymin,  old_xmax, old_ymax = get_value(category, xmin, ymin, xmax, ymax)
#
        if old_category in ["blossom_end_rot", "graymold", "spotting_disease", "spider_mite", "powdery_mildew"]:
            # old_all_objects[old_category].append([old_xmin, old_ymin,  old_xmax, old_ymax])
            parent = object.getparent()
            parent.remove(object)

    for obj in object_list:
        classes, xmin, ymin, xmax, ymax, score = get_new_value(obj)
        # add_node(root, classes, xmin, ymin, xmax, ymax)

        if classes in ["blossom_end_rot", "graymold", "spotting_disease", "spider_mite", "powdery_mildew"]:
            all_objects[classes].append([xmin, ymin, xmax, ymax])

    for one_class in all_objects:
        if len(all_objects[one_class]) == 0:
            continue
        elif len(all_objects[one_class])==1:
            [xmin, ymin, xmax, ymax] = all_objects[one_class][0]
            add_node(root, one_class, xmin, ymin, xmax, ymax)
        else:
            for n1 in range(len(all_objects[one_class])):
                rec_1 = all_objects[one_class][n1]
                Save = True
                for n2 in range(len(all_objects[one_class])):
                    if n1 != n2:
                        rec_2 = all_objects[one_class][n2]
                        Save = selected_bbox(rec_1, rec_2)
                    if Save is False:
                        break

                if Save is True:
                    [xmin, ymin, xmax, ymax] = all_objects[one_class][n1]
                    add_node(root, one_class, xmin, ymin, xmax, ymax)
                    # print("Multibbox",all_objects[one_class][n1])
    tree.write(target_file, pretty_print=True, xml_declaration=False, encoding='utf-8')


def generate_xml_2(object_list, xml_file, target):
    target_file = xml_file.replace('labels', target)

    parser = etree.XMLParser(remove_blank_text=True)  #
    root = etree.parse(xml_file, parser)
    tree = etree.ElementTree(root.getroot())

    for object in tree.findall('object'):
        category = str(object.find('name').text)
        if category not in ['background', 'healthy', 'unknown']:
            parent = object.getparent()
            parent.remove(object)

    if len(object_list) == 0:
        pass

    elif len(object_list) == 1:
        obj = object_list[0]
        classes, xmin, ymin, xmax, ymax, score = get_new_value(obj)
        if classes not in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
            pass
        else:
            add_node(root, classes, xmin, ymin, xmax, ymax)

    elif len(object_list)>1:

        for n1 in range(len(object_list)):
            classes_1, xmin_1, ymin_1, xmax_1, ymax_1, score_1 = get_new_value(object_list[n1])
            if classes_1 in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
                rec_1 = [xmin_1, ymin_1, xmax_1, ymax_1]
                Save = True

                for n2 in range(len(object_list)):
                    classes_2, xmin_2, ymin_2, xmax_2, ymax_2, score_2 = get_new_value(object_list[n2])

                    if n1!=n2 and classes_2 in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
                        rec_2 = [xmin_2, ymin_2, xmax_2, ymax_2]
                        Save = selected_bbox_2(rec_1, rec_2, classes_1, classes_2, score_1, score_2)

                    if Save is False:
                        break

                if Save is True:
                    add_node(root,  classes_1, xmin_1, ymin_1, xmax_1, ymax_1)
        for bhu in range(len(object_list)):
            classes_1, xmin_1, ymin_1, xmax_1, ymax_1, score_1 = get_new_value(object_list[n1])
            if classes_1 in ['background', 'healthy', 'unknown']:
                if score_1 > 0.9:
                    add_node(root,  classes_1, xmin_1, ymin_1, xmax_1, ymax_1)

    tree.write(target_file, pretty_print=True, xml_declaration=False, encoding='utf-8')


def generate_xml_3(object_list, xml_file, target):
    target_file = xml_file.replace('labels', target)

    parser = etree.XMLParser(remove_blank_text=True)  #
    root = etree.parse(xml_file, parser)
    tree = etree.ElementTree(root.getroot())

    for object in tree.findall('object'):
        category = str(object.find('name').text)
        if category not in ['background', 'healthy', 'unknown']:
            parent = object.getparent()
            parent.remove(object)

    if len(object_list) == 0:
        pass

    elif len(object_list) == 1:
        obj = object_list[0]
        classes, xmin, ymin, xmax, ymax, score = get_new_value(obj)
        if classes not in ['wheat',]:
            pass
        else:
            add_node(root, classes, xmin, ymin, xmax, ymax)

    elif len(object_list)>1:

        for n1 in range(len(object_list)):
            classes_1, xmin_1, ymin_1, xmax_1, ymax_1, score_1 = get_new_value(object_list[n1])
            if classes_1 in ['wheat',]:
                rec_1 = [xmin_1, ymin_1, xmax_1, ymax_1]
                Save = True

                for n2 in range(len(object_list)):
                    classes_2, xmin_2, ymin_2, xmax_2, ymax_2, score_2 = get_new_value(object_list[n2])

                    if n1!=n2 and classes_2 in ['wheat',]:
                        rec_2 = [xmin_2, ymin_2, xmax_2, ymax_2]
                        Save = selected_bbox_3(rec_1, rec_2, classes_1, classes_2, score_1, score_2)

                    if Save is False:
                        break

                if Save is True:
                    if xmin_1 < xmax_1-2 and ymin_1 < ymax_1-2:
                        add_node(root,  classes_1, xmin_1, ymin_1, xmax_1, ymax_1)
                    else:
                        print("========")

    tree.write(target_file, pretty_print=True, xml_declaration=False, encoding='utf-8')
