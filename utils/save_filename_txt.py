import os

train_img = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train"
val_img = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val"
test_img = "/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test"

train_name = os.listdir(train_img)
val_name = os.listdir(val_img)
test_name = os.listdir(test_img)

xml_train_path = open('/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/train.txt', 'w')
xml_val_path = open('/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/val.txt', 'w')
xml_test_path = open('/Users/jiuqingdong/Documents/Pycharm_Project_Code/OA-MIL-main/data/GWHD/test.txt', 'w')

for i in train_name:
    i = i.split('.')[0]
    xml_train_path.write(i + '.xml' + '\n')

for i in val_name:
    i = i.split('.')[0]
    xml_val_path.write(i + '.xml' + '\n')

for i in test_name:
    i = i.split('.')[0]
    xml_test_path.write(i + '.xml' + '\n')