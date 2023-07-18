import pickle

#path='/home/multiai3/Jiuqing/OA-MIL/data/GWHD/paprika_train.pkl'#pkl文件所在路径
path='/home/multiai3/Jiuqing/OA-MIL-plants/data/GWHD/paprika_train_selected_control_class.pkl'#pkl文件所在路径
f=open(path,'rb')
data=pickle.load(f,encoding='latin1')
print(data)