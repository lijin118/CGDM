

# 通用参数
dataset_root = "dataset"
batch_size = 32
num_workers = 4
device = "cuda:0"
lr = 0.001
N = 4                    # The times that generator updates within a batch 
num_epoch = 300
models_save = "models_trained"
weight_cls = 0.01
weight_gmn = 0.01
weight_ent = 0.01
weight_dis = 1.0
class_num = 31 
smooth = 0.1
bottleneck = 256
#layer= "wn"
#classifier= "bn"
Q = 0.5
Z = 0.5
gmn_N = class_num      #The number of classes to calulate gradient similarity
resnet = '50'
res_cls_num_layer = 2

#dataset
category = 'office31'          #'digit','office31'
trans = "amazon2dslr"           # office31{"amazon", "dslr", "webcam"} digit{"s2m","u2m","m2u"}


#optimizer
optimizer = 'sgd'