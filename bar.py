import matplotlib.pyplot as plt
import numpy as np


scale_ls = np.arange(4)
data = [85.1, 88.0, 88.4, 89.5]
y_label = ["{:.1f}".format(_y) for _y in data]

plt.gcf().subplots_adjust(bottom=0.15)
plt.tick_params(labelsize=15)

plt.rc('font',family='Times New Roman') 
#plt.title("Microwave",fontdict={'family' : 'Times New Roman', 'size'   : 25})
plt.xlabel("Settings",fontdict={'family' : 'Times New Roman', 'size'   : 25})

plt.ylabel("Accuracy (%)",fontdict={'family' : 'Times New Roman', 'size' : 25})

index_ls = ['w/o both','w/o sup',' w/o gdm','with both']

plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字

plt.ylim(80,95)

plt.bar(scale_ls, data)

for a, b, label in zip(scale_ls, data, y_label):
    plt.text(a, b, label, ha='center', va='bottom',fontdict={'family' : 'Times New Roman', 'size'   : 25})


plt.savefig('ablation.pdf')