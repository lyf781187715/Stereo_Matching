from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt


# cost = Variable(#5纬     1     ，channel*2 #2个特征图合并。32*2 = 64  ，   192/4 视差D ，     高        ，宽个0
#             torch.FloatTensor(1, 640, 48, 96,312).zero_())

# cost = Variable(torch.FloatTensor(1, 4, 8, 16,3).zero_())
# test1 = Variable(torch.FloatTensor(1,2,16,3).fill_(1))
# test2 = Variable(torch.FloatTensor(1,2,16,3).fill_(2))
# for i in range(8):
#     if i > 0:
#         cost[:, :test1.size()[1], i, :, i:] = test1[:, :, :, i:]
#         cost[:, test1.size()[1]:, i, :, i:] = test2[:, :, :, :-i]
#     else:#让第一组8*16*3的第一个16*3等于test1   第二组等于test2
#         cost[:, :test1.size()[1], i, :, :] = test1
#                   #0-2            0
#         cost[:, test1.size()[1]:, i, :, :] = test2
#                   #2-4            0
# cost2 = Variable(torch.FloatTensor(1, 1, 192, 4, 5).fill_(2))
# # print(cost2.size(0))
# print(cost2.size(1))
# print(cost2.size(2))
# print(cost2.size(3))
# print(cost2.size(4))

# print(cost)
# disp = Variable(torch.Tensor(np.reshape(np.array(range(192)),[1,192,1,1])).cpu(), requires_grad=False)
# disp = disp.repeat(1, 1, 4, 5)
# print(disp.size(0))
# print(disp.size(1))
# print(disp.size(2))
# print(disp.size(3))
# print('--------------')
# cost = torch.squeeze(cost2, 1)
# out = torch.sum(cost2*disp,1)
# print(out.size(0))
# print(out.size(1))
# print(out.size(2))
# print(out.size(3))
# print(out)

# list= [1,2,3,4,5,6,7,8]
# list2 = list [:-3]
# print(list2)

x = [0, 15]
x = range(15)
# y1 = [3.57,2.84,3.99,1.91,5.15,3.34,3.32,3.15,2.32,8.55,7.45,7.06,12.5,5.20,7.42]
# y2 = [3.71,2.78,4.57,2.72,7.36,4.28,3.44,3.76,2.35,12.6,11.5,8.56,14.0,5.35,8.87]
# y3 = [5.43,4.81,5.11,5.17,11.6,6.99,4.31,4.23,3.24,14.3,9.78,7.32,13.4,6.30,8.46]
# y4 = [4.41,3.98,5.40,3.17,10.0,6.23,4.62,4.77,3.49,12.7,12.4,10.4,14.5,7.80,8.85]


y1 = [9.63,7.80,9.18,8.96,24.4,8.78,11.9,12.8,6.91,20.6,16.5,14.6,25.8,15.2,15.4]
y2 = [9.68,7.90,9.06,9.73,24.2,8.70,11.9,12.9,7.72,22.3,20.4,16.6,28.8,16.7,16.1]
y3 = [10.9,9.40,11.8,9.08,24.5,14.8,18.4,18.9,8.51,22.9,29.8,17.2,22.5,19.6,14.6]
y4 = [9.29,7.97,11.9,9.96,27.2,14.4,13.9,15.0,10.1,24.5,23.9,17.2,26.1,17.7,17.0]

plt.plot(x, y1, marker = 'o', mec = 'r', mfc = 'w', label = u'MyNet')
plt.plot(x, y2, marker = '*', ms = 10, label = u'GC')
plt.plot(x, y3, marker = '+', ms = 10, label = u'SSB')
plt.plot(x, y4, marker = '.', ms = 10, label = u'SGBM')
plt.legend()
# plt.figure(num = 1,figsize= (10,10))
# line1,=plt.plot(x,y1,label='My net')
# line2,=plt.plot(x,y2,label='GC')
# line3,=plt.plot(x,y3,label='SGBM')
# line4,=plt.plot(x,y4,label='SSB')

plt.ylabel('The bad1.0 error rate in %',fontsize = 14)
plt.xlabel('15 pictures in Middlebury dataset',fontsize = 14, rotation = 0)
# plt.legend(loc='best',handles=[line1,line2,line3,line4],labels=['MyNet','GC','SGBM','SSB'])
#plt.xticks([0,0.75,2.25,3],['0','Nonocc_error','All_error','0'])
# plt.yticks([3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.5,6.0])
#x_label=['Nonocc_error','All_error']
plt.title('Comparing the results about bad1.0')
plt.tight_layout()
plt.show()