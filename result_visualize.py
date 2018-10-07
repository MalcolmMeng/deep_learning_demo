import os
import json
import matplotlib.pyplot as plt
#the json data directory
dir=r'/home/malcolm/data/home/PycharmProjects/test_version'
resnet18='resnet18detail.json'
resnet50='resnet50detail.json'
inception='inception_v3detail.json'
with open(os.path.join(dir,resnet18),'r') as f:
    resnet18=json.load(f)
with open(os.path.join(dir,resnet50),'r') as f:
    resnet50=json.load(f)
with open(os.path.join(dir,inception),'r') as f:
    inception=json.load(f)

X = [i for i in range(25)]
#train loss compare
Y1=resnet18['train']['loss']
Y2=resnet50['train']['loss']
Y3=inception['train']['loss']
plt.title('train loss compare')
plt.xlabel('epoch')
plt.ylabel('loss value')
plt.plot(X,Y1,label='resnet18')
plt.plot(X,Y2,label='resnet50')
plt.plot(X,Y3,label='inception_v3')
plt.legend()
plt.show()
#val loss compare
Y1=resnet18['val']['loss']
Y2=resnet50['val']['loss']
Y3=inception['val']['loss']
plt.title('validation loss compare')
plt.xlabel('epoch ')
plt.ylabel('loss value')
plt.plot(X,Y1,label='resnet18')
plt.plot(X,Y2,label='resnet50')
plt.plot(X,Y3,label='inception_v3')
plt.legend()
plt.show()
#val err compare
Y1=resnet18['val']['err']
Y2=resnet50['val']['err']
Y3=inception['val']['err']
plt.title('validation error compare')
plt.xlabel('epoch ')
plt.ylabel('error value')
plt.plot(X,Y1,label='resnet18')
plt.plot(X,Y2,label='resnet50')
plt.plot(X,Y3,label='inception_v3')
plt.legend()

#accuray compare



plt.show()
