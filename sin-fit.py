from numpy.core.numeric import indices
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR

def generate_rvx(m,n):
    return  np.sort(np.random.rand(m,n),axis=0)
def ground_truth(rvx):
    vy = torch.sin( 2.*np.math.pi*rvx)
    g_noise = torch.empty(len(rvx)).normal_(mean=0,std=(0.05)**2)
    vy[:,0] += g_noise[:]
    return vy
    #return torch.sin( 2.*np.math.pi*rvx)+10*torch.empty(len(rvx)).normal_(mean=0,std=(0.05)**2)


dv = [3,5,9,16,28,50,100,200,1000]
#dv = [3,5,9,16,28,50,100]
for i  in dv:
    #numper of predictions
    number_p = 100
    xx = torch.tensor( np.linspace(0,1,number_p) )

    #DATA points
    dp = i
    #vx = torch.linspace(0,1,100)
    vx = torch.tensor( generate_rvx(dp,1) )
    #vx[:,0] = xx[:dp]
    #scale_factor = 1/max(vx)
    #vx[:,0] *= scale_factor 
    vy = ground_truth(vx)


    model = SVR(kernel = 'poly', C=10 , gamma='auto' , degree=10 , epsilon=0.0,coef0=1)

    print(vx.size())

    X = torch.zeros((number_p,1))

    X[:,0] = xx[:]
    #print(X)
    v_r = model.fit(vx,vy).predict(X)

    Real_y=ground_truth(X)

    #plt.plot(vx,vy[:,1],'o',X,v_r,X,Real_y[:,1])
    plt.plot(X,v_r,X,Real_y)
    plt.show()
