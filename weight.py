import numpy as np
import tensorflow as tf

def c2c_std(w1,b1,w2,alpha,p):
#std all
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    threshold=alpha*np.std(wa)
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(w1[m,k,j,i]>threshold):
                        count=count+1
               
        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w1o=np.array(w1o)
    w2o=np.array(w2[:,:,out,:])
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out

def c2c_std_abs(w1,b1,w2,alpha,p):
#std all abs
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    threshold=alpha*np.std(np.abs(wa))
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(np.abs(w1[m,k,j,i])>threshold):
                        count=count+1

        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w2o=w2[:,:,out,:]
    w1o=np.array(w1o)
    w2o=np.array(w2o)
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out

def c2c_std2(w1,b1,w2,alpha,p):
#std+ std-
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    pos=[wa[i] for i in range(wa.shape[0]) if wa[i]>0]
    neg=[wa[i] for i in range(wa.shape[0]) if wa[i]<0]
    threshold1=alpha*np.std(pos)
    threshold2=alpha*np.std(neg)
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(w1[m,k,j,i]>threshold1):
                        count=count+1
                    if(w1[m,k,j,i]<0 and np.abs(w1[m,k,j,i])>threshold2):
                        count=count+1
        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w2o=w2[:,:,out,:]
    w1o=np.array(w1o)
    w2o=np.array(w2o)
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out

def c2c_mean(w1,b1,w2,alpha,p):
## mean all
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    threshold=alpha*np.mean(wa)
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(w1[m,k,j,i]>threshold):
                        count=count+1

        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w2o=w2[:,:,out,:]
    w1o=np.array(w1o)
    w2o=np.array(w2o)
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out

def c2c_mean_abs(w1,b1,w2,alpha,p):
## mean all abs
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    threshold=alpha*np.mean(np.abs(wa))
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(np.abs(w1[m,k,j,i])>threshold):
                        count=count+1

        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w2o=w2[:,:,out,:]
    w1o=np.array(w1o)
    w2o=np.array(w2o)
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out

def c2c_mean2(w1,b1,w2,alpha,p):
#mean+ mean-
    sh=w1.shape
    wa=np.reshape(w1,sh[0]*sh[1]*sh[2]*sh[3])
    pos=[wa[i] for i in range(wa.shape[0]) if wa[i]>0]
    neg=[wa[i] for i in range(wa.shape[0]) if wa[i]<0]
    threshold1=alpha*np.mean(pos)
    threshold2=alpha*np.mean(neg)
    w1o=[]
    b1o=[]
    out=[]
    allc=sh[0]*sh[1]*sh[2]
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(w1[m,k,j,i]>threshold1 or w1[m,k,j,i]<threshold2):
                        count=count+1

        if((count/allc)>p):
            w1o.append(w1[:,:,:,i])
            b1o.append(b1[i])
            out.append(i)
    w2o=w2[:,:,out,:]
    w1o=np.array(w1o)
    w2o=np.array(w2o)
    w1o=w1o.transpose((1,2,3,0))
    b1o=np.array(b1o)
    return w1o,b1o,w2o,out
 
def acc_w_c2f(w,b,f1,alpha,p):
    sh=w.shape
    wa=np.reshape(w,sh[0]*sh[1]*sh[2]*sh[3])
    threshold=alpha*np.std(wa)
    wo=[]
    bo=[]
    out=[]
    f1o=[]
    allc=sh[0]*sh[1]*sh[2]
    per=int(f1.shape[0]/sh[3])
    for i in range(sh[3]):
        count=0
        for j in range(sh[2]):
            for k in range(sh[1]):
                for m in range(sh[0]):
                    if(w[m,k,j,i]>threshold):
                         count=count+1
        if((count/allc)>p):
            wo.append(w[:,:,:,i])
            bo.append(b[i])
            out.append(i)
    for i in range(per):
        temp=f1[i*sh[3]:(i+1)*sh[3],:]
        f1o.extend(temp[out,:])
    wo=np.array(wo)
    wo=wo.transpose((1,2,3,0))
    f1o=np.array(f1o)
    bo=np.array(bo)
    return wo,bo,f1o,out

def acc_w_f2f(f1,b1,f2,alpha,p):
    sh=f1.shape
    wa=np.reshape(f1,(1,sh[0]*sh[1]))
    threshold=alpha*np.std(wa)
    f1o=[]
    b1o=[]
    out=[]
    f2o=[]
    for i in range(sh[1]):
        count=0
        for j in range(sh[0]):
            if(f1[j,i]>threshold):
               count=count+1

        if(count/sh[0]>p):
            f1o.append(f1[:,i])
            b1o.append(b1[i])
            f2o.append(f2[i,:])  
            out.append(i)
    f1o=np.array(f1o)
    f1o=f1o.transpose((1,0))
    f2o=np.array(f2o)
    b1o=np.array(b1o)
    return f1o,b1o,f2o,out


