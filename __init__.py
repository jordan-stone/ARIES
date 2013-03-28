#!/usr/bin/python
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as mpl
from scipy.interpolate import griddata
import jFits, poly2d
import subprocess, os
from readcol import readcol
import gaussfitter
mpl.ion()

__all__=['readcol','subprocess','os','jFits','poly2d','gauss_func','gaussfitter',\
         'griddata','mpl','ma','np','smooth_climb','climb','get_gauss','unzip']

def gauss_func(mu,sigma):#build a function...
    return lambda x: np.exp(-0.5*( (x-mu)**2. / sigma**2. ))

def smooth_climb(y,im=None,column=None,kernlength=5.):
    col=im[:,column]
    padded=np.zeros(3*len(col))
    padded[len(col):2*len(col)]=col
    kern=np.zeros_like(padded)
    mid=len(col)+int(len(col)/2.)
    kern[mid-int(kernlength/2.):mid+int(kernlength/2.)+(kernlength % 2)]=1./kernlength
    smoothed=np.fft.fftshift(np.fft.ifft(np.fft.fft(kern)*np.fft.fft(padded)))
    smoothed=smoothed[len(col):2*len(col)]
    
    y=int(y)
    up=(np.where(smoothed[y-1:y+2]==np.max(smoothed[y-1:y+2])))[0][0]
    while up<>1 and y > 2 and y < 1022:
        y+=(up-1)
        up=(np.where(smoothed[y-1:y+2]==np.max(smoothed[y-1:y+2])))[0][0]
    return y

def climb(y,av=None):
    inp=y
    y=int(y)
    up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
    while up<>1 and y > 2 and y < len(av)-2:
        y+=(up-1)
        up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
    return y

def get_gauss(snippet,absorption=False):
    snippet1=snippet.copy()
    snippet1[np.isnan(snippet1)]=0
    x=np.arange(len(snippet1))
    snippet1[snippet<0]=0#get rid of any negatives, why are there negatives? totally biasing things here..(prob)
    if sum(snippet1)==0:
        return len(x)/2., -1
    else:
        if absorption:
            snippet1=-1.*snippet1-min(-1.*snippet1)
            mu=sum(x*snippet1)/sum(snippet1)
            sigma=np.sqrt(abs( sum( (x-mu)**2*snippet1 )/sum( snippet1 ) ) )
        else:
            mu=sum(x*snippet1)/sum(snippet1)
            sigma=np.sqrt(abs( sum( (x-mu)**2*snippet1 )/sum( snippet1 ) ) )
        return mu, sigma

def unzip(zippedList):
    return zip(*zippedList)

