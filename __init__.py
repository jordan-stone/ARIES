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
         'griddata','mpl','ma','np','smooth_climb','climb','get_gauss','unzip',\
         'Features']

speed_of_light=float(299792458)#m/s

class Features:
    def __init__(self):
        self.Rydberg=1.0973731569e7#m^-1
        self.lines={'CO_bandheads':np.array([2.2936,2.3227,2.3527,2.38298]),
                    'NIR_Brackett':np.array(map(self.Brackett,xrange(5,20))),
                    'H2':1e4/np.array([4917.,4713.,4498.])}
        all_lines=[]
        for k in self.lines.keys():
            all_lines.extend(self.lines[k].tolist())
        self.lines['all_lines']=np.array(all_lines)
    def Brackett(self,upper):
        inv_lambda=self.Rydberg*((1/16.)-(1/float(upper)**2))
        return (1/inv_lambda)*1.e6
    def redshift(self,velocity):
        self.__init__()
        for k in self.lines.keys():
            self.lines[k]=(1+velocity/speed_of_light)*self.lines[k]
        
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

