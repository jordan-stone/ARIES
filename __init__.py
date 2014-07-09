#!/usr/bin/python
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as mpl
from scipy.interpolate import griddata
import jFits, poly2d
import subprocess, os
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
    '''Return a gaussian function with mean mu and standard 
    deviation sigma.
    Inputs:
    mu    - the mean of the distribution
    sigma - the standard deviation of the distribution
    Returns:
    function - the requested gaussian at any input 
    '''
    return lambda x: np.exp(-0.5*( (x-mu)**2. / sigma**2. ))

def smooth_climb(y,im=None,column=None,kernlength=5.):
    '''smooth a column with a tophat kernel, and then 
    climb uphill along that column from a starting location
    Inputs:
    y          - [float] the initial y coordinate of the starting location
    im         - [2d array] the image
    column     - [int] the initial x-coordinate of the starting loc.
    kernlength - [float] the width of the tophat smoothing kernel
    Returns:
    y_apex     - [int] the local maximum of the smoothed column
                       closest to the input location
    '''
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
    '''climb uphill along that column from a starting location
    Inputs:
    y          - [float] the initial y coordinate of the starting location
    av         - [2d array] the image
    Returns:
    y_apex     - [int] the local maximum of the column
                       closest to the input location
    '''
    inp=y
    y=int(y)
    up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
    while up<>1 and y > 2 and y < len(av)-2:
        y+=(up-1)
        up=(np.where(av[y-1:y+2]==np.max(av[y-1:y+2])))[0][0]
    return y

def get_gauss(snippet,absorption=False):
    '''Return the first and second moment of a distribution
    Inputs:
    snippet    -[1d array] the distribution of y-values (x-values are assumed
                         to be the indices of the values
    absorption -[bool, default=False] flag whether the distribution
                                       is upside down (a local minimum)
                                       or rightside up (a local max)
    Returns:
    mu,sig     -[tuple] the first and second moment respectively in
                        units of the input array indices...
    '''
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
    '''return a previously zipped list to it's original components'''
    return zip(*zippedList)


def readcol(filename,delimiter=None,colNames=[],skipLines=0,format=None):
    '''Read column data from a text file into numpy arrays and store them
    in a dictionary.
    Inputs:
    filename  -[str] the full path of the data file
    delimeter -[str,default is `space`] the feild seperator used in the data file
    colNames  -[list] the dictionary keys for each column. You do not have to supply
                      a name for every column, however, readcol works from left to 
                      right. If you only supply 3 names for a data file with 5 columns,
                      the names will correspond to the left most 3 columns in order,
                      for example.
    skipLines -[int] readcol attempts to automatically find where data begins in a
                     file (skipping header lines automatically) by checking to see
                     if a line conforms with the input format. You can also tell
                     readcol to skip a certain number of lines manually with
                     this keyword.
    format    -[list of strs] readcol assumes that you are looking for floats but you
                              can manually change the data types for each column. 
                              len(format) must equal len(colNames). For each column,
                              specify whether you want floats, integers, or strings,
                              using an 'f', 'i', or 'a', respectively.
    Returns: 
    dict     - a dictionary whose keys are those provided via the colNames keyword,
               and whose values are numpy arrays of the requested data type 
               (provided via the format keyword).
    '''
    #open the file and read lines into a list
    f=open(filename,'r')
    lines=f.readlines()
    f.close()
    #Split up lines into their columns
    if delimiter != None:
        lines_split=[line.split(delimiter) for line in lines]
    else:
        lines_split=[line.split() for line in lines]
    #find the first and last lines with data (i.e. find a continuous block of lines with numbers
    if format == None:
        format=[]
        for name in colNames:
            format.append('f')
    elif not isinstance(format,(list,np.ndarray)):
        print 'format keyword must be a list or numpy array'
        return
    elif len(colNames) != len(format):
        print 'the number of entries in colNames and in format must be equal'
        return
    function_list=[]
    for data_type in format:
        if data_type=='f' or data_type=='F' or data_type=='':
            function_list.append(float)
        elif data_type=='i' or data_type=='I':
            function_list.append(int)
        elif data_type=='a' or data_type=='A':
            function_list.append(str)
        else:
            print 'Format not understood.'
            print 'Please provide "f" for float,"i" for int, or "a" for string'
            return
    first_line=skipLines
    while first_line <= len(lines_split):
        try:
            for i in xrange(len(colNames)):
                test=function_list[i](lines_split[first_line][i])
            break
        except:
            first_line+=1
    dat_dict={}
    for name in colNames:
        dat_dict[name]=[]
    line=first_line
    while line <= len(lines_split):
        try:
            for i in xrange(len(colNames)):
                dat_dict[colNames[i]].append(function_list[i](lines_split[line][i]))
            line+=1
        except:
            break
    for key in dat_dict.keys():
        if isinstance(dat_dict[key][0],float): dat_dict[key]=np.array(dat_dict[key],dtype=float)
        if isinstance(dat_dict[key][0],int): dat_dict[key]=np.array(dat_dict[key],dtype=int)
        if isinstance(dat_dict[key][0],str): dat_dict[key]=np.array(dat_dict[key],dtype=str)
    return dat_dict
