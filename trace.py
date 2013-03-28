from ARIES import *
from ARIES.trace import *

def sniff_lamp_traces(fileNameList,method='mean'):
    dataList=np.array(map(lambda x:jFits.get_fits_array(x)[1],fileNameList))
    if method=='mean':
        sumd=dataList.sum(axis=0)
    elif method=='median':
        sumd=np.median(dataList,axis=0)
    else:
        print 'method must be either mean or median'
        return

    kern=np.zeros_like(sumd)
    sh=kern.shape
    kern[sh[0]/2.-2.5:sh[0]/2.+2.5,sh[1]/2.-2.5:sh[1]/2.+2.5]=1.
    d=(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(sumd)*np.fft.fft2(kern)))).real
    col_maxes={}
    for column in xrange(4,1020,1):
        peaks={}
        for p0 in np.arange(4,1020,2):
            peak=climb(int(p0),d[:,column])
            peaks[peak]=p0

        col_maxes[column]=sorted(peaks.keys())

    columns=sorted(col_maxes.keys())

    traces=[[(r,columns[0])] for r in col_maxes[columns[0]]]
    for c in columns[1:]:
        sort_temp=[]
        new=[]
        for y in col_maxes[c]:
            x_arr=np.array([r[-1][0] if np.abs(r[-1][1]-c) < 15 else 9999 for r in traces] )
            dif=np.abs(x_arr-y)
            if np.min(dif) < 4:
                sort_temp.append(((y,c),(dif==np.min(dif)).nonzero()[0][0]))
            else:
                new.append([(y,c)])
        for found in sort_temp:
            traces[found[1]].append(found[0])
        for n in new:
            traces.append(n)

    ax=jFits.jDisplay(d,figsize=(11,11))

    tr=sorted(traces,key=len)
    tr_zip=[]
    for r in tr:
        x=[x[1] for x in r]
        y=[y[0] for y in r]
        tr_zip.append(zip(x,y))


    for ii,r in enumerate(tr_zip[-27:]):
        print ii
        x,y=unzip(r)
        ax.plot(x,y,'r-')
        ax.text(x[10],y[10],'%i' % ii,color='r')
        ax.set_title('longest 27 traces')
        ax.set_xlim(0,1023)
        ax.set_ylim(0,1023)
    mpl.show()
    return tr_zip

def tmp(arr,method='mean'):
    sh=arr.shape
    col_maxes={}
    for column in xrange(2,sh[1]-2,1):
        peaks={}
        for p0 in np.arange(2,sh[0],1):
            peak=climb(int(p0),arr[:,column])
            peaks[peak]=p0

        col_maxes[column]=sorted(peaks.keys())

    columns=sorted(col_maxes.keys())

    traces=[[(r,columns[0])] for r in col_maxes[columns[0]]]
    for c in columns[1:]:
        sort_temp=[]
        new=[]
        for y in col_maxes[c]:
            x_arr=np.array([r[-1][0] if np.abs(r[-1][1]-c) < 15 else 9999 for r in traces] )
            dif=np.abs(x_arr-y)
            if np.min(dif) < 4:
                sort_temp.append(((y,c),(dif==np.min(dif)).nonzero()[0][0]))
            else:
                new.append([(y,c)])
        for found in sort_temp:
            traces[found[1]].append(found[0])
        for n in new:
            traces.append(n)

    aratio=.5*sh[0]/float(sh[1])
    ax=jFits.jDisplay(arr,figsize=(11,11),aspect=aratio)
    #ax.set_xlim(0,sh[1])
    #ax.set_ylim(0,sh[0])

    tr=sorted(traces,key=len)
    tr_zip=[]
    for r in tr:
        x=[x[1] for x in r]
        y=[y[0] for y in r]
        tr_zip.append(zip(x,y))


    for ii,r in enumerate(tr_zip[-27:]):
        print ii
        x,y=unzip(r)
        ax.plot(x,y,'r-')
        ax.text(x[5],y[5],'%i' % ii,color='r')
        ax.set_title('longest 27 traces')
        ax.set_xlim(0,1023)
        ax.set_ylim(0,1023)
    mpl.show()
    return tr_zip

def sniff_gas_traces(arr,method='mean'):
    col_maxes={}
    sh=arr.shape
    for column in xrange(1,sh[1]-2,1):
        peaks={}
        for p0 in np.arange(1,sh[0]-2,1):
            peak=climb(int(p0),arr[:,column])
            peaks[peak]=p0

        col_maxes[column]=sorted(peaks.keys())

    columns=sorted(col_maxes.keys())

    traces=[[(r,columns[0])] for r in col_maxes[columns[0]]]
    for c in columns[1:]:
        sort_temp=[]
        new=[]
        for y in col_maxes[c]:
            x_arr=np.array([r[-1][0] if np.abs(r[-1][1]-c) < 15 else 9999 for r in traces] )
            dif=np.abs(x_arr-y)
            if np.min(dif) < 4:
                sort_temp.append(((y,c),(dif==np.min(dif)).nonzero()[0][0]))
            else:
                new.append([(y,c)])
        for found in sort_temp:
            traces[found[1]].append(found[0])
        for n in new:
            traces.append(n)


    aratio=1
    aratio=.5*sh[1]/float(sh[0])
    ax=jFits.jDisplay(arr,figsize=(11,11),log=True,aspect=aratio)

    tr=sorted(traces,key=len)
    tr_zip=[]
    for r in tr:
        x=[x[1] for x in r]
        y=[y[0] for y in r]
        tr_zip.append(zip(x,y))


    for ii,r in enumerate(tr_zip[-70:]):
        print ii
        x,y=unzip(r)
        ax.plot(x,y,'r-')
        ax.text(x[10],y[10],'%i' % ii,color='r')
        ax.set_title('longest 27 traces')
        #ax.set_xlim(0,1023)
        #ax.set_ylim(0,1023)
    mpl.show()
    return tr_zip


def poly_extrapolate_lamp_traces(traces,order=2):
    outTraces=[]
    for t in traces:
        x=[x[0] for x in t]
        y=[y[1] for y in t]
        p=np.polyfit(x,y,order)
        fit_y=np.polyval(p,np.arange(1024))
        out_x=np.arange(1024)[np.logical_and(fit_y < 1024,fit_y>0)] 
        fit_y=fit_y[np.logical_and(fit_y < 1024,fit_y>0)]
        outTraces.append(zip(out_x,fit_y))
    return outTraces
    

def write_traces(fname,traces):
    num_traces=len(traces)
    len_traces=np.array([len(tr) for tr in traces])
    line_starts=np.concatenate(([num_traces+2],(len_traces[:-1]+2))).cumsum()
    fo=open(fname,'w')
    for order, start in enumerate(line_starts):
        fo.write('%i %i\n' % (order, start))
    fo.write('\n')
    for order, tr in enumerate(traces):
        fo.write('order %i \n' % order)
        x=[x[0] for x in tr]
        y=[y[1] for y in tr]
        for point in zip(x,y):
            fo.write('%f %f\n' % (point[0],point[1]))
        fo.write('\n')
    fo.close()

def get_trace_offset(im,traces):
    tot={}
    for o in np.arange(10)-5:
        tot[o]=sum([sum(x[~np.isnan(x)]) for x in get_trace_vals(im,traces,offset=o)])
    sorted_os=sorted(tot.keys(), key=tot.get)
    return sorted_os[-1]

def trace_optimize(im,traces,centroid_width=10,offset=None,climb_not_centroid=False):
    if offset is None:
        offset=get_trace_offset(im,traces)
    out_traces=[]
    opt_func=lambda x:get_gauss(x)[0]
    if climb_not_centroid:
        opt_func=lambda x:climb(len(x)/2.,x)
    for i,tr in enumerate(traces):
        x=np.array([x[0] for x in tr])
        #y=[min(max(y[1]+offset,0),1023) for y in tr]
        y=np.array([y[1]+offset for y in tr])
        inds=np.logical_and(y>15,y<1008)
        x=x[inds]
        y=y[inds]
        fit_y=[]
        for point in zip(x,y):
            snippet=im[max(point[1]-centroid_width,0):min(point[1]+centroid_width,1023),point[0]]
            snippet=snippet.copy()#don't want to change the actual image...
            mu=opt_func(snippet)
            fit_y.append(mu+(point[1]-centroid_width))
        out_traces.append(zip(x,fit_y))
    return out_traces

def read_traces(fname,offset=0):
#fname is the name of the text file of stacked x,y values for each order.
    order_info=readcol(fname,colNames=['order','skip'],format=['i','i'],skipLines=0)

    orders=[]
    if offset <> 0:
        for position in order_info['skip']:
            d=readcol(fname,colNames=['x','y'],skipLines=position)
            d['y']=d['y']+offset
            inds=(np.logical_and(d['y'] > 0,d['y'] < 1023)).nonzero()#buffer one row...
            d['x']=d['x'][inds]
            d['y']=d['y'][inds]
            orders.append(zip(d['x'],d['y']))
    else:
        for position in order_info['skip']:
            d=readcol(fname,colNames=['x','y'],skipLines=position)
            orders.append(zip(d['x'],d['y']))
    return orders

def get_trace_vals(im,traces,offset=0):
    spects=[]
    for t in traces:
        s=[im[max(min(xy[1]+offset,1023),0),max(min(xy[0],1023),0)] for xy in t]
        spects.append(np.array(s))
    return spects

def double_trace(single_traces,separation):
    second_traces=[]
    for trace in single_traces:
        x,y=unzip(trace)
        second_traces.append(zip(x,[yy+separation for yy in y]))
    double_traces=single_traces+second_traces
    return sorted(double_traces,key=lambda x:x[0][1])

def transpose_trace(t):
    x,y=unzip(t)
    return zip(y,x)


def unroll_trace(trace,onto_xy='x',degree=2):
    x,y=unzip(trace)
    fit_ys=np.polyval(np.polyfit(x,y,degree),x)

    if onto_xy=='x':
        out_abs=np.append([0],np.sqrt(np.diff(x)**2.+np.diff(fit_ys)**2.).cumsum())+x[0]
        out_ord=np.repeat(fit_ys[0],len(x))
    if onto_xy=='y':
        out_abs=np.repeat(x[0],len(fit_ys))
        out_ord=np.append([0],np.sqrt(np.diff(x)**2.+np.diff(fit_ys)**2.).cumsum())+y[0]
    return out_abs, out_ord
