from ARIES import *
from scipy.interpolate import interp1d

def roll_float(arr,f):
    arr_len=len(arr)
    int_part=np.trunc(f)
    frac_part=f-int_part
    int_roll=np.roll(arr,int(int_part))
    fill=np.mean([int_roll[0],int_roll[-1]])

    x0=np.arange(arr_len) 
    x=x0+frac_part

    func=interp1d(x,int_roll,bounds_error=False,fill_value=fill)
    return func(np.arange(arr_len))

def make_model(spec,freq_cutoff=0.015,check_ft=False):
    ft=np.fft.fft(spec)
    freqs=np.fft.fftfreq(len(spec))
    if check_ft:
        f=mpl.figure()
        a=f.add_subplot(111)
        a.plot(freqs,ft)
        a.plot((freq_cutoff,)*2,a.get_ylims(),'k--')
        a.plot((-freq_cutoff,)*2,a.get_ylims(),'k--')
        return 
    else:
        modelfcs=np.where(np.abs(freqs)<freq_cutoff,ft,0)
        model=np.fft.ifft(modelfcs)
        return model.real

def make_order_model(arr,yorder_lims,xorder_lims=(None,None)):
    order=arr[slice(yorder_lims[0],yorder_lims[1]),
              slice(xorder_lims[0],xorder_lims[1])]
    canvas=np.empty_like(order)
    for i,row in ((i,order[i,:]) for i in xrange(order.shape[0])):
        pf=np.polyfit(np.arange(len(row))[:-10],row[:-10],3)
        psub=row-np.polyval(pf,np.arange(order.shape[1]))
        canvas[i,:]=make_model(psub)
    return canvas

def get_flat_fringe(flat_order,sci_profile,nsources=1):
    flat=flat_order*sci_profile[:,None]
    params=sum(map(lambda n:
                            [flat.sum(axis=0).max(),flat.shape[0]*(2*n+1)/float(2*nsources),1],
                            range(nsources)),
                            [])
    pc,mc,ec,xc=gaussfitter.multigaussfit(np.arange(flat.shape[0]),
                                          flat.sum(axis=1),
                                          ngauss=nsources,
                                          params=params)
    sort_inds=np.argsort(pc[1::3])
    amp    =pc[0::3][sort_inds]
    spatial=pc[1::3][sort_inds]
    sigma  =pc[2::3][sort_inds]
    if nsources==1:
        yslice=slice( np.max((0,spatial[0]-2*sigma[0])), np.min((flat.shape[0],spatial[0]+2*sigma[0])) )
        specs=[flat[yslice,:].sum(axis=0)]
    else:
        models=[]
        slices=[]
        for i in xrange(nsources):
            models.append(gaussfitter.onedgaussian(np.arange(flat.shape[0]),
                                                   0,
                                                   amp[i],
                                                   spatial[i],
                                                   sigma[i]))
            slices.append(slice(np.max((0,spatial[i]-2*sigma[i])), np.min((flat.shape[0],spatial[i]+2*sigma[i]))))
        models=np.array(models)
        models_subtracted=flat-models[::-1,:,None]
        specs=[models_subtracted[0,slices[0],:].sum(axis=0),models_subtracted[1,slices[1],:].sum(axis=0)]
    fringes=[]
    for spec in specs:
        pf=np.polyfit(np.arange(len(spec)),spec,3)
        pm=np.polyval(pf,np.arange(len(spec)))
        fringes.append(make_model(spec)-pm+pm.mean())
    return fringes

def get_flat_fringe2(flat,sci_profile,nsources=1,centers=(1,10),hanning=True):
    if nsources==1:
        yslice=slice( np.max((0,centers[0]-4)), np.min((flat.shape[0],centers[0]+4)) )
        specs=[flat[yslice,:].sum(axis=0)]
    else:
        slices=[]
        for i in xrange(nsources):
            slices.append(slice(np.max((0,centers[i]-4)), np.min((flat.shape[0],centers[i]+4))))
        specs=[flat[slices[0],:].sum(axis=0),flat[slices[1],:].sum(axis=0)]
    fringes=[]
    for spec in specs:
        pf=np.polyfit(np.arange(len(spec)),spec,3)
        pm=np.polyval(pf,np.arange(len(spec)))
        if hanning:
            fringes.append(make_model(np.hanning(len(spec))*(spec-pm)))
        else:
            fringes.append(make_model(spec-pm))
    return fringes

def defringe_science(sci_arr,flat_fringe=None,windows=[None],sigmas=[None]):
    '''fit for the fringe in a science spectrum but first replace regions of the spectrum
    with spectral features with noise, or optionally with noise ontop of the fringe extracted from the flat.
    arguments:
    sci_arr: the telluric corrected science spec
    flat_fringe: the fringe extracted from the corresponding flat order
    windows: a list of slice objects, specifiying the windows to replace with noise
    sigmas: a list of scalars specifying the amplitude of the noise to replace with.'''
    if flat_fringe is None:
        flat_fringe=np.zeros_like(sci_arr)
    if len(sigmas)<>len(windows):
        sigmas=list(sigmas)*len(windows)/len(sigmas)
    for window_sigma in zip(windows,sigmas):
        sci_arr[window_sigma[0]]=flat_fringe[window_sigma[0]]+window_sigma[1]*np.random.randn(len(sci_arr[window_sigma[0]]))
    




#hw,dw=jFits.get_fits_array('../flat_dewarped_yoff_147.fits')
#canvas=np.zeros_like(dw)
#mask=np.ones_like(dw)
#mask2=np.zeros_like(dw)
#lim_dir='../mask_regions/order_lims_order_'
#colors=('purple','g','b','c','m','y','k','orange','r','lime','salmon','pink','periwinkle')
#
#for i in range(18):
#    order_lims=readcol(lim_dir+str(i)+'.txt',colNames=['low','high'])
#    order_lims['low'][0]-=2#kluge
#    order_lims['high'][0]+=2#kluge
#    order=dw[order_lims['low'][0]:order_lims['high'][0],:]
#    canvas_order=canvas[order_lims['low'][0]:order_lims['high'][0],:]
#    f=mpl.figure(figsize=(16,8))
#    a=f.add_subplot(311)
#    b=f.add_subplot(312)
#    c=f.add_subplot(313)
#    for i,row in ((i,order[i,:]) for i in xrange(order.shape[0])):
#        color=colors[i]
#        pf=np.polyfit(np.arange(len(row))[:-10],row[:-10],3)
#        psub=row-np.polyval(pf,np.arange(1037))
#
#        a.plot(row+10000*i,color=color)
#        a.plot(np.polyval(pf,np.arange(1037))+10000*i,color=color)
#        a.set_ylim(0,max((row+10000*i)[400:600]))
#
#        b.plot(psub+8000*i,color=color)
#        b.set_ylim(-8000,max((psub+8000*i)[400:600]))
#
#        hft=np.fft.fft( np.hanning(1017)*psub[10:-10] )
#        ft=np.fft.fft( psub[10:-10] )
#        c.plot(np.fft.fftfreq(1017),np.abs(hft)+600000*i,color=color)
#        c.set_xlim((-.2,.2))
#        c.plot([0.009643,0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.009643,-0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.015,-0.015],c.get_ylim(),'k',linestyle=':')
#        c.plot([0.015,0.015],c.get_ylim(),'k',linestyle=':')
#
#        fringe_ft=np.where(np.logical_and(np.fft.fftfreq(1017) > -0.015,
#                                          np.fft.fftfreq(1017) <  0.015),
#                           ft,0)
#        fringe_model=np.fft.ifft(fringe_ft)
#        canvas_order[i,10:-10]=fringe_model
#        b.plot(np.arange(1037)[10:-10],fringe_model+8000*i,color='r')
#
#for i in (18,19):
#    order_lims=readcol(lim_dir+str(i)+'.txt',colNames=['low','high'])
#    order_lims['low'][0]-=2#kluge
#    order_lims['high'][0]+=2#kluge
#    order=dw[order_lims['low'][0]:order_lims['high'][0],:]
#    canvas_order=canvas[order_lims['low'][0]:order_lims['high'][0],:]
#    f=mpl.figure(figsize=(16,8))
#    a=f.add_subplot(311)
#    b=f.add_subplot(312)
#    c=f.add_subplot(313)
#    for i,row in ((i,order[i,:]) for i in xrange(order.shape[0])):
#        color=colors[i]
#        pf=np.polyfit(np.arange(len(row))[:960],row[:960],3)
#        psub=row-np.polyval(pf,np.arange(1037))
#
#        a.plot(row+10000*i,color=color)
#        a.plot(np.polyval(pf,np.arange(1037))+10000*i,color=color)
#        a.set_ylim(0,max((row+10000*i)[400:600]))
#
#        b.plot(psub+8000*i,color=color)
#        b.set_ylim(-8000,max((psub+8000*i)[400:600]))
#
#        hft=np.fft.fft( np.hanning(950)*psub[10:960] )
#        ft=np.fft.fft( psub[10:960] )
#        c.plot(np.fft.fftfreq(950),np.abs(hft)+600000*i,color=color)
#        c.set_xlim((-.2,.2))
#        c.plot([0.009643,0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.009643,-0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.015,-0.015],c.get_ylim(),'k',linestyle=':')
#        c.plot([0.015,0.015],c.get_ylim(),'k',linestyle=':')
#
#        fringe_ft=np.where(np.logical_and(np.fft.fftfreq(950) > -0.015,
#                                          np.fft.fftfreq(950) <  0.015),
#                           ft,0)
#        fringe_model=np.fft.ifft(fringe_ft)
#        canvas_order[i,10:960]=fringe_model
#        b.plot(np.arange(1037)[10:960],fringe_model+8000*i,color='r')
#
#for i in (20,):
#    order_lims=readcol(lim_dir+str(i)+'.txt',colNames=['low','high'])
#    order_lims['low'][0]-=2#kluge
#    order_lims['high'][0]+=2#kluge
#    order=dw[order_lims['low'][0]:order_lims['high'][0],:]
#    canvas_order=canvas[order_lims['low'][0]:order_lims['high'][0],:]
#    f=mpl.figure(figsize=(16,8))
#    a=f.add_subplot(311)
#    b=f.add_subplot(312)
#    c=f.add_subplot(313)
#    for i,row in ((i,order[i,:]) for i in xrange(order.shape[0])):
#        color=colors[i]
#        pf=np.polyfit(np.arange(len(row))[:850],row[:850],3)
#        psub=row-np.polyval(pf,np.arange(1037))
#
#        a.plot(row+10000*i,color=color)
#        a.plot(np.polyval(pf,np.arange(1037))+10000*i,color=color)
#        a.set_ylim(0,max((row+10000*i)[400:600]))
#
#        b.plot(psub+8000*i,color=color)
#        b.set_ylim(-8000,max((psub+8000*i)[400:600]))
#
#        hft=np.fft.fft( np.hanning(840)*psub[10:850] )
#        ft=np.fft.fft( psub[10:850] )
#        c.plot(np.fft.fftfreq(840),np.abs(hft)+600000*i,color=color)
#        c.set_xlim((-.2,.2))
#        c.plot([0.009643,0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.009643,-0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.015,-0.015],c.get_ylim(),'k',linestyle=':')
#        c.plot([0.015,0.015],c.get_ylim(),'k',linestyle=':')
#
#        fringe_ft=np.where(np.logical_and(np.fft.fftfreq(840) > -0.015,
#                                          np.fft.fftfreq(840) <  0.015),
#                           ft,0)
#        fringe_model=np.fft.ifft(fringe_ft)
#        canvas_order[i,10:850]=fringe_model
#        b.plot(np.arange(1037)[10:850],fringe_model+8000*i,color='r')
#
#for i in (21,):
#    order_lims=readcol(lim_dir+str(i)+'.txt',colNames=['low','high'])
#    order_lims['low'][0]-=2#kluge
#    order_lims['high'][0]+=2#kluge
#    order=dw[order_lims['low'][0]:order_lims['high'][0],:]
#    canvas_order=canvas[order_lims['low'][0]:order_lims['high'][0],:]
#    f=mpl.figure(figsize=(16,8))
#    a=f.add_subplot(311)
#    b=f.add_subplot(312)
#    c=f.add_subplot(313)
#    for i,row in ((i,order[i,:]) for i in xrange(order.shape[0])):
#        color=colors[i]
#        pf=np.polyfit(np.arange(len(row))[:520],row[:520],3)
#        psub=row-np.polyval(pf,np.arange(1037))
#
#        a.plot(row+10000*i,color=color)
#        a.plot(np.polyval(pf,np.arange(1037))+10000*i,color=color)
#        a.set_ylim(0,max((row+10000*i)[400:600]))
#
#        b.plot(psub+8000*i,color=color)
#        b.set_ylim(-8000,max((psub+8000*i)[400:600]))
#
#        hft=np.fft.fft( np.hanning(510)*psub[10:520] )
#        ft=np.fft.fft( psub[10:520] )
#        c.plot(np.fft.fftfreq(510),np.abs(hft)+600000*i,color=color)
#        c.set_xlim((-.2,.2))
#        c.plot([0.009643,0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.009643,-0.009643],c.get_ylim(),'r',linestyle=':')
#        c.plot([-0.015,-0.015],c.get_ylim(),'k',linestyle=':')
#        c.plot([0.015,0.015],c.get_ylim(),'k',linestyle=':')
#
#        fringe_ft=np.where(np.logical_and(np.fft.fftfreq(510) > -0.015,
#                                          np.fft.fftfreq(510) <  0.015),
#                           ft,0)
#        fringe_model=np.fft.ifft(fringe_ft)
#        canvas_order[i,10:520]=fringe_model
#        b.plot(np.arange(1037)[10:520],fringe_model+8000*i,color='r')
