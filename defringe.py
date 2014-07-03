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
    #print len(spec), freq_cutoff
    ft=np.fft.fft(spec)
    freqs=np.fft.fftfreq(len(spec))
    if check_ft:
        f=mpl.figure()
        a=f.add_subplot(111)
        a.plot(freqs,ft)
        a.plot((freq_cutoff,)*2,a.get_ylim(),'k--')
        a.plot((-freq_cutoff,)*2,a.get_ylim(),'k--')
        return 
    else:
        modelfcs=np.where(np.abs(freqs)<freq_cutoff,ft,0)
        model=np.fft.ifft(modelfcs)
        return model.real

def find_maxes(profile,nsources):
    diffs=[]
    diffs.append((profile[0]-profile[1]>0))
    for i in xrange(1,len(profile)-1):
        diffs.append( ((profile[i]-profile[i-1])>0) and ((profile[i]-profile[i+1])>0))
    diffs.append((profile[-1]-profile[-2])>0)
    inds=np.array(diffs).nonzero()[0]
    if len(inds)==nsources:
        return inds
    else:
        iinds=np.argsort(profile[inds])
        return sorted(inds[iinds][-nsources:])


def get_flat_fringe(flat,sci_profile,nsources=1,hanning=True):
    centers=find_maxes(sci_profile,nsources)
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

def make_model_noise(sci_arr0,flat_fringe=None,window_inds=None,sigma=None,**modelkw):
    sci_arr=sci_arr0.copy()
    if flat_fringe is None:
        flat_fringe=np.zeros_like(sci_arr)
    mn=sci_arr[~window_inds].mean()
    sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
    model=make_model(sci_arr,**modelkw)
    for i in xrange(999):
        sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
        model+=make_model(sci_arr,**modelkw)
    return model/1000.

def make_model_noise_interpolate(sci_arr0,window_inds=None,sigma=None,**modelkw):
    func=interp1d(np.arange(len(sci_arr0))[~window_inds],sci_arr0[~window_inds],
                            bounds_error=False,
                            fill_value=sci_arr0[~window_inds].mean())
    sci_arr00=func(np.arange(len(sci_arr0)))
    sci_arr=sci_arr00+sigma*np.random.randn(len(sci_arr00))*window_inds
    model=make_model(sci_arr,**modelkw)
    for i in xrange(999):
        sci_arr=sci_arr00+sigma*np.random.randn(len(sci_arr00))*window_inds
        model+=make_model(sci_arr,**modelkw)
    return model/1000.

def make_model_noise_std(sci_arr0,flat_fringe=None,window_inds=None,sigma=None,**modelkw):
    sci_arr=sci_arr0.copy()
    if flat_fringe is None:
        flat_fringe=np.zeros_like(sci_arr)
    mn=sci_arr[~window_inds].mean()
    sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
    models=[]
    models.append(make_model(sci_arr,**modelkw))
    for i in xrange(999):
        sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
        models.append(make_model(sci_arr,**modelkw))
    models=np.array(models)
    return models.mean(axis=0),models.std(axis=0)

def make_model_noise_iterations(sci_arr0,flat_fringe=None,window_inds=None,sigma=None,iterations=1000,**modelkw):
    sci_arr=sci_arr0.copy()
    if flat_fringe is None:
        flat_fringe=np.zeros_like(sci_arr)
    mn=sci_arr[~window_inds].mean()
    sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
    models=[]
    models.append(make_model(sci_arr,**modelkw))
    for i in xrange(999):
        sci_arr[window_inds]=flat_fringe[window_inds]+sigma*np.random.randn(len(sci_arr[window_inds]))+mn
        models.append(make_model(sci_arr,**modelkw))
    models=np.array(models)
    return models

def make_model_noise_no_seed(sci_arr0,window_inds=None,sigma=None,**modelkw):
    sci_arr=sci_arr0.copy()
    mn=sci_arr[~window_inds].mean()
    sci_arr[window_inds]=sigma*np.random.randn(len(sci_arr[window_inds]))+mn
    model=make_model(sci_arr,**modelkw)
    for i in xrange(999):
        sci_arr[window_inds]=sigma*np.random.randn(len(sci_arr[window_inds]))+mn
        model+=make_model(sci_arr,**modelkw)
    return model/1000.

class Click_feature_windows:
    def __init__(self,sky_spec,flat_fringe,outname='windows.dat'):
        self.outname=outname
        self.sky_spec=sky_spec
        self.flat_fringe=flat_fringe
        self.f=mpl.figure(figsize=(16,8))
        self.a=self.f.add_subplot(111)
        self.a.plot(*sky_spec,color='b')
        self.a.plot(flat_fringe[0],flat_fringe[1]+np.mean(sky_spec[1]),color='g')
        self.a.set_ylim(sky_spec[1].mean()-2.5*sky_spec[1].std(),
                        sky_spec[1].mean()+2.5*sky_spec[1].std())
        if max([abs(lim) for lim in self.a.get_ylim()]) > 2.:
            self.a.set_ylim(-0.1,2)
        self.cid=self.f.canvas.mpl_connect('button_press_event',self)
        self.starts=[]
        self.stops=[]
        self.clicks=[]
        self.ylims=self.a.get_ylim()
        print 'click boundaries of defringe windows'

    def __call__(self,event):
        print 'button %i' % event.button
        if event.button == 1:
            self.clicks.append(event.xdata)
            l=len(self.clicks)
            if l % 2 == 0:
                self.stops.append(self.clicks[-1])
                self.a.plot([self.clicks[-1]]*2,self.ylims,'k--')
                self.a.set_ylim(*self.ylims)
                mpl.show()
            else:
                self.starts.append(self.clicks[-1])
                self.a.plot([self.clicks[-1]]*2,self.ylims,'k-')
                self.a.set_ylim(*self.ylims)
                mpl.show()
        elif event.button == 3:
            l=len(self.clicks)
            if l%2 == 0:
                self.starts.pop(-1)
            else:
                self.stops.pop(-1)
        elif event.button == 2:
            self.button_2_meth()

    def button_2_meth(self):
        self.f.canvas.mpl_disconnect(self.cid)
        self.make_windows()
        self.save_windows(outname=self.outname)
        
    def make_windows(self):
        winds=map(lambda low,high:np.logical_and(self.sky_spec[0]>low,
                                                 self.sky_spec[0]<high),
                                                 self.starts,
                                                 self.stops)
        self.windows=np.sum(winds,axis=0)>0
        self.not_windows=~self.windows

    def save_windows(self,outname='windows.dat'):
        fo=open(outname,'w')
        for ind in self.windows:
            fo.write('%i\n'%ind)
        fo.close()



class Click_and_defringe(Click_feature_windows):
    def button_2_meth(self):
        self.f.canvas.mpl_disconnect(self.cid)
        self.make_windows()
        self.make_model_noise(sigma=0.1)
        self.out=self.sky_spec[1]-self.model
        self.fout=mpl.figure(figsize=(16,8))
        self.aout=self.fout.add_subplot(111)
        self.aout.plot(*self.sky_spec,linestyle=':',label='observed')
        self.aout.plot(self.sky_spec[0],self.model,linestyle='--',label='fringe model')
        self.aout.plot(self.flat_fringe[0],self.flat_fringe[1]+np.mean(self.sky_spec[1]),'k--',label='flat fringe')
        self.aout.plot(self.sky_spec[0],self.out+np.mean(self.sky_spec[1]),label='defringed')
        self.aout.legend(loc='best')

    def make_model_noise(self,sigma=None,**modelkw):
        if not modelkw.has_key('freq_cutoff'):
            modelkw['freq_cutoff']=15./float(len(self.sky_spec[1]))
        sci_arr=self.sky_spec[1].copy()
        mn=sci_arr[self.not_windows].mean()
        sci_arr[self.windows]=self.flat_fringe[1][self.windows]+\
                           sigma*np.random.randn(len(sci_arr[self.windows]))+\
                           mn
        model=make_model(sci_arr,**modelkw)
        for i in xrange(999):
            sci_arr[self.windows]=self.flat_fringe[1][self.windows]+\
                                 sigma*np.random.randn(len(sci_arr[self.windows]))+\
                                 mn
            model+=make_model(sci_arr,**modelkw)
        self.model=model/1000.



                
