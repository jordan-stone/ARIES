from ARIES import *
from scipy.interpolate import interp1d

def roll_float(arr,f):
    '''probably should just use scipy's shift...'''
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
    '''Make a model of the spectral fringing using a Fourier filter
    Inputs:
    spec        -[1d array] the flux spectrum
    freq_cutoff -[float] the maximum frequency to make it into the model (in pix^-1)
    check_ft    -[bool, default=False] if True make_model will display the FT of spec
                                       so you can double check if freq_cutoff is appropriate
    Returns:
    1d array    - A model of the spectral fringing shape
    '''
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
    '''find the location of local maxima in the spatial profile of an order.
    Input:
    profile  -[1d array] the spatial profile
    nsources -[int] the number of local maxima to find
    Returns:
    the indices of the nsources maxima
    '''
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
    '''Get the 1-d flat-field fringe by summing all the flat field rows in the 
    vicinity of the maximum of the spatial profile of the science trace and
    fitting using a fourier filter.
    Inputs:
    flat  -[2d Array] the flat-field order
    sci_profile -[1d array] the spatial profile of the science order
    nsources    -[int] the number of sources in the order
    hanning     -[bool] whether to use a hanning window when fourier filtering
    Returns:
    list - the elements of which are 1-d arrays containing the fourier model flat-field fringes...
    '''
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
    '''make a fringe model using a fourier filter, but excise spectral regions within
    input windows and replace those portions of the spectrum with the
    corresponding portion from the flat fringe and noise. This is done 1000 times
    and the average fringe is returned.
    Inputs:
    sci_arr0    -[1d array] the science spectrum
    flat_fringe -[1d array] the flat-field fringe (possibly adjusted to fit the
                             science array)
    window_inds -[1d array,bool] True at indices within windows false without. Windows
                                 are removed and replaced...
    sigma       -[bool] the amplitude of the noise that is added to the flat fringe
                        when it is placed in the spectrum in place of the spectral
                        features...

    modelKw     -keywords fed to make_model
    Returns:
    1d array    -the average model fringe
    '''
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
    '''make a fringe model using a fourier filter, but interpolate over spectral 
    regions within windows. Noise is added to the interpolated region in order
    to downweight this section int he fitting. 1000 realizations of the noise and a 
    corresponding 1000 model fringes are produced. The average fringe is returned.
    Inputs:
    sci_arr0    -[1d array] the science spectrum
    flat_fringe -[1d array] the flat-field fringe (possibly adjusted to fit the
                             science array)
    window_inds -[1d array,bool] True at indices within windows false without. Windows
                                 are removed and replaced...
    sigma       -[bool] the amplitude of the noise that is added to the flat fringe
                        when it is placed in the spectrum in place of the spectral
                        features...

    modelKw     -keywords fed to make_model
    Returns:
    1d array    -the average model fringe
    '''
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
    '''make a fringe model using a fourier filter, but excise spectral regions within
    input windows and replace those portions of the spectrum with the
    corresponding portion from the flat fringe and noise. This is done 1000 times
    and the average fringe and the standard deviation of the fringe in each pixel
    is returned.
    Inputs:
    sci_arr0    -[1d array] the science spectrum
    flat_fringe -[1d array] the flat-field fringe (possibly adjusted to fit the
                             science array)
    window_inds -[1d array,bool] True at indices within windows false without. Windows
                                 are removed and replaced...
    sigma       -[bool] the amplitude of the noise that is added to the flat fringe
                        when it is placed in the spectrum in place of the spectral
                        features...

    modelKw     -keywords fed to make_model
    Returns:
    model  -[1d array]the average model fringe
    std    -[1d array]the std (in each pixel) of the model fringe
    '''
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
    '''Make a set of fringe models using a fourier filter. Each model has been made by first
    excising the portion of the spectrum within the input windows ard replacing it with the
    corresponding portion of the flat-field fringe and a new realization of noise. A 2d array 
    of shape (iterations, len(sci_arr)) is returned.
    Inputs:
    sci_arr0    -[1d array] the science spectrum
    flat_fringe -[1d array] the flat-field fringe (possibly adjusted to fit the
                             science array)
    window_inds -[1d array,bool] True at indices within windows false without. Windows
                                 are removed and replaced...
    sigma       -[bool] the amplitude of the noise that is added to the flat fringe
                        when it is placed in the spectrum in place of the spectral
                        features...
    iterations  -[int] the number of model fringes to produce.

    modelKw     -keywords fed to make_model
    Returns:
    models  -[2d array]the zeroeth dimension indexes all the individual models, the
                       first dimension indexes the wavelength dimension of the fringe.
    '''
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
    '''make a fringe model using a fourier filter, but replace spectral 
    regions within windows with the spectral mean and noise. 1000 realizations
    of the noise and a corresponding 1000 model fringes are produced. 
    The average fringe is returned.
    Inputs:
    sci_arr0    -[1d array] the science spectrum
    flat_fringe -[1d array] the flat-field fringe (possibly adjusted to fit the
                             science array)
    window_inds -[1d array,bool] True at indices within windows false without. Windows
                                 are removed and replaced...
    sigma       -[bool] the amplitude of the noise that is added to the flat fringe
                        when it is placed in the spectrum in place of the spectral
                        features...

    modelKw     -keywords fed to make_model
    Returns:
    1d array    -the average model fringe
    '''
    sci_arr=sci_arr0.copy()
    mn=sci_arr[~window_inds].mean()
    sci_arr[window_inds]=sigma*np.random.randn(len(sci_arr[window_inds]))+mn
    model=make_model(sci_arr,**modelkw)
    for i in xrange(999):
        sci_arr[window_inds]=sigma*np.random.randn(len(sci_arr[window_inds]))+mn
        model+=make_model(sci_arr,**modelkw)
    return model/1000.

class Click_feature_windows:
    '''This class facilitates a graphical interface for selecting feature windows.
    initialize by supplying the science spectrum as a tuple (waves,fluxes), also
    provide the corresponding flat-field fringe and an output name'''
    def __init__(self,sky_spec,flat_fringe,outname='windows.dat',**modelkw):
        '''Inputs:
           sky_spec    - (tuple,len=2)the zeroeth element is the wavelength array
                                      the first element is the flux array 
           flat_fringe - (tuple,len=2)the flat field fringe model in a tuple
                                      similar to sky_spec
           outname     - (str,default='windows.dat') a filename to output the selected
                                                     windows.
        '''
        self.modelkw=modelkw
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
        '''This method creates the attribute self.windows
        which is a boolean array true inside a window
        and false outside a window. The attribute
        self.not_windows is also created, which has
        True values outside of selected windows and 
        False values inside'''
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
    '''This class does everything that Click_feature_windows does,
    but also automatically uses the selected windows to create a
    model fringe using an approach similar to defringe.make_model_noise
    '''
    def button_2_meth(self):
        self.f.canvas.mpl_disconnect(self.cid)
        self.make_windows()
        self.make_model_noise(sigma=0.1,**self.modelkw)
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



                
