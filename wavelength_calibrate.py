from ARIES import *
lines=readcol('transdata_1_5_mic',colNames=['wn','A'])
lines['w']=1.e4/lines['wn']
minstep=-(np.diff(lines['w']).max())

def get_lines(wave_lims):
    '''from the high-resolution transmission spectrum, grab the spectral
    region of interest and interpolate it onto a regular grid, with the
    minimum gridspacing present in the spectrum
    Input:
    wave_lims -[len=2 tuple] the low and high wavelength limits of interest
                             in microns
    Returns:
    len=2 tuple the zeroeth element is the wavelength array, the first
    element is the transmission spectrum array'''
    inds=np.logical_and(lines['w']>wave_lims[0],lines['w']<wave_lims[1])
    #make sure wavelengths are increasing
    if lines['w'][-1] < lines['w'][0]:
        l=lines['w'][inds][::-1]
        A=lines['A'][inds][::-1]
    else:
        l=lines['w'][inds]
        A=lines['A'][inds]
    wave_lengths=np.arange(l.min(),(l.max()+minstep),minstep)
    delta_spec=np.interp(wave_lengths,l,A)
    return (wave_lengths,delta_spec)

def make_gauss_kern(array_length,R=30000.,central_lambda=2.,minstep=minstep):
    '''make a gaussian shaped kernel for smoothing from a resolution of infinity
    to a resolution of R.
    Inputs:
    array_length -[int] the length of the array of the spectrum to be smoothed
    R            -[float] the resolution to smooth to assuming an input resolution
                          of infinity.
    central_lambda-[float] the central wavelength (in microns) of the array to be
                           smoothed
    minstep       -[float] the minimum wavelength stepsize of the array to be smoothed
    Returns:
    1-d array of len=array_length, a normalized gaussian kernel with width appropriate
    to smooth a resolution of infinity to R
    '''
    delta_lambda=central_lambda/R
    sig=0.5*delta_lambda/minstep
    x=np.arange(array_length)
    g=np.exp( - ( (x-x.mean())**2. / (2*sig**2) ) )
    return g/g.sum()

def smooth(lines,kernel):
    '''Convolve a spectrum with a kernel, no padding is performed.
    spectrum and kernel should have same length.
    Inputs:
    lines -[1d array] the input spectrum
    kernel-[1d array] the smoothing kernel the same length as lines
    Returns:
    1d array (complex), ifft(fft(lines)*fft(kernel))
    '''
    return np.fft.fftshift( np.fft.ifft( np.fft.fft(lines) * np.fft.fft(kernel) ) )

def get_smoothed_lines(wave_lims,**kern_args):
    '''Return the telluric transmission spectrum between wave_lims,
    and smooth with a gaussian kernel.
    Inputs:
    wave_lims -[len=2 tuple]the low and high wavelength limits of interest
                            in microns.
    **kern_args - keywords passed to make_gauss_kern
    Returns:
    len=2 tuple, the zeroeth element is the wavelength array. The first
    element is the smoothed telluric absorption spectrum.
    '''
    inds=np.logical_and(lines['w']>wave_lims[0],lines['w']<wave_lims[1])
    #make sure wavelengths are increasing
    l=lines['w'][inds][::-1]
    A=lines['A'][inds][::-1]
    wave_lengths=np.arange(l.min(),(l.max()+minstep),minstep)
    delta_spec=np.interp(wave_lengths,l,A)
    g=make_gauss_kern(len(wave_lengths),**kern_args)
    out=smooth(delta_spec,g)
    return (wave_lengths,out.real)

    
class pick_corresponding_lines:
    '''This class implements a graphical interface for associating lines in an
    observed spectrum and a template spectrum. You will click the lines, starting
    by clicking a line in the observed spectrum, then the corresponding line in the
    template spectrum.'''
    def __init__(self,sky_spec,obs_spec):
        self.sky_spec=sky_spec
        self.obs_spec=obs_spec
        self.f=mpl.figure(figsize=(16,8))
        self.a=self.f.add_subplot(111)
        self.a.plot(*sky_spec,color='b')
        self.a.plot(*obs_spec,color='g')
        if max([abs(lim) for lim in self.a.get_ylim()]) > 10.:
            self.a.set_ylim(-0.1,10)
        self.cid=self.f.canvas.mpl_connect('button_press_event',self)
        self.corr_points=[]
        self.obs_points=[]
        self.sky_points=[]
        self.obs_peak_inds=[]
        self.sky_peak_inds=[]
        self.clicked_obs_points=[]
        self.clicked_sky_points=[]
        print 'left click corresponding lines. Do obs,sky,obs,sky,obs,sky...'
        print "right click to correct a mistake...Don't do this twice in a row..." 
        print "middle click to accept the selected points and perform the fit..."
    def __call__(self,event):
        print 'button %i' % event.button
        if event.button == 1:
            inx=event.xdata
            self.corr_points.append(inx)
            l=len(self.corr_points)
            if l % 2 == 0:
                spec=self.sky_spec
                self.clicked_sky_points.append(self.corr_points[-1])
                color='b'
                out_str='sky'
            else:
                spec=self.obs_spec
                self.clicked_obs_points.append(self.corr_points[-1])
                color='g'
                out_str='obs'

            print 'added '+out_str + ' point'

            i=np.where(abs(spec[0]-self.corr_points[-1])==abs(spec[0]-self.corr_points[-1]).min())[0]
            peak=climb(i,-spec[1])

            self.a.plot(spec[0][peak],spec[1][peak],color+'o')
            mpl.show()
        elif event.button == 3:
            l=len(self.corr_points)
            if l%2 == 0:
                self.clicked_sky_points.pop(-1)
                out_str='sky'
            else:
                self.clicked_obs_points.pop(-1)
                out_str='obs'
            print 'subtracted last '+out_str+' point'
            self.corr_points.pop(-1)
            self.a.lines.pop(-1)
            mpl.show()
        elif event.button == 2:
            print 'finding solution...'
            self.f.canvas.mpl_disconnect(self.cid)
            self.corr_points=np.array(self.corr_points)
            self.centroid_lines()
            self.get_transform_coefficients()
            self.a.plot(self.x,self.y,'r-')
            mpl.show()

    def disconnect_clicker(self):
        self.f.canvas.mpl_disconnect(self.cid)
        self.corr_points=np.array(self.corr_points)

    def centroid_lines(self):
        self.obs_points=[]
        self.sky_points=[]

        for p in self.clicked_obs_points:
            i=np.where(abs(self.obs_spec[0]-p)==abs(self.obs_spec[0]-p).min())[0]
            peak=climb(i,-self.obs_spec[1])
            self.obs_peak_inds.append(peak)
            x= self.obs_spec[0][peak-4:peak+4]
            y=-self.obs_spec[1][peak-4:peak+4]
            y-=y.min()
            mu=(x*y).sum()/y.sum()
            self.obs_points.append(mu)

        for p in self.clicked_sky_points:
            i=np.where(abs(self.sky_spec[0]-p)==abs(self.sky_spec[0]-p).min())[0]
            peak=climb(i,-self.sky_spec[1])
            self.sky_peak_inds.append(peak)
            x= self.sky_spec[0][peak-121:peak+121]
            y=-self.sky_spec[1][peak-121:peak+121]
            y-=y.min()
            mu=(x*y).sum()/y.sum()
            self.sky_points.append(mu)


    def get_transform_coefficients(self,degree=2):
        self.transform_coefficients=np.polyfit(self.obs_points,self.sky_points,degree)
        self.transformer=np.poly1d(self.transform_coefficients)
        self.a.plot(np.polyval(self.transform_coefficients,self.obs_spec[0]),(self.obs_spec[1]/self.obs_spec[1].mean()),'r-')
        self.x=np.polyval(self.transform_coefficients,self.obs_spec[0])
        self.y=self.obs_spec[1]/self.obs_spec[1].mean()
    
def do_calibration(file_name,dlamb=0.035,low_lamb=2.15,write=True,offset=0.4,h=[],iter=True,trim=False,**kernkw):
    '''implement wavelength_calibrate.pick_corresponding_lines and return the transformed x-axis values
    (i.e. return an array that has been mapped from pixels to microns...)
    Inputs:
    file_name -[str] the file name containing the observed spectrum col0=pixel value col1=flux
    dlamb     -[float] a guess at the width of the order in microns
    low_lamb  -[float] a guess at the smallest wavelength in the order
    write     -[bool] if True the wavelength solution will be written to a file 
                      named 'wavelength_soln.txt'
    offset    -[float] a vertical offset so that you can see both spectra
    h         -[list, default=[]] if you want, you can supply coefficients for a 
                                  preliminary 2nd degree polynomial shift which
                                  may help align the spectra so you can identify similar 
                                  lines in both.
    iter      -[bool] if True, you will pick_corresponding_lines twice, this may help you
                      notice some lines you missed in the first pass (since the spectra will 
                      be better aligned the second time)
    trim      -[int] trim the ends of the observed spectrum by trim pixels.
    **kernkw  -keywords fed to wavelength_calibrate.make_gauss_kern
    Returns:
    len=2 tuple, zeroeth element is the pixel array, 1st element is the wavelength array.
    '''
    d=readcol(file_name,colNames=['p','f'])
    if trim:
        obsx=d['p'][trim:-trim]
        obsy=d['f'][trim:-trim]
    else:
        obsx=d['p']
        obsy=d['f']

    if len(h) ==0:
        xl=get_lines((low_lamb-0.005,low_lamb+dlamb+0.005))
        g=make_gauss_kern(len(xl[0]),**kernkw)
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

        a2=0.0
        a1=dlamb/float(max(d['p']))
        a0=low_lamb
        h=[a2,a1,a0]
    else:
        xl=get_lines(np.polyval(h,(-100,1100)))
        g=make_gauss_kern(len(xl[0]),**kernkw)
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

    denom=max(obsy.mean(),1.)
    obs=(np.polyval(h,obsx),obsy/denom+offset)
    lamcal=pick_corresponding_lines(sky,obs)

    if iter:
        print 'press enter when done to iterate once...'
        foo=raw_input()

        obs2=(lamcal.x.copy(),lamcal.y+offset/4.)

        xl=get_lines((lamcal.x[0]-0.001,lamcal.x[-1]+0.001))
        g=make_gauss_kern(len(xl[0]))
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

        lamcal2=pick_corresponding_lines(sky,obs2)

        print 'press enter when done ...'
        foo=raw_input()
        tot_transform=np.poly1d(lamcal2.transform_coefficients)(np.poly1d(lamcal.transform_coefficients)(np.poly1d(h)))
    else:
        print 'press enter when done'
        foo=raw_input()
        tot_transform=np.poly1d(lamcal.transform_coefficients)(np.poly1d(h))

    print 'transform coefficients:'
    print tot_transform.c
    print 'transform pretty print:'
    print tot_transform
    if write:
        fo=open('wavelength_soln.txt','w')
        fo.write('#wavelength transform: \n')
        fo.write('#%s\n' % tot_transform)
        fo.write('#coefficents:\n')
        fo.write(('#%e '*len(tot_transform.c))%tuple(tot_transform.c))
        fo.write('#\n')
        fo.write('#pix,lam (mu)\n')
        for pp in zip(d['p'],tot_transform(d['p'])):
            fo.write('%i %f\n' % pp)
        fo.close()
    return d['p'],tot_transform(d['p'])

def do_binary_calibration(file_name,dlamb=0.035,low_lamb=2.15,write=True,offset=0.4,h=[],iter=True,trim=False,source=1):
    '''Same as wavelength_calibrate.do_calibration but assumes that the file has
    three columns col0=pixel, col1=flux_star1, col2=flux_star2. You will
    wavelength_calibrate.pick_corresponding_lines for each
    '''
    print file_name
    d=readcol(file_name,colNames=['p','f1','f2'])
    if trim:
        obsx=d['p'][trim:-trim]
        obsy=(d['f1'][trim:-trim],d['f2'][trim:-trim])
    else:
        obsx=d['p']
        obsy=(d['f1'],d['f2'])

    if len(h) ==0:
        xl=get_lines((low_lamb-0.001,low_lamb+dlamb+0.003))
        g=make_gauss_kern(len(xl[0]))
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

        a2=0.0
        a1=dlamb/float(max(d['p']))
        a0=low_lamb
        h=[a2,a1,a0]
    else:
        xl=get_lines(np.polyval(h,(-150,1180)))
        g=make_gauss_kern(len(xl[0]))
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

    oy=obsy[source-1]
    denom=max(oy.mean(),1.)
    obs=(np.polyval(h,obsx),oy/denom+offset)
    lamcal=pick_corresponding_lines(sky,obs)

    if iter:
        print 'press enter when done to iterate once...'
        foo=raw_input()

        obs2=(lamcal.x.copy(),lamcal.y+offset/4.)

        xl=get_lines((lamcal.x[0]-0.001,lamcal.x[-1]+0.001))
        g=make_gauss_kern(len(xl[0]))
        out=smooth(xl[1],g)
        sky=(xl[0],out.real)

        lamcal2=pick_corresponding_lines(sky,obs2)

        print 'press enter when done ...'
        foo=raw_input()
        tot_transform=np.poly1d(lamcal2.transform_coefficients)(np.poly1d(lamcal.transform_coefficients)(np.poly1d(h)))
    else:
        print 'press enter when done'
        foo=raw_input()
        tot_transform=np.poly1d(lamcal.transform_coefficients)(np.poly1d(h))

    print 'transform coefficients:'
    print tot_transform.c
    print 'transform pretty print:'
    print tot_transform
    if write:
        fo=open('wavelength_soln%i.txt'%source,'w')
        fo.write('#wavelength transform: \n')
        fo.write('#%s\n' % tot_transform)
        fo.write('#coefficents:\n')
        fo.write(('#%e '*len(tot_transform.c))%tuple(tot_transform.c))
        fo.write('#\n')
        fo.write('#pix,lam (mu)\n')
        for pp in zip(d['p'],tot_transform(d['p'])):
            fo.write('%i %f\n' % pp)
        fo.close()
    return d['p'],tot_transform(d['p'])
    


def get_mu(x,y):
    return (x*y).sum()/y.sum()






        
