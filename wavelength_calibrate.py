from ARIES import *
lines=readcol('/home/jstone/my_python/ARIES/transdata_1_5_mic',colNames=['wn','A'])
lines['w']=1.e4/lines['wn']
#precision of atmos cat
#since lines['w'] is a descending array, 
#the max diff will be the smallest diff (all vals of diff will be neg)
minstep=-(np.diff(lines['w']).max())

def get_lines(wave_lims):

    inds=np.logical_and(lines['w']>wave_lims[0],lines['w']<wave_lims[1])

    #make sure wavelengths are increasing
    l=lines['w'][inds][::-1]
    A=lines['A'][inds][::-1]

    wave_lengths=np.arange(l.min(),(l.max()+minstep),minstep)

    delta_spec=np.interp(wave_lengths,l,A)

    return (wave_lengths,delta_spec)

def make_gauss_kern(array_length,R=30000.,central_lambda=2.,minstep=minstep):
    delta_lambda=central_lambda/R
    sig=0.5*delta_lambda/minstep
    x=np.arange(array_length)
    g=np.exp( - ( (x-x.mean())**2. / (2*sig**2) ) )
    return g/g.sum()

def smooth(lines,kernel):
    return np.fft.fftshift( np.fft.ifft( np.fft.fft(lines) * np.fft.fft(kernel) ) )

def resids(obs_spec,sky_spec):
    obs_x,obs_y=obs_spec
    sky_x,sky_y=sky_spec
    return obs_y - np.interp(obs_x,sky_x,sky_y)
    
def mpFunc(obs_spec,sky_spec):
    def f(p,fjac=None):
        obs_x,obs_y=obs_spec
        obs_y=obs_y/obs_y.mean()
        sky_x,sky_y=sky_spec
        return [0,obs_y-np.interp( np.polyval(p,obs_x) ,sky_x, sky_y, left=0.0, right=0 )]
    return f

class pick_corresponding_lines:
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
    
def do_calibration(file_name,dlamb=0.035,low_lamb=2.15,write=True,offset=0.4,h=[],iter=True,trim=False):

    d=readcol(file_name,colNames=['p','f'])
    if trim:
        obsx=d['p'][trim:-trim]
        obsy=d['f'][trim:-trim]
    else:
        obsx=d['p']
        obsy=d['f']

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
        xl=get_lines(np.polyval(h,(-50,1080)))
        g=make_gauss_kern(len(xl[0]))
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
    


def get_mu(x,y):
    return (x*y).sum()/y.sum()






        
