from ARIES import *
from ARIES.trace import *
#Extraction object
class select_order:
    def __init__(self,im,centroid_width=10):
        self.im=im
        self.centroid_width=centroid_width
        self.ax=jFits.jDisplay(im,figsize=(16,8),subplot=121,log=True)
        self.cmap=self.ax.images[0].get_cmap()
        self.cid=self.ax.figure.canvas.mpl_connect('button_press_event',self)
        self.num_rows,self.num_cols=self.ax.images[0].get_size()
        print "click interesting order"
    def __call__(self,event):
        iny=event.ydata
        print 'clicked row: ',iny
        tot={}
        x=np.arange(self.num_cols)
        trace=zip(x,np.repeat(int(np.rint(iny)),len(x)))
        tot=[]
        for o in np.arange(11)-5:
            x=get_trace_vals(self.im,[trace],offset=o)[0]
            if len(x[~np.isnan(x)]) > 0:
                tot.append( sum( x[ ~np.isnan(x) ] ) )
            else:
                print x
                print o
                print trace
                tot.append(-9999)
        best_o=climb(5,tot)-5
        traceOpt=trace_optimize(self.im,[trace],centroid_width=self.centroid_width,offset=best_o)[0]
        y=np.array([y[1] for y in traceOpt])
        self.center=int(np.rint(np.mean(y[~np.isnan(y)])))
        print 'converged on row: ',self.center
        self.order=self.im[self.center-self.centroid_width:self.center+self.centroid_width,:]
        self.profile=self.order.sum(axis=1)
        self.profile/=self.profile.sum()
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.order_ax=jFits.jDisplay(self.order,\
                                     figure=self.ax.figure,\
                                     subplot=122,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)
        self.ax.figure.canvas.mpl_disconnect(self.cid)

    def extract(self):
        self.profile_cent, self.profile_width=get_gauss(self.profile)

        self.dif=np.rint((self.profile_cent+2*self.profile_width)-\
                    (self.profile_cent-2*self.profile_width))

        self.bot=np.rint(self.profile_cent)-np.rint(self.dif/2.)
        self.top=np.rint(self.profile_cent)+np.rint(self.dif/2.)

        self.fit_prof=self.profile[self.bot:self.top]
        self.fit_prof_cent=get_gauss(self.fit_prof)[0]

        self.spectrum=[]
        self.spatial=[]
        for col in xrange(self.order.shape[1]):
            snip=(self.order[:,col])[self.bot:self.top]
            cent,sig=get_gauss(snip)
            jog=np.rint(cent)-np.rint(self.fit_prof_cent)
            self.spatial.append(cent)
            yi=(snip[self.bot+jog:self.top+jog])
            self.spectrum.append(yi.sum())

        f=mpl.figure(figsize=(16,8))

        a2=f.add_subplot(223)
        a4=f.add_subplot(224)
        a3=f.add_subplot(222)

        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        a1=jFits.jDisplay(self.order,figure=f,subplot=221,log=True,aspect=aratio)

        a2.plot(self.spectrum)
        a2.set_xlim(0,len(self.spectrum))

        a3.hist(range(len(self.fit_prof)),\
                      weights=self.fit_prof,\
                      bins=len(self.fit_prof),\
                      edgecolor='k',\
                      facecolor='none')

        a4.plot(self.spatial)
        a4.set_xlim(0,len(self.spatial))

        a1.set_title('Order Image')
        a2.set_title('Extracted Spectrum')
        a3.set_title('Order profile')
        a4.set_title('Astrometric spectrum')
        
    def show_profile(self):
        f=mpl.figure()
        a=f.add_subplot(111)
        a.hist(range(len(self.profile)),\
                      weights=self.profile,\
                      bins=len(self.profile),\
                      edgecolor='k',\
                      facecolor='none')

    def hack_extract(self,center,sig):
        self.profile_cent=center
        self.profile_width=sig
        dif=np.rint((self.profile_cent+2*self.profile_width)-\
                    (self.profile_cent-2*self.profile_width))
        bot=self.profile_cent-dif/2.
        top=self.profile_cent+dif/2.
        fit_prof=self.profile[bot:top]
        self.spectrum=[]
        self.spatial=[]
        for col in xrange(self.order.shape[1]):
            cent,sig=get_gauss((self.order[:,col])[bot:top])
            self.spatial.append(cent)
            yi=(self.order[:,col])[max(0,cent-dif/2.):\
                                   min(cent+dif/2.,self.order.shape[0])]
            self.spectrum.append(yi.sum())
        f=mpl.figure(figsize=(16,8))
        a2=f.add_subplot(223)
        a4=f.add_subplot(224)
        a3=f.add_subplot(222)
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        a1=jFits.jDisplay(self.order,figure=f,subplot=221,log=True,aspect=aratio)
        a2.plot(self.spectrum)
        a4.plot(self.spatial)
        a3.hist(range(len(self.profile)),\
                      weights=self.profile,\
                      bins=len(self.profile),\
                      edgecolor='k',\
                      facecolor='none')
        a4.set_title('Astrometric spectrum')
        a3.set_title('Order profile')
        a2.set_title('Extracted Spectrum')
        a1.set_title('Order Image')

    def save_flux_spectrum(self,fname,x=None):
        fo=open(fname,'w')
        if x is not None:
            fo.write('lambda,f\n')
            for p in zip(x,self.spectrum):
                fo.write('%f %f\n' % p)
        else:
            fo.write('pix, f\n')
            for i,f in enumerate(self.spectrum):
                fo.write('%i %f\n' % (i,f))
        fo.close()

    def save_spatial_spectrum(self,fname,x=None):
        fo=open(fname,'w')
        if x is not None:
            fo.write('x,f\n')
            for p in zip(x,self.spatial):
                fo.write('%f %f\n' % p)
        else:
            fo.write('pix, f\n')
            for i,f in enumerate(self.spatial):
                fo.write('%i %f\n' % (i,f))
        fo.close()

    def save_profile(self,fname,x=None):
        fo=open(fname,'w')
        if x is not None:
            fo.write('x,f\n')
            for p in zip(x,self.profile):
                fo.write('%f %f\n' % p)
        else:
            fo.write('pix, f\n')
            for i,f in enumerate(self.profile):
                fo.write('%i %f\n' % (i,f))
        fo.close()

class select_order_argon:
    def __init__(self,im,argon_im,centroid_width=10):
        self.im=im
        self.argon_im=argon_im
        self.centroid_width=centroid_width
        self.ax=jFits.jDisplay(im,figsize=(16,8),subplot=121,log=True)
        self.cmap=self.ax.images[0].get_cmap()
        self.cid=self.ax.figure.canvas.mpl_connect('button_press_event',self)
        self.num_rows,self.num_cols=self.ax.images[0].get_size()
        print "click interesting order"

    def __call__(self,event):
        iny=event.ydata
        print 'clicked row: ',iny
        tot={}
        x=np.arange(self.num_cols)
        trace=zip(x,np.repeat(int(np.rint(iny)),len(x)))
        tot=[]
        for o in np.arange(11)-5:
            x=get_trace_vals(self.im,[trace],offset=o)[0]
            if len(x[~np.isnan(x)]) > 0:
                tot.append( sum( x[ ~np.isnan(x) ] ) )
            else:
                print x
                print o
                print trace
                tot.append(-9999)
        best_o=climb(5,tot)-5
        traceOpt=trace_optimize(self.im,[trace],centroid_width=self.centroid_width,offset=best_o)[0]
        y=np.array([y[1] for y in traceOpt])
        self.center=int(np.rint(np.mean(y[~np.isnan(y)])))
        print 'converged on row: ',self.center
        self.order_lims=(slice(self.center-self.centroid_width,self.center+self.centroid_width),slice(None))
        self.order=self.im[self.order_lims]
        self.argon_order=self.argon_im[self.order_lims]
        self.profile=self.order.sum(axis=1)/self.order.sum()
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.order_ax=jFits.jDisplay(self.order,\
                                     figure=self.ax.figure,\
                                     subplot=222,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)
        self.argon_order_ax=jFits.jDisplay(self.argon_order,\
                                     figure=self.ax.figure,\
                                     subplot=224,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)
        self.ax.figure.canvas.mpl_disconnect(self.cid)

    def set_new_order_lims(self,zeroeth,relative=True):
        if relative:
            new_lower0=self.order_lims[0].start+zeroeth[0]
            new_upper0=self.order_lims[0].start+zeroeth[1]
            self.order_lims=(slice(new_lower0,new_upper0),slice(None))
            self.order=self.im[self.order_lims]
            self.argon_order=self.argon_im[self.order_lims]

    def show_profile(self):
        f=mpl.figure()
        a=f.add_subplot(111)
        a.hist(range(len(self.profile)),\
                      weights=self.profile,\
                      bins=len(self.profile),\
                      edgecolor='k',\
                      facecolor='none')

    def show_order(self):
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.order_ax=jFits.jDisplay(self.order,\
                                     subplot=211,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)
        self.argon_order_ax=jFits.jDisplay(self.argon_order,\
                                     figure=self.order_ax.figure,\
                                     subplot=212,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)

    def extract(self,argon=False):
        if argon:
            order=self.argon_order
        else:
            order=self.order
        self.profile_cent, self.profile_width=get_gauss(self.profile)

        self.dif=np.rint((self.profile_cent+2*self.profile_width)-\
                    (self.profile_cent-2*self.profile_width))

        self.bot=np.rint(self.profile_cent)-np.rint(self.dif/2.)
        self.top=np.rint(self.profile_cent)+np.rint(self.dif/2.)

        self.fit_prof=self.profile[self.bot:self.top]
        self.fit_prof_cent=get_gauss(self.fit_prof)[0]

        self.spectrum=[]
        self.spatial=[]
        for col in xrange(order.shape[1]):
            snip=(order[:,col])[self.bot:self.top]
            cent,sig=get_gauss(snip)
            jog=np.rint(cent)-np.rint(self.fit_prof_cent)
            self.spatial.append(cent)
            yi=(snip[self.bot+jog:self.top+jog])
            self.spectrum.append(yi.sum())

        f=mpl.figure(figsize=(16,8))

        a2=f.add_subplot(223)
        a4=f.add_subplot(224)
        a3=f.add_subplot(222)

        aratio=0.5*order.shape[1]/float(order.shape[0])
        a1=jFits.jDisplay(order,figure=f,subplot=221,log=True,aspect=aratio)

        a2.plot(self.spectrum)
        a2.set_xlim(0,len(self.spectrum))

        a3.hist(range(len(self.fit_prof)),\
                      weights=self.fit_prof,\
                      bins=len(self.fit_prof),\
                      edgecolor='k',\
                      facecolor='none')

        a4.plot(self.spatial)
        a4.set_xlim(0,len(self.spatial))

        a1.set_title('Order Image')
        a2.set_title('Extracted Spectrum')
        a3.set_title('Order profile')
        a4.set_title('Astrometric spectrum')
        
        

def extract_all_orders(fname,trace_input,optimize=True):
    mpl.ioff()
    h,d=jFits.get_fits_array(fname)
    if optimize:
        tr=trace_optimize(d,trace_input)
    else:
        tr=trace_input
    dw_d,yoff=do_dewarp(d,tr,plot=False,return_yoffset=True)
    order_ys=map(lambda t: t[0][1]+yoff,tr)
    mpl.ion()
    ax=jFits.jDisplay(dw_d,log=True)
    for i,in_y in enumerate(order_ys):
        ax.lines=[]
        ax.plot(np.arange(1024),np.repeat(in_y,1024),'r-')
        ax.set_xlim(0,1023)
        ax.set_ylim(0,1023)
        mpl.show()
        order=auto_order(dw_d,in_y,i,fname.split('.')[0])
        order.jae_extract()
        order.save_flux_spectrum()
