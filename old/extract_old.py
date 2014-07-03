from ARIES import *
from ARIES.trace import *
#Extraction object
class order:
    def __init__(self,im,center,centroid_width=10):
        self.im=im
        self.centroid_width=centroid_width
        self.num_rows,self.num_cols=self.im.shape
        self.center=center
        zeroeth_slice=slice(self.center-np.floor(self.centroid_width/2.),
                            self.center+np.ceil (self.centroid_width/2.))
        first_slice=slice(None)
        self.order_lims=(zeroeth_slice,first_slice)
        self.order=self.im[self.order_lims]
        self.profile=self.order.sum(axis=1)/self.order.sum()

    def show_order(self):
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.order_ax=jFits.jDisplay(self.order,\
                                     subplot=111,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=aratio)

    def show_profile(self):
        f=mpl.figure()
        a=f.add_subplot(111)
        a.hist(range(len(self.profile)),\
                      weights=self.profile,\
                      bins=len(self.profile),\
                      edgecolor='k',\
                      facecolor='none')

    def extract(self,save_summary_fig_name=None,show=True):
        self.profile_cent, self.profile_width=get_gauss(self.profile)

        self.dif=np.rint((self.profile_cent+2*self.profile_width)-\
                    (self.profile_cent-2*self.profile_width))

        self.bot=max(np.rint(self.profile_cent)-np.rint(self.dif/2.),0)
        self.top=min(np.rint(self.profile_cent)+np.rint(self.dif/2.),len(self.profile))
        if self.bot==self.top:
            self.bot=0
            self.top=len(self.profile)

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
        a1=jFits.jDisplay(self.order,figure=f,subplot=221,log=True,aspect=aratio,show=show)

        a2.plot(self.spectrum)
        a2.set_xlim(0,len(self.spectrum))
        a2.set_ylim(0,2.5*np.mean(self.spectrum))

        a3.hist(range(len(self.fit_prof)),\
                      weights=self.fit_prof,\
                      bins=len(self.fit_prof),\
                      edgecolor='k',\
                      facecolor='none')

        a4.plot(self.spatial)
        a4.set_xlim(0,len(self.spatial))
        a4.set_ylim(np.mean(self.spatial)-0.5,np.mean(self.spatial)+0.5)

        a1.set_title('Order Image')
        a2.set_title('Extracted Spectrum')
        a3.set_title('Order profile')
        a4.set_title('Astrometric spectrum')
        if save_summary_fig_name:
            f.savefig(save_summary_fig_name)

    def adjust_order_lims(self,lims):
        if len(lims) == 2:
            slice1=self.order_lims[1]
        if len(lims) == 4:
            if self.order_lims[1].start==None:
                sl=slice(0,self.order.shape[1])
            else:
                sl=self.order_lims[1]
            new_lower1=sl.start+lims[2]
            new_upper1=sl.start+lims[3]
            slice1=slice(new_lower1,new_upper1)

        new_lower0=self.order_lims[0].start+lims[0]
        new_upper0=self.order_lims[0].start+lims[1]

        self.order_lims=(slice(new_lower0,new_upper0),slice1)

        self.order=self.im[self.order_lims]
        self.profile=self.order.sum(axis=1)/self.order.sum()

    def set_new_order_lims(self,newSlice0,newSlice1=slice(None)):
        self.order_lims=(newSlice0,newSlice1)
        self.order=self.im[self.order_lims]
        self.profile=self.order.sum(axis=1)/self.order.sum()

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

    def save_order(self,fname):
        jFits.pyfits.writeto(fname,self.order)

class select_order(order):
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
        zeroeth_slice=slice(self.center-np.floor(self.centroid_width/2.),
                            self.center+np.ceil (self.centroid_width/2.))
        first_slice=slice(None)
        self.order_lims=(zeroeth_slice,first_slice)
        self.order=self.im[self.order_lims]
        self.profile=self.order.sum(axis=1)
        self.profile/=self.profile.sum()
        self.aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.order_ax=jFits.jDisplay(self.order,\
                                     figure=self.ax.figure,\
                                     subplot=222,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=self.aratio)
        self.ax.figure.canvas.mpl_disconnect(self.cid)


class select_order_argon(select_order):
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
        select_order.__call__(self,event)
        self.argon=order(self.argon_im,self.argon_im.shape[0]/2)
        self.argon.set_new_order_lims(self.order_lims)
        self.aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        self.argon.ax=jFits.jDisplay(self.argon.order,\
                                     figure=self.ax.figure,\
                                     subplot=224,\
                                     color_scale=self.cmap,\
                                     log=True,\
                                     aspect=self.aratio)


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
