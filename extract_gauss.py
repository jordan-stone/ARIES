from ARIES import *
from ARIES.trace import *
import pdb
#Extraction object
class order:
    def __init__(self,im,center,nsources=1,centroid_width=10,optimize=False):
        '''Given the full dewarped array (im), focus on a single order
           centered on "center" with total width=centroid_width. 
           Optionally optimize "center" (explore from down by three
           pixels to up by three pixels) by maximizing the total flux
           in the extracted order. This can fail if the noise outside 
           of the true order is of large amplitude.'''
        self.nsources=nsources
        self.im=im
        self.centroid_width=centroid_width
        self.num_rows,self.num_cols=self.im.shape
        if optimize:
            test_sums=[]
            for c in np.linspace(center-3,center+3,6):
                zeroeth_slice=slice(c-np.floor(self.centroid_width/2.),
                                    c+np.ceil (self.centroid_width/2.))
                first_slice=slice(None)
                self.order_lims=(zeroeth_slice,first_slice)
                test_sums.append(self.im[self.order_lims].sum())
            center=(np.linspace(center-3,center+3)[test_sums==np.max(test_sums)])[0]
        self.center=center
        zeroeth_slice=slice(self.center-np.floor(self.centroid_width/2.),
                            self.center+np.ceil (self.centroid_width/2.))
        first_slice=slice(None)
        self.order_lims=(zeroeth_slice,first_slice)
        self.order=self.im[self.order_lims]
        self.profile=self.order.sum(axis=1)/self.order.shape[1]

    def show_order(self):
        '''Display the extracted order with a logarithmic scaling, 
           and a 1:2 aspect ratio.'''
        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        return jFits.jDisplay(self.order,\
                                     subplot=111,\
                                     log=True,\
                                     aspect=aratio)

    def show_profile(self):
        '''Display a plot of average spectral profile accross an order:
           self.profile.order.sum(axis=1)/self.order.shape[1]'''
        f=mpl.figure()
        a=f.add_subplot(111)
        a.hist(range(len(self.profile)),\
                      weights=self.profile,\
                      bins=len(self.profile),\
                      edgecolor='k',\
                      facecolor='none')
        return a

    def get_profile_params(self,profile,nsources=1,params=[]):
        if len(params)==0:
            params=sum(map(lambda n:
                                    [1,self.order.shape[0]*(2*n+1)/float(2*nsources),1],
                                    range(nsources)),
                                    [])
        p,m,e,x=gaussfitter.multigaussfit(range(len(profile)),
                                          profile,
                                          ngauss=nsources,
                                          params=params)

        p[1::3]=np.where(p[1::3]>0,p[1::3],0)
        p[1::3]=np.where(p[1::3]<len(self.profile),p[1::3],len(self.profile))
        profile_cent =p[1::3]
        profile_width=p[2::3]
        dif=np.rint((profile_cent.max()+2*profile_width.max())-\
                    (profile_cent.min()-2*profile_width.max()))

        bot=max(np.rint(profile_cent.mean())-np.rint(dif/2.),0)
        top=min(np.rint(profile_cent.mean())+np.rint(dif/2.),len(profile))
        if bot > top:
            print "something's wrong self.bot>self.top. Entering debugger"
            pdb.set_trace()
        if bot==top:
            bot=0
            top=len(self.profile)
        return p, bot, top
    
    def extract(self,nsources=1):
        '''Perform the spectral extraction on the selected order. A multi-gaussian model
           is used to fit the flux in each column of the order. Note that nsources can 
           be one. Since self.profile represents an average profile (i.e. with higher S/N)
           a fit is first made to self.profile using a naive guess of the initial 
           parameters: [1,self.order.shape[0]*(2*n+1)/float(2*nsources),1] where n 
           indexes the each source. The fit to self.profile is then used as the initial
           guess when fitting each column of self.order. A summary image of the extraction
           (the order, the profile, the flux, spatial, and optionally the second moment
           spectra) will be produced. All of the extracted components can then be saved 
           using methods of self specific to the task.'''        
        #generate naive guesses for fit to profile...
        self.p, self.bot, self.top =self.get_profile_params(
                                           self.profile,nsources=self.nsources)
        #Now perform a gaussian fit to each column of the order to extract the (spatial, flux
        #and sigma) spectra. the flux spectra will be the sum of the flux within 2 sigma of the
        #center after subtracting the gaussian model for the other source (if it exists).
        amp=[]
        spatial=[]
        sigma=[]
        models=np.empty((self.nsources,)+self.order.shape)
        for col in xrange(self.order.shape[1]):
            snip=(self.order[:,col])[self.bot:self.top]
            pos_inds=(snip>=0)
            if pos_inds.sum() < 3:
                pc=np.array(self.nsources*3*[-999])
            else:
                pc,mc,ec,xc=gaussfitter.multigaussfit(np.arange(len(snip))[pos_inds],
                                                      snip[pos_inds],
                                                      ngauss=self.nsources,
                                                      params=self.p)
                pc[1::3]=np.where(pc[1::3]>0,pc[1::3],0)#centers
                pc[1::3]=np.where(pc[1::3]<len(self.profile),pc[1::3],len(self.profile))
            #enforce ordering so that lower source is listed first (sources could have
            #switched during the fitting process)
            sort_inds=np.argsort(pc[1::3])
            amp.append     (pc[0::3][sort_inds])
            spatial.append (pc[1::3][sort_inds])
            sigma.append   (pc[2::3][sort_inds])
            #create the model arrays. enforce the same sorting as above...
            model_tmp=[gaussfitter.onedgaussian(np.arange(self.order.shape[0]),
                                                0,*pc[3*sort_inds[i]:3*(1+sort_inds[i])]) 
                                                for i in xrange(self.nsources)]
            models[:,:,col]=model_tmp
        self.models=np.array(models)
        #subtract the model of the other source from the image...
        self.model_subtracted_orders=self.order[None,:,:] #nsources=1
        if self.nsources==2:
            self.model_subtracted_orders=self.order[None,:,:]-self.models[::-1,:,:]
        mod_subtracted_profs=self.model_subtracted_orders.sum(axis=2)
        flux_sum=[]
        for i in xrange(self.nsources):
            psp,botsp,topsp=self.get_profile_params(mod_subtracted_profs[i])
            flux_sum.append(self.model_subtracted_orders[i,botsp:topsp,:].sum(axis=0))
        self.spectrum=np.array(zip(*flux_sum))
        self.amp=np.array(amp)
        self.spatial=np.array(spatial)
        self.sigma=np.array(sigma)


    def summary_im(self,save_summary_fig_name=None,show=True,plot_second=False):
        self.f=mpl.figure(figsize=(16,8))
        if plot_second:
            a2=self.f.add_subplot(223)#fspec
            a3=self.f.add_subplot(322)#prof
            a4=self.f.add_subplot(324)#sspec
            a5=self.f.add_subplot(326)#second
        else:
            a2=self.f.add_subplot(223)
            a3=self.f.add_subplot(222)
            a4=self.f.add_subplot(224)

        aratio=0.5*self.order.shape[1]/float(self.order.shape[0])
        a1=jFits.jDisplay(self.order,figure=self.f,subplot=221,log=True,aspect=aratio,show=show)

        colors=('b','g')
        for src in xrange(self.nsources):
            a2.plot(self.spectrum[:,src],color=colors[src])
        a2.set_xlim(0,self.spectrum.shape[0])
        try:
            lims=sorted((0,min(0.2*self.spectrum[50:-50].mean(axis=0).max()+self.spectrum[50:-50].max(),
                              2*self.spectrum[50:-50].mean(axis=0).max())))
            a2.set_ylim(*lims)
        except:
            lims=sorted((0,0.2*self.spectrum.mean(axis=0).max()+self.spectrum.max()))
            a2.set_ylim(*lims)
            

        a3.hist(np.arange(len(self.profile[self.bot:self.top]))+self.bot,\
                      weights=self.profile[self.bot:self.top],\
                      bins=len(self.profile[self.bot:self.top]),\
                      edgecolor='k',\
                      facecolor='none')
       # a3.plot(range(len(self.fit_prof)),m[self.bot:self.top])

        self.g=[]
        for i in xrange(self.nsources):
            self.g.append(
                gaussfitter.onedgaussian(
                    np.arange(len(self.profile[self.bot:self.top]))+self.bot,0,*self.p[3*i:3*(1+i)]))
            a3.plot(np.arange(len(self.profile[self.bot:self.top]))+self.bot,self.g[-1],color=colors[i])

        for src in xrange(self.nsources):
            a4.plot(self.spatial[:,src],color=colors[src])
        a4.set_xlim(0,self.spatial.shape[0])
        a4.set_ylim(self.spatial.mean(axis=0).min()-0.5,self.spatial.mean(axis=0).max()+0.5)

        if plot_second:
            for src in xrange(self.nsources):
                a5.plot(self.sigma[:,src])
            a5.set_xlim(0,self.spatial.shape[0])
            try:
                a5.set_ylim(self.sigma.mean(axis=0).min()-self.sigma[50:-50].std(),
                            self.sigma.mean(axis=0).max()+self.sigma[50:-50].std())
            except:
                a5.set_ylim(self.sigma.mean(axis=0).min()-self.sigma.std(),
                            self.sigma.mean(axis=0).max()+self.sigma.std())
            a5.set_title('Second order spectrum')

        a1.set_title('Order Image')
        a2.set_title('Flux Spectrum')
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

    def save_data(self,fname,save='flux',x=None):
        ''' save must be one of "flux", "spatial", "profile", "sigma", or "second"'''
        dat_dict={'flux':self.spectrum,
                  'spatial':self.spatial,
                  'profile':self.profile,
                  'sigma':self.sigma,
                  'second':self.sigma}
        key=[k for k in dat_dict.keys() if k.startswith(save)][0]
        data=dat_dict[key]
        fo=open(fname,'w')
        if x is not None:
            fo.write('lambda,f\n')
            for p in zip(x,data):
                fo.write('%f '*(1+len(p[1]) % (p[:1]+tuple(p[1]))+'\n'))
        else:
            fo.write('pix, f\n')
            for i,f in enumerate(data):
                fo.write('%i ' % i + ('%f '*len(f)) % tuple(f) + '\n')
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
                                     cmap=self.cmap,\
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
