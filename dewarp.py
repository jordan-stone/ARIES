from ARIES import *
from ARIES.trace import *

def get_dewarp_coefficients(traces,onto_xy='x',fit_degree=3,dewarp_degree=5):
    in_xs=[]
    fit_ys=[]
    out_xs=[]
    out_ys=[]

    print 'Polyfitting each trace and making polyfit2d_transform inputs...'
    for t in traces:
        x,y=unzip(t)
        in_xs.extend(x)
        p=np.polyfit(x,y,fit_degree)
        fit_y=np.polyval(p,x)
        fit_ys.extend(fit_y)
        outx,outy=unroll_trace(t,onto_xy=onto_xy,degree=fit_degree)
        out_xs.extend(outx)
        out_ys.extend(outy)

    print 'Getting transform coefficients...'
    coefs=poly2d.polyfit2d_transform(in_xs,fit_ys,out_xs,out_ys,order=dewarp_degree)
    return coefs

def dewarp(im,dewarp_coefs,return_yoffset=False):
    transformer=poly2d.poly2d_transform(*dewarp_coefs)

    print 'dewarping...(this step takes a bit)'
    inds=np.indices(im.shape)

    tx,ty=transformer(inds[1].flatten(),inds[0].flatten())
    print transformer
    xw=np.rint(tx.max()-tx.min())
    yw=np.rint(ty.max()-ty.min())
    print 'interpolating...(this step takes a bit)'
    dw_x,dw_y=np.indices((yw+1,xw+1))
    znew=griddata(zip(ty-ty.min(),tx-tx.min()),\
                  im.flatten(),\
                  (dw_x,dw_y),\
                  method='cubic',\
                  fill_value=0)
    if return_yoffset:
        return (znew,-ty.min())
    else:
        return (znew,)
    
def do_dewarp(im,traces,outname=None,plot=True,return_yoffset=False):
    coefs=get_dewarp_coefficients(traces)
    dw=dewarp(im,coefs,return_yoffset=return_yoffset)

    if outname is not None:
        print 'writing dewarped image to %s' % outname
        try:
            jFits.pyfits.writeto(outname,dw[0])
        except IOError:
            print 'file already exists?'
            print 'attempting to write %s' % (outname.split('.')[0]+'_2.fits')
            jFits.pyfits.writeto(outname.split('.')[0]+'_2.fits',dw[0])

    print 'Done!'
    if plot:
        print 'displaying dewarped image'
        ax=jFits.jDisplay(dw[0],log=True)

    return dw

def jroll(arr,step,axis):
    if step-float(int(step))==0.0:
        return np.roll(arr,step,axis=axis)
    else:
        inds=np.indices(arr.shape)
        if len(inds) < axis: raise ValueError
        shift_dum=np.zeros(len(inds))
        shift_dum[axis]=step
        in_data=map(lambda i,s:i.flatten()-s,inds,shift_dum)
        out=griddata(zip(*in_data),\
                     arr.flatten(),\
                     (inds[0],inds[1]),\
                     method='cubic',\
                     fill_value=0)
        return out

