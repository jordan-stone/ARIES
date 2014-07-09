from ARIES import *
from ARIES.trace import *

def get_dewarp_coefficients(traces,onto_xy='x',fit_degree=3,dewarp_degree=5):
    '''Given a set of ARIES spectral order traces (i.e. x,y coordinates along
    each trace) calculate the best-fit 2-d polynomial coefficients to rectify
    the image.
    Inputs:
    traces        -[list of lists of len=2 tuples] each element of traces is a list
                                                   of len=2 x,y coordinates. 1 element
                                                   per spectral order.
    onto_xy       -[str,'x' or 'y'] are you trying to straighten out the orders by making
                                    them horizontal('x') or vertical('y')?
    fit_degree    -[int]each of the spectral traces in traces is fit with a polynomial
                        of this degree before solving for the coefficients
    dewarp_degree -[int]the 2-d polynomial transformation will have this degree
    Returns:
    the coefficients of the 2d polynomial transform as returned by 
    poly2d.polyfit2d_tranform
    '''
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

def dewarp(im,dewarp_coefs,return_yoffset=False,yoffset=0,xoffset=0):
    '''apply a 2-d polynomial transformation to rectify the spectral
    traces in an ARIES echelle image.
    Input:
    im - [2d array] the echelle image
    dewarp_coefs - the coefficients defining the 2d polynomial transformation
                   (from e.g. dewarp.get_dewarp_coefficients)
    return_yoffset -[bool, default=False] After the dewarping process the dewarped
                                          array will have a new shape. If you'd like
                                          dewarp can return the y-offset that needs
                                          to be added to each of the old traces to find
                                          their new starting points in the dewarped array.
    yoffset -[int] this function assumes that the origin is in the bottom left corner
                   of the array, but if for somereason you'd like to move the origin
                   you can use this and the following keyword.
    xoffset -[int] similar to yoffset but for the x-coordinate of the origin.
    Returns:
    a tuple either len=1 or len=2 depending on whether return_yoffset is True or False
    the zeroeth element is the dewarped 2-d array. The first element, if it is requested,
    is the yoffset.
    '''
    transformer=poly2d.poly2d_transform(*dewarp_coefs)

    print 'dewarping...(this step takes a bit)'
    inds=np.indices(im.shape)

    tx,ty=transformer(inds[1].flatten()-xoffset,inds[0].flatten()-yoffset)
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
    '''Combine dewarp.get_dewarp_coefficients and dewarp.dewarp to dewarp an
    image starting with a list of traces and an image.
    Inputs:
    im     -[2d array] the ARIES echellogram
    traces -[list of lists of len=2 tuples] the traces of the spectral orders
    outname-[str] If supplied the dewarped array will be saved to this filename
                  as a fits file.
    plot   -[str] if true, display the default image
    Returns:
    same as for dewarp.dewarp
    '''
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


