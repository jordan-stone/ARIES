from ARIES import *
from ARIES.corquad import get_hot_pix

def fix_hots(arr,p):
    '''ARIES hot pixels look like crosshairs. Try to fix them
    Inputs:
    arr -[2d array] the image
    p   -[len=2 tuple] the point
    Returns:
    None, but updates arr...
    '''
    stamp=arr[p[0]-1:p[0]+2,p[1]-1:p[1]+2]
    copy=stamp.copy()
    flat_copy=copy.view().reshape(9)
    spokes=flat_copy[1::2]
    hub=flat_copy[4:5]
    neighbor_vals=np.array([stamp[0,:][::2],
                            stamp[:,0][::2],
                            stamp[:,2][::2],
                            stamp[2,:][::2]])
    replace_vals=neighbor_vals.mean(axis=1)
    spokes[:]=replace_vals
    hub[:]=replace_vals[np.array((1,2))].mean()
    stamp[:,:]=copy

def do_fix_hots(dark,sci,returnhots=False):
    '''Find and fix the hot pixels in an image using a dark image
    as a reference. Optionally return the location of the hotpixels
    Inputs:
    dark -[2d array] the dark image
    sci  -[2d array] the science image (echellogram)
    returnhots -[bool] if True, the hot pixels will be 
                       returned in a list of len=2 tuples.
    Returns:
    None but updates sci
    or
    updates sci and returns a list of len=2 tuples, the coordinates
    of the hot pixels
    '''
    hotx,hoty=get_hot_pix(dark,max_pixels=1e6)    
    hotxr=hotx+1
    hots=zip(hotx,hoty)
    rights=zip(hotxr,hoty)
    for r in rights:
        if r in hots:
            hots.pop(hots.index(r))
    print 'Found %i 5-sigma outliers'%len(hots)
    for p in hots:
        fix_hots(sci,p[::-1])
    if returnhots:
        return zip(*hots)
