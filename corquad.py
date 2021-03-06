from ARIES import *
def locate_cool(x,y):
    '''Find the cold pixels in the 3 other
    quadrants given a hot-pixel position
    Inputs:
    x - the x coordinate of hot pixel
    y - the y coordinate of hot pixel
    Returns:
    tuple - a len=3 tuple of len=2 tuples that
            are each the x,y location of a cold
            pixel
    '''
    if x > 512:
        xadd=-512
    else:
        xadd=512
    if y > 512:
        yadd=-512
    else:
        yadd=512
    return ((y,x+xadd),(y+yadd,x+xadd),(y+yadd,x))

def median_darks(darkList):
    '''median filter several ARIES images
    Inputs:
    darkList - [list, str] a list of filenames
    Returns:
    2d array - the median image
    '''
    dataL=np.array(map(lambda x:(jFits.get_fits_array(x))[1],darkList))
    return np.median(dataL,axis=0)

def get_hot_pix(d,max_pixels=100):
    '''Automatically find a set of hot pixels
    Inputs:
    d          -[2d Array] the dark image
    max_pixels -[int,default 100] 
    Returns:
    tuple - a len=2 tuple of numpy arrays. The zeroeth
            element is an array of x-coordinates, the
            first element is an array of y-coordinates
    '''
    hot=np.where(d>np.median(d)+5*np.std(d))

    #indsa=np.logical_and(hot[0]>256,hot[0]<502)
    #indsb=np.logical_and(hot[0]>778,hot[0]<1014)
    #inds1=np.logical_or(indsa,indsb)
    #indsc=np.logical_and(hot[1]>10,hot[1]<502)
    #indsd=np.logical_and(hot[1]>778,hot[1]<940)
    #inds2=np.logical_or(indsc,indsd)

    indsa=np.logical_and(hot[0]>11,hot[0]<502)
    indsb=np.logical_and(hot[0]>525,hot[0]<940)
    inds1=np.logical_or(indsa,indsb)
    indsc=np.logical_and(hot[1]>11,hot[1]<502)
    indsd=np.logical_and(hot[1]>525,hot[1]<940)
    inds2=np.logical_or(indsc,indsd)


    inds=np.logical_and(inds1,inds2)

    hotx=hot[1][inds]
    hoty=hot[0][inds]

    vals=np.array([d[yx[0],yx[1]] for yx in zip(hoty,hotx)])
    if len(vals) > max_pixels:
        big_inds=np.argsort(vals)[-max_pixels:]
    else:
        big_inds=np.argsort(vals)

    hotx=hotx[big_inds]
    hoty=hoty[big_inds]
    return hotx, hoty

def vet_hot_pix(hotx,hoty,d):
    '''from a large set of hot pixels, identify a sub-set
    of very-hot and isolated pixels that will be best for
    calculating the corquad coefficients. You will loop
    through each hot pixel and view it and it's cross
    talk images.
    Inputs:
    hotx - [1d array] an array of x-coordinates
    hoty - [1d array] an array of y-coordinates
    d    - [2d array] the image
    Returns:
    tuple - a len=2 tuple, the zeroeth element is an array
            of x-coordinates, the first element is an array of 
            y-coordinates
    '''
    f=mpl.figure(figsize=(5,5))
    imax=jFits.jDisplay(d,figsize=(5,5),log=True,cmap='gray')
    cool=map(locate_cool,hotx,hoty)
    hot_qual=[]
    ghotx=[]
    ghoty=[]
    for point in zip(hoty,hotx,cool):
        print point[0],point[1]
        imax.plot([point[1]],[point[0]],'ro')
        imax.plot([point[2][0][1]],[point[2][0][0]],'bo')
        imax.plot([point[2][1][1]],[point[2][1][0]],'bo')
        imax.plot([point[2][2][1]],[point[2][1][0]],'bo')
        imax.set_xlim(0,1023)
        imax.set_ylim(0,1023)
        a=d[point[0]-9:point[0]+9,point[1]-9:point[1]+9]
        ax=jFits.jDisplay(a,log=True,figure=f,subplot=221,cmap='gray')
        ax.set_title('bright')
        mpl.show()
        for i, p in enumerate(point[2]):
            print p[0], p[1]
            cold1=d[p[0],p[1]]
            cold2=d[p[0],p[1]+1]
            cold3=d[p[0],p[1]+2]
            print 'colds: ',cold1,cold2,cold3
            median=np.median(d[p[0]-10:p[0]+10,p[1]-10:p[1]+10])
            aa=d[p[0]-9:p[0]+9,p[1]-9:p[1]+9]
            ax2=jFits.jDisplay(aa,\
                               log=True,\
                               figure=f,\
                               subplot='22%i' %(i+2),\
                               cmap='gray')
            ax2.set_title('dark')
            mpl.show()
        print 'Hows it look (1/0)?'
        qual=raw_input()
        while qual != '0' and qual !='1':
            print "I didn't understand that: if worth keeping type 1, otherwise type 0"
            qual=raw_input()
        if qual=='1':
            ghotx.append(point[1])
            ghoty.append(point[0])
        imax.lines=[]
        imax.plot(ghotx,ghoty,'go')
        imax.set_xlim(0,1023)
        imax.set_ylim(0,1023)
    imax.lines=[]
    imax.plot(ghotx,ghoty,'go')
    imax.set_xlim(0,1023)
    imax.set_ylim(0,1023)
    return ghotx, ghoty

def make_corquad(hotx,hoty,d):
    '''Calculate the corquad coefficients.
    Inputs: 
    hotx - [1d array] x-coordinates of hot pixels
    hoty - [1d array] y-coordinates of hot pixels
    d    - [2d array] the image
    Return:
    tuple - a len=3 tuple with the 0th, 1st, and 2nd corquad coefficients
    '''
    cool=map(locate_cool,hotx,hoty)
    kern0=[]
    kern1=[]
    kern2=[]
    for point in zip(hoty,hotx,cool):
        print point[0],point[1]
        hot0=d[point[0],point[1]-1]
        hot1=d[point[0],point[1]]
        hot2=d[point[0],point[1]+1]
        hot3=d[point[0],point[1]+2]
        print 'hots: ',hot0,hot1,hot2,hot3
        for p in point[2]:
            print p[0], p[1]
            cold1=d[p[0],p[1]]
            cold2=d[p[0],p[1]+1]
            cold3=d[p[0],p[1]+2]
            print 'colds: ',cold1,cold2,cold3
            median=np.median(d[p[0]-10:p[0]+10,p[1]-10:p[1]+10])
            
            k0=(median-cold1)/hot1
            k1=((median-cold2)-(hot2*k0))/hot1
            k2=((median-cold3)-(hot3*k0)-(hot2*k1))/hot1
            
            kern0.append(k0)
            kern1.append(k1)
            kern2.append(k2)
                
    print 'kern 1: ',np.mean(kern0), ' +/- ',np.std(kern0)         
    print 'kern 2: ',np.mean(kern1), ' +/- ',np.std(kern1)         
    print 'kern 3: ',np.mean(kern2), ' +/- ',np.std(kern2)         

    return np.mean(kern0), np.mean(kern1), np.mean(kern2)

def do_make_corquad(darkList):
    '''Combine corquad.median_darks, corquad.get_hot_pix, corquad.vet_hot_pix
    and corquad.make_corquad.
    Inputs:
    darkList - [list, str] a list of dark image filenames
    Returns:
    tuple - a len=3 tuple with the 0th, 1st, and 2nd corquad coefficients
    '''
    d=median_darks(darkList)
    hotx,hoty=get_hot_pix(d)
    goodx, goody=vet_hot_pix(hotx,hoty,d)
    fo=open('corquad_hotpix.txt','w')
    fo.write('hotx, hoty')
    for p in zip(goodx,goody):
        fo.write('%i %i\n' %p)
    return make_corquad(goodx, goody,d)
    

def corquad_grid(kern_mu_sig,btlr_box,fname,
                 cor_coefs_file="/home/jstone/.corquad_backup",
                 write_dot_corquad_file='/home/jstone/writeDotCorquad.sh',
                 corquad_executable_file='/home/jstone/corquad-linux'):
    '''fit for the best corquad coeficients using a grid-based approach
    Inputs:
    kern_mu_sig            - a len=3 tuple. Each element is a len=2 tuple including 
                             a good guess of the mean and std of the corquad coefficient 
                             value. These values will define the range of the grid, which 
                             will go from mu+/std for each coefficient
    btlr_box               - the bottom, top, left, and right extents of
                             a sub-array the standard deviation of which will be the 
                             figure of merit for the fit. (usually want to put this around
                             a cool pixel...)
    fname                  - the filename of the dark image the fit will be run on
    cor_coefs_file_backup  - the file name of a file containing the original corquad coefficients
    write_dot_corquad_file - path to the writeDotCorquad.sh executable
    cor_coefs_file         - path to the ".corquad" file that the corquad executable will read
    corquad_executable_file- path to the corquad executable.
    Returns:
    None                    - will print the 10 best sets of coefficients...

    '''
    params=readcol(cor_coefs_file_backup,\
                   colNames=['name','value'],format=['a','f'])
    bname='q'+os.path.basename(fname)
    qname=os.path.join(os.path.dirname(fname)+bname)
    devs={}
    subprocess.check_call(["touch",qname])
    for kern0 in np.linspace(kern_mu_sig[0][0]-kern_mu_sig[0][1],\
                             kern_mu_sig[0][0]+kern_mu_sig[0][1],\
                             11):
        for kern1 in np.linspace(kern_mu_sig[1][0]-kern_mu_sig[1][1],\
                                 kern_mu_sig[1][0]+kern_mu_sig[1][1],\
                                 11):
            for kern2 in np.linspace(kern_mu_sig[2][0]-kern_mu_sig[2][1],\
                                     kern_mu_sig[2][0]+kern_mu_sig[2][1],\
                                     11):
                os.remove(qname)
                args=[\
                "%s %s" % (params['name'][0],params['value'][0]),\
                "%s %s" % (params['name'][1],params['value'][1]),\
                "%s %s" % (params['name'][2],params['value'][2]),\
                "%s %s" % (params['name'][3],params['value'][3]),\
                "%s %s" % (params['name'][4],params['value'][4]),\
                "%s %s" % (params['name'][5], str(kern0)),\
                "%s %s" % (params['name'][6], str(kern1)),\
                "%s %s" % (params['name'][7], str(kern2)) ]
                subprocess.check_call([write_dot_corquad_file,\
                                       cor_coefs_file]+\
                                       args)
                subprocess.check_call([corquad_executable_file,\
                                       fname])
                h,d=jFits.get_fits_array(qname)
                devs[(kern0,kern1,kern2)]=np.std(d[btlr_box[0]:btlr_box[1],\
                                                   btlr_box[2]:btlr_box[3]])


    skeys=sorted(devs.keys(),key=devs.get)
    print 'The best 10 kern values and the std merit value'
    print 'kern0, kern1, kern2, std'
    for k in skeys[:10]:
        print k, devs[k]
    #fo=open('corquadFitParams.txt','w')
    #fo.write('kern0 kern1 kern2 std\n')
    #for k in sorted(devs,key=devs.get):
    #    fo.write(('%.5f '*4+'\n') % (k+(devs[k],)))
    #fo.close()

