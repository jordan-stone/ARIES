from ARIES import *
from scipy.interpolate import interp1d
from fnmatch import filter as fnfilter
from os import listdir

def ccm_deredden(microns,flux,Av,Rv=3.1):
    '''
    I converted the idlastro function to python, I changed the name of Wave to
    microns and I changed the input to be Av instead of E(B-V).

    pro ccm_UNRED, wave, flux, ebv, funred, R_V = r_v
    ;+
    ; NAME:
    ;     CCM_UNRED
    ; PURPOSE:
    ;     Deredden a flux vector using the CCM 1989 parameterization 
    ; EXPLANATION:
    ;     The reddening curve is that of Cardelli, Clayton, and Mathis (1989 ApJ.
    ;     345, 245), including the update for the near-UV given by O'Donnell 
    ;     (1994, ApJ, 422, 158).   Parameterization is valid from the IR to the 
    ;     far-UV (3.5 microns to 0.1 microns).    
    ;
    ;     Users might wish to consider using the alternate procedure FM_UNRED
    ;     which uses the extinction curve of Fitzpatrick (1999).
    ; CALLING SEQUENCE:
    ;     CCM_UNRED, wave, flux, ebv, funred, [ R_V = ]      
    ;             or 
    ;     CCM_UNRED, wave, flux, ebv, [ R_V = ]      
    ; INPUT:
    ;     WAVE - wavelength vector (Angstroms)
    ;     FLUX - calibrated flux vector, same number of elements as WAVE
    ;             If only 3 parameters are supplied, then this vector will
    ;             updated on output to contain the dereddened flux.
    ;     EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
    ;             then fluxes will be reddened rather than deredenned.
    ;
    ; OUTPUT:
    ;     FUNRED - unreddened flux vector, same units and number of elements
    ;             as FLUX
    ;
    ; OPTIONAL INPUT KEYWORD
    ;     R_V - scalar specifying the ratio of total selective extinction
    ;             R(V) = A(V) / E(B - V).    If not specified, then R_V = 3.1
    ;             Extreme values of R(V) range from 2.75 to 5.3
    ;
    ; EXAMPLE:
    ;     Determine how a flat spectrum (in wavelength) between 1200 A and 3200 A
    ;     is altered by a reddening of E(B-V) = 0.1.   Assume an "average"
    ;     reddening for the diffuse interstellar medium (R(V) = 3.1)
    ;
    ;       IDL> w = 1200 + findgen(40)*50      ;Create a wavelength vector
    ;       IDL> f = w*0 + 1                    ;Create a "flat" flux vector
    ;       IDL> ccm_unred, w, f, -0.1, fnew  ;Redden (negative E(B-V)) flux vector
    ;       IDL> plot,w,fnew                   
    ;
    ; NOTES:
    ;     (1) The CCM curve shows good agreement with the Savage & Mathis (1979)
    ;             ultraviolet curve shortward of 1400 A, but is probably
    ;             preferable between 1200 and 1400 A.
    ;     (2)  Many sightlines with peculiar ultraviolet interstellar extinction 
    ;             can be represented with a CCM curve, if the proper value of 
    ;             R(V) is supplied.
    ;     (3)  Curve is extrapolated between 912 and 1000 A as suggested by
    ;             Longo et al. (1989, ApJ, 339,474)
    ;     (4) Use the 4 parameter calling sequence if you wish to save the 
    ;               original flux vector.
    ;     (5) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
    ;             curve (3.3 -- 8.0 um-1).    But since their revised curve does
    ;             not connect smoothly with longer and shorter wavelengths, it is
    ;             not included here.
    ;
    ; REVISION HISTORY:
    ;       Written   W. Landsman        Hughes/STX   January, 1992
    ;       Extrapolate curve for wavelengths between 900 and 1000 A   Dec. 1993
    ;       Use updated coefficients for near-UV from O'Donnell   Feb 1994
    ;       Allow 3 parameter calling sequence      April 1998
    ;       Converted to IDLV5.0                    April 1998
    ;-
    '''
    x = 1./ microns # Convert to inverse microns 
    a = np.empty_like(x)  
    b = np.empty_like(x)

    good = np.logical_and( x > 0.3,x < 1.1 )#Infrared
    if good.sum()>0:
        a[good] =  0.574 * x[good]**(1.61)
        b[good] = -0.527 * x[good]**(1.61)

    good = np.logical_and( x >= 1.1, x < 3.3)           #Optical/NIR
    if good.sum()>0:             #Use new constants from O'Donnell (1994)
        y = x[good] - 1.82
        #c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,    $ ;Original
        #            0.01979, -0.77530,  0.32999 ]               ;coefficients
        #c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,    $ ;from CCM89
        #        -0.62251,  5.30260, -2.09002 ]
        c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137,    \
        -1.718,   -0.827,    1.647, -0.505 ]        #from O'Donnell (1994)
        c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985,    \
        11.102,    5.491,  -10.805,  3.347 ]

        a[good] = np.polyval(c1[::-1], y)#numpy.polyval likes backwards coefficients...
        b[good] = np.polyval(c2[::-1], y)

    good = np.logical_and( x >= 3.3,x < 8)           #Mid-UV
    if good.sum()>0:
        y = x[good]
        F_a = np.zeros_like(y)    
        F_b = np.zeros_like(y)
        good1 = y > 5.9
        if good1.sum()>0:
            y1 = y[good1] - 5.9
            F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
            F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3

        a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
        b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b

    good = np.logical_and(x >= 8,x < 11 )         #Far-UV
    if good.sum()>0:
        y = x[good] - 8.
        c1 = [ -1.073, -0.628,  0.137, -0.070 ]
        c2 = [ 13.670,  4.257, -0.420,  0.374 ]
        a[good] = np.polyval(c1[::-1],y)
        b[good] = np.polyval(c2[::-1],y)

    A_lambda = Av * (a + b/Rv)
    return flux*10.**(0.4*A_lambda)     

def make_inds(w_arr,windows):
    '''
    Inputs:
    w_arr   - [1d array] an array of wavelengths
    windows - [list of tuples] each tuple defines the lower and upper
                               extent (in units of w_arr) of a window
    Returns:
    1d array - indices of w_arr which are within windows are set to 1
               0 otherwise
    '''
    inds=[]
    for interval in windows:
        inds.append(np.logical_and(w_arr>interval[0],w_arr<interval[1]))
    return np.any(inds,axis=0)

def grid_veil_fit(obs_spec, model_spec_path, shift_stepsize, shift_total_size,
                  template_glob='phoenix_model*.dat',windows=[(0,np.inf)]):
    '''
    Determine the necessary shift in wavelength necessary to align a template spectrum 
    and an observed spectrum
    Inputs:
    obs_spec        -[str] filename of observed spectrum (col0 is wavelength, col1 is flux)
    model_spec_path -[str] the path to the photospheric spectra files. Each file in this 
                           dir should have col0 wavelength and col1 flux
    shift_stepsize  -[float] the step size to use while gridding (only steps in one 
                             direction...)
    shift_total_size-[float] the total amount to shift during the fitting process.
    template_globe  -[str] a glob string that identifies all of the desired template
                           datafiles in model_spec_path
    windows         -[list of tuples] to be fed to analysis.make_inds. These are 
                                      windows around spectral features that the fitter will use
                                      while generating it's figure of merit value...
    Returns:
    None            - prints the best fit model name and shift
    '''
    spec=readcol(obs_spec,colNames=['w','f'])
    inds=make_inds(spec['w'],windows)
    model_photospheres={}
    fnames=sorted(fnfilter(listdir(model_spec_path),'phoenix_model*.dat'))
    print len(fnames)
    for f in fnames:
        model_photospheres[f]=readcol(model_spec_path+f,colNames=['w','f'])
        pm=np.polyfit(model_photospheres[f]['w'],model_photospheres[f]['f'],3)
        polynomial=np.polyval(pm,model_photospheres[f]['w'])
        polynomial-=polynomial.mean()
        model_photospheres[f]['f']-=polynomial
    fit_dict={}
    for shift in np.arange(0,shift_total_size,shift_stepsize):
        for model in fnames:
            model_func=interp1d(model_photospheres[model]['w']+shift,model_photospheres[model]['f'])
            diff=(spec['f']-model_func(spec['w']))**2
            fit_dict[(model,shift)]=diff[inds].sum()
    skeys=sorted(fit_dict,key=fit_dict.get)
    print skeys[0]

def get_ew(obs_spec,int_limits):
    '''return the equivalent width of a spectral feature inside
    the provided integration limits. The continuum level is 
    calculated as the mean outside the integration limits.
    Inputs:
    obs_spec   -[len=2 tuple] zeroeth element is the wavelength array
                             first element is the flux array
    int_limits -[len=2 tuple] the lower and upper limits (in wavelength
                              units).
    Returns:
    float      -the equivalent width (in the same units as the wavelength
                array) of the spectral feature within int_limits...
    '''
    w=obs_spec[0]
    f=obs_spec[1]
    inside=np.logical_and(w>int_limits[0],w<int_limits[1])
    f0=f[~inside].mean()
    return ((1-f[inside]/f0)*np.diff(w[inside]).mean()).sum()

def get_ew_median(obs_spec,int_limits):
    '''return the equivalent width of a spectral feature inside
    the provided integration limits. The continuum level is 
    calculated as the median outside the integration limits.
    Inputs:
    obs_spec   -[len=2 tuple] zeroeth element is the wavelength array
                             first element is the flux array
    int_limits -[len=2 tuple] the lower and upper limits (in wavelength
                              units).
    Returns:
    float      -the equivalent width (in the same units as the wavelength
                array) of the spectral feature within int_limits...
    '''
    w=obs_spec[0]
    f=obs_spec[1]
    inside=np.logical_and(w>int_limits[0],w<int_limits[1])
    f0=np.median(f[~inside])
    return ((1-f[inside]/f0)*np.diff(w[inside]).mean()).sum()

