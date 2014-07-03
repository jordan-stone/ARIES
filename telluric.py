from ARIES import *
import matplotlib.pyplot as mpl
from astropysics.coords.ephems import earth_pos_vel
import sidereal
from scipy.interpolate import interp1d
print "telluric standard HR1620, A7V"
NextGen=readcol('/home/jstone/my_python/ARIES/A7V_NextGen.out',colNames=['Angstroms','f'])

def vac2air(wavelength, density=1.0):
    """Calculate refractive index of air from Cauchy formula.

    Input: wavelength in Angstrom, density of air in amagat (relative to STP,
    e.g. ~10% decrease per 1000m above sea level).
    Returns N = (n-1) * 1.e6. 

    From http://phoenix.ens-lyon.fr/Grids/FORMAT
    The IAU standard for conversion from air to vacuum wavelengths is given
    in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
    Angstroms, convert to air wavelength (AIR) via: 

    AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    """                                                                                                              
    wl = np.array(wavelength)
    wl2inv = (1.e4/wl)**2
    refracstp = 272.643 + 1.2288 * wl2inv  + 3.555e-2 * wl2inv**2
    return wavelength/(1.+1.e-6*density * refracstp)
        
def redshift(wavelengths,velocity):
    '''velocity in m/s'''
    return (wavelengths*velocity/299792458.)+wavelengths

def rotationally_broaden(wavelengths,fluxes,vsini):
    '''vsini in m/s'''
    mn=min(wavelengths)
    mx=max(wavelengths)
    wl_interp=np.linspace(mn,mx,(mx-mn)/min(np.diff(wavelengths)))
    interp_f=interp1d(wavelengths,fluxes)
    mean_wl=mn+(mx-mn)/2.
    delta_wl=(mean_wl*vsini/299792458.)

    step_size=(np.max(wl_interp)-np.min(wl_interp))/len(wl_interp)
    sig=delta_wl/step_size
    x=np.arange(len(wl_interp))
    mu=0.5 + len(x)/2.

    g=np.exp(-0.5*( (x-mu)**2. /(sig**2.) ) )

    convolved=(np.fft.fftshift( np.fft.ifft( np.fft.fft(interp_f(wl_interp)) * np.fft.fft(g) ) )).real
    out_reinterp=interp1d(wl_interp,convolved)
    return out_reinterp(wavelengths)
    
def vgeo(strdate,ra,dec,vhel=0):
    '''strdate like yyyy-mm-dd[Thh[:mm[:ss]]]
    vhel in  km/s
    returns km/s'''
    dt=sidereal.parseDatetime(strdate)
    jd_obj=sidereal.JulianDate.fromDatetime(dt)
    x,v=earth_pos_vel(float(jd_obj))
    vgeo=v[0]*np.cos(dec)*np.cos(ra)+\
         v[1]*np.cos(dec)*np.sin(ra)+\
         v[2]*np.sin(dec)
    return vhel-vgeo


def correct(sx,sy,tx,ty,return_inds=False): 
    ave=sy.mean()
    telluric_func=interp1d(tx,ty,bounds_error=False,fill_value=ave) 
    return sy/telluric_func(sx) 

