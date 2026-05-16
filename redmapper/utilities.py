"""Miscellaneous methods and classes for redmapper.
"""

# miscellaneous utilities and functions

import numpy as np
from scipy import special
from scipy.linalg import solve_banded
from importlib import resources
import scipy.interpolate as interpolate
import fitsio
from scipy.special import erf
from numpy import random
import hpgeom as hpg
import hpgeom.healpy_compat as hpc
import esutil
import sys
import os
import warnings

###################################
## Useful constants/conversions ##
###################################
TOTAL_SQDEG = 4 * 180**2 / np.pi
SEC_PER_DEG = 3600

def decode_string(s):
    """
    Decode a string/bytes object safely for Python 3.

    Parameters
    ----------
    s: `str` or `bytes`
       String or bytes to decode.
    """
    if hasattr(s, 'decode'):
        return s.decode().rstrip()
    return s.rstrip() if hasattr(s, 'rstrip') else s

def astro_to_sphere(ra, dec):
    """
    Convert astronomical ra/dec to healpix theta, phi coordinates.

    Parameters
    ----------
    ra: `np.array`
       Float array of right ascension
    dec: `np.array`
       Float array of declination

    Returns
    -------
    theta: `np.array`
       Float array of healpix theta
    phi: `np.array`
       Float array of healpix phi
    """
    return np.radians(90.0-dec), np.radians(ra)

#Equation 7 in Rykoff et al. 2014
def chisq_pdf(data, k):
    """
    Compute the chi-squared probability density function for an array of values
    and given number of degrees of freedom.  Implementing Equation 7 of
    Rykoff++2014.

    Parameters
    ----------
    data: `np.array`
       Float array of values
    k: `float`
       Number of effective degrees of freedom.

    Returns
    -------
    pdf: `np.array`
       Float array of pdf values
    """
    normalization = 1./(2**(k/2.) * special.gamma(k/2.))
    return normalization * data**((k/2.)-1) * np.exp(-data/2.)

def gaussFunction(x, *p):
    """
    Compute a normalizes Gaussian G(x) for a given set of parameters

    Parameters
    ----------
    x: `np.array`
       Float array of x values
    A: `float`
       Normalization of the Gaussian
    mu: `float`
       Mean value (mu) for the Gaussian
    sigma: `float`
       Gaussian sigma

    Returns
    -------
    pdf: `np.array`
       Float array of Gaussian pdf values
    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2./(2.*sigma**2))

def schechter_pdf(x, alpha=-1.0, mstar=0.0):
    """
    Compute an unnormalized Schechter function pdf.

    Parameters
    ----------
    x: `np.array`
       Float array of x (magnitude) values
    alpha: `float`, optional
       Faint-end slope of the Schechter function.  Default is -1.0.
    mstar: `float`, optional
       Pivot magnitude mstar.  Default is 0.0.

    Returns
    -------
    pdf: `np.array`
       Float array of unnormalized Schechter pdf values.
    """
    pdf = 10.**(0.4*(alpha + 1.0) * (mstar - x)) * np.exp(-10.**(0.4 * (mstar - x)))
    return pdf

def nfw_pdf(x, rscale=0.15, corer=0.1, radfactor=False):
    """
    Compute an unnormalized projected NFW pdf.

    Parameters
    ----------
    x: `np.array`
       Float array of x (radius) values
    rscale: `float`, optional
       NFW radial scaling.  Default is 0.15 Mpc.
    corer: `float`, optional
       Maximum radius to substitute a flat core.  Default is 0.1 Mpc.
    radfactor: `bool`, optional
       Multiply by 2*pi*r factor?  Default is False.

    Returns
    -------
    pdf: `np.array`
       Unnormalized projected NFW pdf.
    """

    xscale = x / rscale
    corex = corer / rscale

    sigx = np.zeros(x.size)

    low, = np.where(xscale < corex)
    mid, = np.where((xscale >= corex) & (xscale <= 0.999))
    high, = np.where((xscale >= 1.001) & (xscale < 10.0 / rscale))
    other, = np.where((xscale > 0.999) & (xscale < 1.001))

    if low.size > 0:
        arg = np.sqrt((1. - corex)/(1. + corex))
        pre = 2./(np.sqrt(1. - corex**2))
        front = 1./(corex**2. - 1)
        sigx[low] = front * (1. - pre*0.5*np.log((1. + arg)/(1. - arg)))

    if mid.size > 0:
        arg = np.sqrt((1. - xscale[mid])/(1. + xscale[mid]))
        pre = 2./(np.sqrt(1. - xscale[mid]**2.))
        front = 1./(xscale[mid]**2. - 1.)
        sigx[mid] = front * (1. - pre*0.5*np.log((1. + arg)/(1. - arg)))

    if high.size > 0:
        arg = np.sqrt((xscale[high] - 1.)/(xscale[high] + 1.))
        pre = 2./(np.sqrt(xscale[high]**2 - 1.))
        front = 1./(xscale[high]**2 - 1)
        sigx[high] = front * (1. - pre*np.arctan(arg))

    if other.size > 0:
        xlo, xhi = 0.999, 1.001
        arglo, arghi = np.sqrt((1 - xlo)/(1 + xlo)), np.sqrt((xhi - 1)/(xhi + 1))
        prelo, prehi = 2./np.sqrt(1.-xlo**2), 2./np.sqrt(xhi**2 - 1)
        frontlo, fronthi = 1./(xlo**2 - 1), 1./(xhi**2 - 1)
        testlo = frontlo * (1. - prelo*0.5*np.log((1 + arglo)/(1 - arglo)))
        testhi = fronthi * (1. - prehi*np.arctan(arghi))
        sigx[other] = (testlo + testhi)/2.

    if radfactor:
        sigx *= (2. * np.pi * x)

    return sigx



def cubic_spline_compute_y2(x, y, yp=None):
    """
    Compute the second derivatives for a cubic spline.

    Parameters
    ----------
    x: `np.array`
       Float array of node positions
    y: `np.array`
       Float array of node values
    yp: `str` or `list`, optional
       Type of spline. Default is None, which is "natural".

    Returns
    -------
    y2: `np.array`
       Float array of second derivatives at nodes.
    """
    npts = len(x)
    mat = np.zeros((3, npts))
    # enforce continuity of 1st derivatives
    mat[1,1:-1] = (x[2:  ]-x[0:-2])/3.
    mat[2,0:-2] = (x[1:-1]-x[0:-2])/6.
    mat[0,2:  ] = (x[2:  ]-x[1:-1])/6.
    bb = np.zeros(npts)
    bb[1:-1] = ((y[2:  ]-y[1:-1])/(x[2:  ]-x[1:-1]) -
                (y[1:-1]-y[0:-2])/(x[1:-1]-x[0:-2]))
    if yp is None: # natural cubic spline
        mat[1,0] = 1.
        mat[1,-1] = 1.
        bb[0] = 0.
        bb[-1] = 0.
    elif yp == '3d=0':
        mat[1, 0] = -1./(x[1]-x[0])
        mat[0, 1] =  1./(x[1]-x[0])
        mat[1,-1] =  1./(x[-2]-x[-1])
        mat[2,-2] = -1./(x[-2]-x[-1])
        bb[ 0] = 0.
        bb[-1] = 0.
    else:
        mat[1, 0] = -1./3.*(x[1]-x[0])
        mat[0, 1] = -1./6.*(x[1]-x[0])
        mat[2,-2] =  1./6.*(x[-1]-x[-2])
        mat[1,-1] =  1./3.*(x[-1]-x[-2])
        bb[ 0] = yp[0]-1.*(y[ 1]-y[ 0])/(x[ 1]-x[ 0])
        bb[-1] = yp[1]-1.*(y[-1]-y[-2])/(x[-1]-x[-2])
    y2 = solve_banded((1,1), mat, bb)
    return y2

def cubic_spline_interpolate(x_eval, x, y, y2, fixextrap=False):
    """
    Compute cubic spline interpolation.

    Parameters
    ----------
    x_eval: `np.array`
       Float array of x values to compute interpolation
    x: `np.array`
       Float array of node positions
    y: `np.array`
       Float array of node values
    y2: `np.array`
       Float array of second derivatives at nodes
    fixextrap: `bool`, optional
       Fix the extrapolation at the end of the node positions.
       Default is False.

    Returns
    -------
    y_eval: `np.array`
       Spline interpolated values at x_eval
    """
    npts = len(x)
    lo = np.searchsorted(x, x_eval)-1
    lo = np.clip(lo, 0, npts-2)
    hi = lo + 1
    dx = x[hi] - x[lo]
    a = (x[hi] - x_eval)/dx
    b = (x_eval-x[lo])/dx
    vals = (a*y[lo]+b*y[hi]+
            ((a**3-a)*y2[lo]+(b**3-b)*y2[hi])*dx**2./6.)

    if fixextrap:
        if np.isscalar(x_eval):
            if x_eval < x[0]:
                return y[0]
            if x_eval > x[-1]:
                return y[-1]
            return vals

        lo_extrap, = np.where(x_eval < x[0])
        vals[lo_extrap] = y[0]

        hi_extrap, = np.where(x_eval > x[-1])
        vals[hi_extrap] = y[-1]

    return vals

def read_mstar(survey, band):
    """
    Read the MStar(z) look-up table.

    Parameters
    ----------
    survey: `str`
       Name of survey to get mstar look-up table
    band: `str`
       Name of band to get mstar look-up table

    Returns
    -------
    mstar_data: `dict`
       Dictionary containing mstar data and spline coefficients.
    """
    survey = survey.strip()
    band = band.strip()

    ref = resources.files("redmapper").joinpath("data/mstar/mstar_%s_%s.fit" % (survey, band))
    with resources.as_file(ref) as f:
        mstar_file = f.as_posix()

    if not os.path.isfile(mstar_file):
        raise IOError("Could not find mstar file %s" % (mstar_file))

    mstar_arr = fitsio.read(mstar_file, ext=1, upper=True)
    y2 = cubic_spline_compute_y2(mstar_arr['Z'], mstar_arr['MSTAR'])

    return {
        'z': mstar_arr['Z'],
        'mstar': mstar_arr['MSTAR'],
        'y2': y2
    }

def get_mstar(mstar_data, z):
    """
    Return mstar at redshifts z.

    Parameters
    ----------
    mstar_data: `dict`
       Dictionary containing mstar data and spline coefficients.
    z: `np.array`
       Float array of redshifts

    Returns
    -------
    mstar: `np.array`
       mstar at redshifts z
    """
    return cubic_spline_interpolate(z, mstar_data['z'], mstar_data['mstar'], mstar_data['y2'])

def read_redgal_initial_colors(redgal_template):
    """
    Read the RedGalInitialColors look-up table.

    Parameters
    ----------
    redgal_template: `str`
       Name of redgal_template file to get look-up table.

    Returns
    -------
    redgal_data: `dict`
       Dictionary containing redgal template data.
    """
    from .catalog import Catalog

    ref = resources.files("redmapper").joinpath("data/initcolors/%s" % (redgal_template))
    with resources.as_file(ref) as f:
        template_file = f.as_posix()

    if not os.path.isfile(template_file):
        template_file = os.path.abspath(redgal_template)

    if not os.path.isfile(template_file):
        raise IOError("Could not find redgal_template file %s in resource or path." % (redgal_template))

    template = Catalog.from_fits_file(template_file, ext=1)
    template_hdr = fitsio.read_header(template_file, ext=1)

    if ',' in template_hdr['BANDS']:
        template_bands = template_hdr['BANDS'].rstrip().split(',')
    else:
        template_bands = list(template_hdr['BANDS'].rstrip())

    color_names = []
    for i in range(len(template_bands) - 1):
        color_names.append('%s-%s' % (template_bands[i].upper(), template_bands[i+1].upper()))

    return {
        'z': template.z,
        'color': template.color,
        'bands': template_bands,
        'color_names': color_names,
        'template_file': template_file
    }

def get_redgal_initial_color(redgal_data, band1, band2, z):
    """
    Return the initial color guess (band1 - band2) at redshift z.

    Parameters
    ----------
    redgal_data: `dict`
       Dictionary containing redgal template data.
    band1: `str`
       The first band in the color
    band2: `str`
       The second band in the color
    z: `np.array`
       Float array of redshifts

    Returns
    -------
    color: `np.array`
       Initial red galaxy color at redshifts z
    """
    name = '%s-%s' % (band1.upper(), band2.upper())
    if name not in redgal_data['color_names']:
        raise ValueError("Color %s not in initial template file %s." % (name, redgal_data['template_file']))

    index = redgal_data['color_names'].index(name)
    return interpol(redgal_data['color'][:, index], redgal_data['z'], z)

def calc_theta_i(mag, mag_err, maxmag, limmag):
    """
    Calculate the luminosity function smooth cutoff function, theta_i.

    Parameters
    ----------
    mag: `np.array`
       Magnitude values to compute theta_i
    mag_err: `np.array`
       Magnitude errors to compute theta_i
    maxmag: `float`
       Maximum possible magnitude in the catalog (hard cutoff)
    limmag: `float`
       Limiting magnitude to compute smooth cutoff

    Returns
    -------
    theta_i: `np.array`
       Float array of theta_i(mag, mag_err)
    """
    theta_i = np.ones(mag.size)
    eff_lim = np.clip(maxmag, 0, limmag)
    dmag = eff_lim - mag
    calc, = np.where(dmag < 5.0 * mag_err)
    if calc.size > 0:
        theta_i[calc] = 0.5 + 0.5 * erf(dmag[calc] / (np.sqrt(2) * mag_err[calc]))
    hi, = np.where(mag > limmag)
    if hi.size > 0:
        theta_i[hi] = 0.0
    return theta_i

def apply_errormodels(maskgals, mag_in, b=None, err_ratio=1.0, fluxmode=False,
                      nonoise=False, inlup=False, lnscat=None, sigma0=0.0, rng=None):
    """
    Apply error models to a set of magnitudes.

    Parameters
    ----------
    maskgals: `redmapper.Catalog`
       Catalog of maskgals which have individual limmag, nsig values.
    mag_in: `np.array`
       Float array of raw magnitudes
    b: `np.array`, optional
       Luptitude softening parameters.  Default is None (not luptitudes)
    err_ratio: `float`, optional
       Error scaling ratio (for testing/adjustments).  Default is 1.0.
    fluxmode: `bool`, optional
       Return fluxes rather than magnitudes/luptitudes.  Default is False.
    nonoise: `bool`, optional
       Do not apply noise to output fluxes?  Default is False.
    inlup: `bool`, optional
       Input magnitudes are actually luptitudes.  Default is False.
       If True, b array must be supplied.
    lnscat: `float`, optional
       Apply additional ln(scatter) term (increasing errors).
       Default is None (no ln(scatter) term).
    sigma0: `float`, optional
       Additional noise floor term to apply.  Default is 0.0 (no floor).
    rng : `np.random.RandomState`, optional
        Random number generator.

    Returns
    -------
    mag: `np.array`
       Array of noised magnitudes/luptitudes/fluxes
    mag_err: `np.array`
       Array of magnitude/luptitude/flux errors
    """
    if rng is None:
        rng = np.random.RandomState()

    f1lim = 10.**((maskgals.limmag - maskgals.zp[0])/(-2.5))
    fsky1 = (((f1lim**2.) * maskgals.exptime)/(maskgals.nsig[0]**2.) - f1lim)
    fsky1 = np.clip(fsky1, 0.001, None)

    if inlup:
        bnmgy = b*1e9
        tflux = maskgals.exptime*2.0*bnmgy*np.sinh(-np.log(b)-0.4*np.log(10.0)*mag_in)
    else:
        tflux = maskgals.exptime*10.**((mag_in - maskgals.zp[0])/(-2.5))

    noise = err_ratio*np.sqrt(fsky1*maskgals.exptime + tflux)

    if sigma0 > 0.0:
        noise = np.sqrt(noise**2. + ((np.log(10.)/2.5) * sigma0 * tflux)**2.)

    if lnscat is not None:
        noise = np.exp(np.log(noise) + lnscat * rng.normal(size=noise.size))

    if nonoise:
        flux = tflux
    else:
        flux = tflux + noise * rng.standard_normal(mag_in.size)

    if fluxmode:
        mag = flux/maskgals.exptime
        mag_err = noise/maskgals.exptime
    else:
        if b is not None:
            bnmgy = b*1e9

            flux_new = flux/maskgals.exptime
            noise_new = noise/maskgals.exptime

            mag = 2.5*np.log10(1.0/b) - np.arcsinh(0.5*flux_new/bnmgy)/(0.4*np.log(10.0))
            mag_err = 2.5*noise_new/(2.0*bnmgy*np.log(10.0)*np.sqrt(1.0+(0.5*flux_new/bnmgy)**2.0))
        else:
            # Want to suppress warnings here (because we check for infinites below)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mag = maskgals.zp[0]-2.5*np.log10(flux/maskgals.exptime)
                mag_err = (2.5/np.log(10.0))*(noise/flux)

            bad, = np.where(np.isfinite(mag) == False)
            mag[bad] = 99.0
            mag_err[bad] = 99.0

    return mag, mag_err

#####################
## IDL interpol
#####################

def interpol(v, x, xout):
    """
    Port of IDL interpol.py.  Does fast and simple linear interpolation.

    Parameters
    ----------
    v: `np.array`
       Float array of y (dependent) values to interpolate between
    x: `np.array`
       Float array of x (independent) values to interpolate between
    xout: `np.array`
       Float array of x values to compute interpolated values

    Returns
    -------
    yout: `np.array`
       Float array of y output values associated with xout
    """

    m = v.size
    nOut = m

    s = np.clip(np.searchsorted(x, xout) - 1, 0, m - 2)

    diff = v[s + 1] - v[s]

    return (xout - x[s]) * diff / (x[s + 1] - x[s]) + v[s]

#####################
## IDL cic
#####################

def cic(value, posx=None, nx=None, posy=None, ny=None, posz=None, nz=None, average=False, isolated=True):
    """
    Port of idl astronomy utils cic.pro

    Interpolate an irregularly sampled field using Cloud-in-Cells

    Parameters
    ----------
    value: `np.array`
       Array of sample weights (field values)
    posx: `np.array`, optional
       Array of x coordinates of field samples, unit indices: [0, nx)
       Default is None
    nx: `int`, optional
       Range of x values
    posy: `np.array`, optional
       Array of y coordinates of field samples, unit indices: [0, ny)
       Default is None
    ny: `int`, optional
       Range of y values
    posz: `np.array`, optional
       Array of z coordinates of field samples, unit indices: [0, nz)
       Default is None
    nz: `int`, optional
       Range of z values
    average: `bool`, optional
       True if nodes contain field samples.  The value at each grid point
       will be the weighted average of all samples allocated.  If False,
       the value will be the weighted sum of all nodes.  Default is False.
    isolated: `bool`, optional
       True if data is not periodic.  Default is True.

    Returns
    -------
    field: `np.array`
       1d, 2d, or 3d float array of CIC field values.
    """

    # need type checks

    nrsamples = value.size
    dim = 0
    if (posx is not None and nx is not None):
        dim += 1
    if (posy is not None and ny is not None):
        dim += 1
    if (posz is not None and nz is not None):
        dim += 1

    if dim <= 2:
        nz = 1
        if dim == 1:
            ny = 1

    nxny = nx * ny

    # X-direction

    # coordinates of nearest grid point (ngp)
    ngx = posx.astype(np.int32) + 0.5

    # distance from sample to ngp
    dngx = ngx - posx

    # Index of ngp
    kx1 = ngx - 0.5
    # weight of ngp
    wx1 = 1.0 - np.abs(dngx)

    # other side
    left, = np.where(dngx < 0.0)
    # The following is only correct if x(ngp)>posx (ngp to the right).
    kx2 = kx1 - 1
    # Correct points where x(ngp)<posx (ngp to the left).
    if left.size > 0:
        kx2[left] += 2
    wx2 = np.abs(dngx)

    # free memory
    left = None

    bad, = np.where(kx2 == -1)
    if bad.size > 0:
        kx2[bad] = nx - 1
        if isolated:
            wx2[bad] = 0.0
    bad, = np.where(kx2 == nx)
    if bad.size > 0:
        kx2[bad] = 0
        if isolated:
            wx2[bad] = 0.0
    bad = None

    # Y-direction

    if dim >= 2:
        ngy = posy.astype(np.int32) + 0.5

        # distance from sample to ngp
        dngy = ngy - posy

        # Index of ngp
        ky1 = ngy - 0.5
        # weight of ngp
        wy1 = 1.0 - np.abs(dngy)

        # other side
        left, = np.where(dngy < 0.0)
        # The following is only correct if y(ngp)>posy (ngp to the right).
        ky2 = ky1 - 1
        # Correct points where x(ngp)<posx (ngp to the left).
        if left.size > 0:
            ky2[left] += 2
        wy2 = np.abs(dngy)

        # free memory
        left = None

        bad, = np.where(ky2 == -1)
        if bad.size > 0:
            ky2[bad] = ny - 1
            if isolated:
                wy2[bad] = 0.0
        bad, = np.where(ky2 == ny)
        if bad.size > 0:
            ky2[bad] = 0
            if isolated:
                wy2[bad] = 0.0
        bad = None
    else:
        ky1 = 0
        ky2 = 0
        wy1 = 1
        wy2 = 1

    # Z-direction

    if dim == 3:
        ngz = posz.astype(np.int32) + 0.5

        # distance from sample to ngp
        dngz = ngz - posz

        # Index of ngp
        kz1 = ngz - 0.5
        # weight of ngp
        wz1 = 1.0 - np.abs(dngz)

        # other side
        left, = np.where(dngz < 0.0)
        # The following is only correct if z(ngp)>posz (ngp to the right).
        kz2 = kz1 - 1
        # Correct points where z(ngp)<posz (ngp to the left).
        if left.size > 0:
            kz2[left] += 2
        wz2 = np.abs(dngz)

        # free memory
        left = None

        bad, = np.where(kz2 == -1)
        if bad.size > 0:
            kz2[bad] = nz - 1
            if isolated:
                wz2[bad] = 0.0
        bad, = np.where(kz2 == nz)
        if bad.size > 0:
            kz2[bad] = 0
            if isolated:
                wz2[bad] = 0.0
        bad = None
    else:
        kz1 = 0
        kz2 = 0
        wz1 = 1
        wz2 = 1

    # Interpolate samples to grid

    field = np.zeros(nx * ny * nz)

    if average:
        totcicweight = np.zeros_like(field)

    index = (kx1 + ky1*nx + kz1 * nxny).astype(np.int32)
    cicweight = wx1 * wy1 * wz1
    np.add.at(field, index, cicweight * value)
    if average:
        np.add.at(totcicweight, index, cicweight)

    index = (kx2 + ky1 * nx + kz1 * nxny).astype(np.int32)
    cicweight = wx2 * wy1 * wz1
    np.add.at(field, index, cicweight * value)
    if average:
        np.add.at(totcicweight, index, cicweight)

    if dim >= 2:
        index = (kx1 + ky2*nx + kz1*nxny).astype(np.int32)
        cicweight = wx1 * wy2 * wz1
        np.add.at(field, index, cicweight * value)
        if average:
            np.add.at(totcicweight, index, cicweight)
        index = (kx2 + ky2*nx + kz1*nxny).astype(np.int32)
        cicweight = wx2 * wy2 * wz1
        np.add.at(field, index, cicweight * value)
        if average:
            np.add.at(totcicweight, index, cicweight)
        if dim == 3:
            index = (kx1 + ky1 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx1 * wy1 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx2 + ky1 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx2 * wy1 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx1 + ky2 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx1 * wy2 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)
            index = (kx2 + ky2 * nx + kz2 * nxny).astype(np.int32)
            cicweight = wx2 * wy2 * wz2
            np.add.at(field, index, cicweight * value)
            if average:
                np.add.at(totcicweight, index, cicweight)

    index = None

    if average:
        good, = np.where(totcicweight != 0)
        field[good] /= totcicweight[good]

    if dim == 1:
        return field
    elif dim == 2:
        return field.reshape((ny, nx))
    else:
        return field.reshape((nz, ny, nx))

#########################
# MakeNodes
#########################

def make_nodes(zrange, nodesize, maxnode=None):
    """
    Make spline nodes over a given range, mostly but not entirely uniform.

    This will return nodes that fill the space desired, and will make the
    highest redshift node spacing larger to fill the space.  In addition, the
    maxnode maximum useful node can be specified, and this will create one
    final node at the high redshift end.  This is useful when you have a color
    (e.g. u-g) that is only fittable over a restricted redshift range.

    Parameters
    ----------
    zrange: `np.array` or `list`
       2-element float array with redshift range
    nodesize: `float`
       Default (and minimum) node spacing
    maxnode: `float`, optional
       Maximum useful node.  Default is None (do full redshift range)

    Returns
    -------
    nodes: `np.array`
       Float array of redshift nodes.
    """

    if maxnode is None or maxnode < 0.0:
        _maxnode = zrange[1]
    else:
        _maxnode = np.clip(maxnode, 0.0, zrange[1])

    # Start with a simple arange
    nodes = np.arange(zrange[0], _maxnode, nodesize)

    # Should we extend the last bin?
    if ((_maxnode - nodes.max()) > (nodesize / 2. + 0.01)):
        # one more node!
        nodes = np.append(nodes, _maxnode)
    elif not np.allclose(np.max(nodes), _maxnode):
        nodes[-1] = _maxnode

    # and finally check if maxnode was lower
    if _maxnode < zrange[1]:
        nodes = np.append(nodes, zrange[1])

    return nodes

#######################
## Sample from a pdf
#######################

def sample_from_pdf(f, ran, step, nsamp, rng, **kwargs):
    """
    Sample from a PDF described by a function f.

    Parameters
    ----------
    f: `function`
       PDF function that can be called
    ran: `np.arange` or `list`
       Two-element range over which to sample.
    step: `float`
       Step size for interpolation.
    nsamp: `int`
       Number of samples from pdf
    **kwargs: `dict`
       Extra arguments to call f()

    Returns
    -------
    samples: `np.array`
       Float array of samples from PDF.
    """

    x = np.arange(ran[0], ran[1], step)
    pdf = f(x, **kwargs)
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf, dtype=np.float64)
    cdfi = (cdf * x.size).astype(np.int32)

    rand = (rng.uniform(size=nsamp) * x.size).astype(np.int32)

    samples = np.zeros(nsamp)
    for i in range(nsamp):
        test, = np.where(cdfi >= rand[i])
        samples[i] = x[test[0]]

    return samples

# for multiprocessing classes
def _pickle_method(m):
    """
    Method to allow object pickling.

    Parameters
    ----------
    m: `object` or None
       Object pickle?

    Returns
    -------
    attr: Attributes to pickle?
    """
    if m.__self__ is None:
        return getattr, (m.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)


def histoGauss(ax, array):
    """
    Plot a histogram and fit a Gaussian to it.  Modeled after IDL histogauss.pro.

    Parameters
    ----------
    ax: `matplotlib.Axis`
       Plot axis object.  If None, no plot is created, only the fit.
    array: `np.array`
       Float array of values to create a histogram from.

    Returns
    -------
    A: `float`
       Fit normalization A
    mu: `float`
       Fit mean mu
    sigma: `float`
       Fit width sigma
    """

    import scipy.optimize
    import esutil

    q13 = np.percentile(array,[25,75])
    binsize=2*(q13[1] - q13[0])*array.size**(-1./3.)

    hist=esutil.stat.histogram(array,binsize=binsize,more=True)

    p0=[array.size,
        np.median(array),
        np.std(array)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # This fit might throw a warning, which we don't need now.
            # Note that in the future if we use the output from this fit in an
            # automated way we'll need to tell the parent that we didn't converge
            coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, hist['center'],
                                                        hist['hist'], p0=p0)
    except:
        # set to starting values...
        coeff = p0

    hcenter=hist['center']
    hhist=hist['hist']

    rangeLo = coeff[1] - 5*coeff[2]
    rangeHi = coeff[1] + 5*coeff[2]

    lo,=np.where(hcenter < rangeLo)
    ok,=np.where(hcenter > rangeLo)
    hhist[ok[0]] += np.sum(hhist[lo])

    hi,=np.where(hcenter > rangeHi)
    ok,=np.where(hcenter < rangeHi)
    hhist[ok[-1]] += np.sum(hhist[hi])

    if ax is not None:
        ax.plot(hcenter[ok],hhist[ok],'b-',linewidth=3)
        ax.set_xlim(rangeLo,rangeHi)

        xvals=np.linspace(rangeLo,rangeHi,1000)
        yvals=gaussFunction(xvals,*coeff)

        ax.plot(xvals,yvals,'k--',linewidth=3)
        ax.locator_params(axis='x',nbins=6)  # hmmm

    return coeff

##################
## Create a lock
##################

def make_lockfile(lockfile, block=False, maxtry=300, waittime=2):
    """
    Make a lockfile with atomic linking.

    Parameters
    ----------
    lockfile: `str`
       Name of lockfile to create
    block: `bool`, optional
       Block execution until lockfile is created?  Default is False.
    maxtry: `int`, optional
       Maximum number of tries to create lockfile (if blocking).  Default is 300.
    waittime: `float`, optional
       Wait time in seconds between tries (if blocking).  Default is 2.

    Returns
    -------
    locked: `bool`
       True if lockfile created, False if not.
    """

    import os
    import tempfile
    import time

    tf = tempfile.mkstemp(prefix='', dir=os.path.dirname(lockfile))
    tempfilename = tf[1]
    os.close(tf[0])

    if not block:
        # Non-blocking
        try:
            os.link(tempfilename, lockfile)
            os.unlink(tempfilename)
            # We successfully got the lock
            return True
        except:
            os.unlink(tempfilename)
            # We did not get the lock; return anyway
            return False
    else:
        # Blocking ... wait to get the lock up to maxtry

        ctr = 0
        locked = False
        while (ctr < maxtry) and (not locked):
            try:
                os.link(tempfilename, lockfile)
                os.unlink(tempfilename)
                locked = True
            except:
                ctr += 1
                time.sleep(waittime)

        if not locked:
            os.unlink(tempfilename)

        return locked

def read_members(catfile):
    """
    Read members associated to a catalog file.

    Parameters
    ----------
    catfile: `str`
       Filename of a cluster catalog.

    Returns
    -------
    members: `redmapper.GalaxyCatalog`
       Member catalog as a GalaxyCatalog
    """
    import os

    from .galaxy import GalaxyCatalog

    memfile = '%s_members.fit' % (catfile.rstrip('.fit'))

    if not os.path.isfile(memfile):
        raise RuntimeError("Could not find member file %s" % (memfile))

    return GalaxyCatalog.from_fits_file(memfile)


def getMemoryString(location):
    """
    Get a string for memory usage (current and peak) for logging.

    Parameters
    ----------
    location: `str`
       A short string which denotes where in the code the memory was recorded.

    Returns
    -------
    memory_string: `str`
       A string suitable for logging that says the memory usage.
    """

    status = None
    result = {'peak':0, 'rss':0}
    memoryString = ''
    try:
        with open('/proc/self/status') as status:
            for line in status:
                parts = line.split()
                key = parts[0][2:-1].lower()
                if key in result:
                    result[key] = int(parts[1])/1000

            memoryString = 'Memory usage at %s: %d MB current; %d MB peak.' % (
                location, result['rss'], result['peak'])
    except:
        memoryString = 'Could not get process status for memory usage at %s!' % (location)

    return memoryString

#############################
## Cutting healpix maps up...
#############################

def get_hpmask_subpix_indices(submask_nside, submask_hpix, submask_border, nside_mask, hpix):
    """
    """

    nside_cutref = np.clip(submask_nside * 4, 256, nside_mask)

    # Find out which cutref pixels are inside the main pixel
    theta, phi = hpg.pixel_to_angle(nside_cutref, np.arange(hp.nside2npix(nside_cutref)), lonlat=False, nest=False)
    ipring_coarse = hpg.angle_to_pixel(submask_nside, theta, phi, lonlat=False, nest=False)
    inhpix, = np.where(ipring_coarse == submask_hpix)

    # If there is a border, we need to find the boundary pixels
    if submask_border > 0.0:
        boundaries = hpc.boundaries(submask_nside, submask_hpix, step=nside_cutref//submask_nside)
        # These are all the pixels that touch the boundary
        for i in range(boundaries.shape[1]):
            pixint = hpc.query_disc(nside_cutref, boundaries[:, i],
                                    np.radians(submask_border), inclusive=True, fact=8)
            inhpix = np.append(inhpix, pixint)
            # Need to uniqify here because of overlapping pixels
            inhpix = np.unique(inhpix)

    # And now choose just those depthmap pixels that are in the inhpix region
    theta, phi = hpg.pixel_to_angle(nside_mask, hpix, lonlat=False, nest=False)
    ipring = hpg.angle_to_pixel(nside_cutref, theta, phi, lonlat=False, nest=False)

    _, use = esutil.numpy_util.match(inhpix, ipring)

    return use

def get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside):
    """
    Retrieve the coverage pixels that intersect the region, with a border.

    Parameters
    ----------
    subpix_nside: `int`
       Nside for the subregion
    subpix_hpix: `list`
       Pixel numbers for the subregion (ring format).
    subpix_border: `float`
       Border radius to cover outside subpix_hpix
    coverage_nside: `int`
       Nside of the healsparse coverage map
    """

    if len(subpix_hpix) == 0:
        raise RuntimeError("subpix_hpix cannot be an empty list.")
    if subpix_border > 0.0 and len(subpix_hpix) > 1:
        raise NotImplementedError("Cannot read multiple subpixels with a border.")

    # First, we need to know which pixel(s) from nside_coverage are covered by
    # subpix_hpix

    if subpix_nside == 0:
        # Special case for the full sky
        return None
    elif subpix_nside == coverage_nside:
        covpix = hpg.ring_to_nest(subpix_nside, subpix_hpix)
    elif subpix_nside > coverage_nside:
        # what pixels are these contained in?
        theta, phi = hpg.pixel_to_angle(subpix_nside, subpix_hpix, lonlat=False, nest=False)
        covpix = hpg.angle_to_pixel(coverage_nside, theta, phi, lonlat=False, nest=True)
    else:
        # This is subpix_nside < coverage_nside
        # what coverage pixels are contained in subpix_hpix?
        subpix_hpix_nest = hpg.ring_to_nest(subpix_nside, subpix_hpix)
        bit_shift = 2 * int(np.round(np.log(coverage_nside / subpix_nside) / np.log(2)))
        n_pix_per_pix = 2**bit_shift
        covpix = np.zeros(len(subpix_hpix_nest) * n_pix_per_pix, dtype=np.int64)
        for i, hpix in enumerate(subpix_hpix_nest):
            covpix[i * n_pix_per_pix: (i + 1) * n_pix_per_pix] = np.left_shift(hpix, bit_shift) + np.arange(n_pix_per_pix)

    # And now if we have a border...(only if the length is 1)
    if subpix_border > 0.0:
        nside_testing = max([coverage_nside * 4, subpix_nside * 4])
        boundaries = hpc.boundaries(subpix_nside, subpix_hpix[0], step=nside_testing//subpix_nside)

        extrapix = np.zeros(0, dtype=np.int64)

        # These are pixels that touch the boundary
        for i in range(boundaries.shape[1]):
            pixint = hpc.query_disc(nside_testing, boundaries[:, i],
                                    np.radians(subpix_border), inclusive=True, fact=8)
            extrapix = np.append(extrapix, pixint)

        extrapix = np.unique(extrapix)
        theta, phi = hpg.pixel_to_angle(nside_testing, extrapix, lonlat=False, nest=False)
        covpix = np.unique(np.append(covpix, hpg.angle_to_pixel(coverage_nside, theta, phi, lonlat=False)))

    return covpix

