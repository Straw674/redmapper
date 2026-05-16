"""Functions and classes for describing geometry masks in redmapper.

This file contains functions and classes for reading and using geometry masks,
transitioning to a functional style.
"""
import esutil
import fitsio
import numpy as np
import os
from scipy.special import erf
import scipy.integrate
import healsparse

from .catalog import Catalog, Entry
from .utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere, calc_theta_i, apply_errormodels
from .utilities import make_lockfile, sample_from_pdf, chisq_pdf, schechter_pdf, nfw_pdf
from .utilities import get_healsparse_subpix_indices

CURRENT_MASKGAL_VERSION = 7

def read_maskgals(maskgalfile):
    """
    Read the "maskgal" file for monte carlo estimation of coverage.

    Parameters
    ----------
    maskgalfile: `str`
       Filename of maskgal file with monte carlo galaxies

    Returns
    -------
    maskgals_all: `redmapper.Catalog`
        All maskgals read from file
    """
    if not os.path.isfile(maskgalfile):
        raise RuntimeError("Could not find maskgalfile %s.  Please run mask.gen_maskgals(maskgalfile)" % (maskgalfile))

    # Check version
    hdr = fitsio.read_header(maskgalfile, ext=1)
    if (hdr['version'] != CURRENT_MASKGAL_VERSION):
        raise RuntimeError("maskgalfile %s is old version.  Please run mask.gen_maskgals(maskgalfile)" % (maskgalfile))

    return Catalog.from_fits_file(maskgalfile)

def select_maskgals_sample(config, maskgals_all, rng, maskgal_index=None):
    """
    Select a subset of maskgals by sampling.

    Parameters
    ----------
    config: `redmapper.Config`
        Configuration object
    maskgals_all: `redmapper.Catalog`
        All maskgals
    rng: `np.random.RandomState`
        Random state to use
    maskgal_index: `int`, optional
       Pre-selected index to sample from (for reproducibility).
       Default is None (select randomly).

    Returns
    -------
    maskgals: `redmapper.Catalog`
        Subset of maskgals
    maskgal_index: `int`
        The index used for sampling
    """
    if maskgal_index is None:
        maskgal_index = rng.choice(config['maskgal_nsamples'])

    maskgals = maskgals_all[maskgal_index * config['maskgal_ngals']:
                           (maskgal_index + 1) * config['maskgal_ngals']]

    return maskgals, maskgal_index

def gen_maskgals(config, maskgalfile, rng=None):
    """
    Function to generate the maskgal monte carlo galaxies.

    Parameters
    ----------
    config: `redmapper.Config`
        Configuration object
    maskgalfile: `str`
       Name of maskgal file to generate.
    rng: `np.random.RandomState`, optional
        Random state to use.
    """
    if rng is None:
        rng = np.random.RandomState(seed=config['randomseed'])

    minrad = np.clip(np.floor(10.*config['percolation_r0'] * (3./100.)**config['percolation_beta']) / 10., None, 0.5)
    maxrad = np.ceil(10.*config['percolation_r0'] * (300./100.)**config['percolation_beta']) / 10.

    nradbins = np.ceil((maxrad - minrad) / config['maskgal_rad_stepsize']).astype(np.int32) + 1
    radbins = np.arange(nradbins, dtype=np.float32) * config['maskgal_rad_stepsize'] + minrad

    nmag = config['nmag']
    ncol = nmag - 1

    ngals = config['maskgal_ngals'] * config['maskgal_nsamples']

    maskgals = Catalog.zeros(ngals, dtype=[('r', 'f4'),
                                           ('phi', 'f4'),
                                           ('x', 'f4'),
                                           ('y', 'f4'),
                                           ('r_uniform', 'f4'),
                                           ('x_uniform', 'f4'),
                                           ('y_uniform', 'f4'),
                                           ('m', 'f4'),
                                           ('refmag', 'f4'),
                                           ('refmag_obs', 'f4'),
                                           ('refmag_obs_err', 'f4'),
                                           ('chisq', 'f4'),
                                           ('cwt', 'f4'),
                                           ('chisq_pdf', 'f4'),
                                           ('nfw', 'f4'),
                                           ('dzred', 'f4'),
                                           ('zwt', 'f4'),
                                           ('lumwt', 'f4'),
                                           ('lum_pdf', 'f4'),
                                           ('limmag', 'f4'),
                                           ('limmag_dered', 'f4'),
                                           ('exptime', 'f4'),
                                           ('m50', 'f4'),
                                           ('eff', 'f4'),
                                           ('w', 'f4'),
                                           ('theta_r', 'f4', nradbins),
                                           ('mark', bool),
                                           ('radbins', 'f4', nradbins),
                                           ('nin', 'f4', nradbins),
                                           ('nin_orig', 'f4', nradbins),
                                           ('zp', 'f4'),
                                           ('ebv', 'f4'),
                                           ('extinction', 'f4'),
                                           ('nsig', 'f4')])

    maskgals['radbins'] = np.tile(radbins, maskgals.size).reshape(maskgals.size, nradbins)

    # Generate chisq
    maskgals.chisq = sample_from_pdf(chisq_pdf, [0.0, config['chisq_max']],
                                     config['chisq_max'] / 10000.,
                                     maskgals.size, rng, k=ncol)
    # Generate mstar
    maskgals.m = sample_from_pdf(schechter_pdf,
                                 [-2.5*np.log10(10.0),
                                   -2.5*np.log10(config['lval_reference']) + config['maskgal_dmag_extra']],
                                 0.002, maskgals.size, rng,
                                 alpha=config['calib_lumfunc_alpha'], mstar=0.0)
    # Generate nfw(r)
    maskgals.r = sample_from_pdf(nfw_pdf,
                                 [0.001, maxrad],
                                 0.001, maskgals.size, rng, radfactor=True)

    # Generate phi
    maskgals.phi = 2. * np.pi * rng.random(size=maskgals.size)

    # Precompute x/y
    maskgals.x = maskgals.r * np.cos(maskgals.phi)
    maskgals.y = maskgals.r * np.sin(maskgals.phi)

    # And uniform x/y
    maskgals.r_uniform = config['bkg_local_annuli'][1] * np.sqrt(rng.uniform(size=maskgals.size))
    theta_new = rng.uniform(size=maskgals.size)*2*np.pi
    maskgals.x_uniform = maskgals.r_uniform*np.cos(theta_new)
    maskgals.y_uniform = maskgals.r_uniform*np.sin(theta_new)

    # Compute weights to go with these values

    # Chisq weight
    maskgals.cwt = chisq_pdf(maskgals.chisq, ncol)
    maskgals.chisq_pdf = maskgals.cwt

    # Nfw weight
    maskgals.nfw = nfw_pdf(maskgals.r, radfactor=True)

    # luminosity weight

    # We just choose a reference mstar for the normalization code
    mstar = 19.0
    normmag = mstar - 2.5 * np.log10(config['lval_reference'])
    steps = np.arange(10.0, normmag, 0.01)
    f = schechter_pdf(steps, alpha=config['calib_lumfunc_alpha'], mstar=mstar)
    n = scipy.integrate.simpson(y=f, x=steps)
    maskgals.lum_pdf = schechter_pdf(maskgals.m + mstar, mstar=mstar, alpha=config['calib_lumfunc_alpha'])
    maskgals.lumwt = maskgals.lum_pdf / n

    # zred weight
    maskgals.dzred = rng.normal(loc=0.0, scale=config['maskgal_zred_err'], size=maskgals.size)
    maskgals.zwt = (1. / (np.sqrt(2.*np.pi) * config['maskgal_zred_err'])) * np.exp(-(maskgals.dzred**2.) / (2.*config['maskgal_zred_err']**2.))

    # And we need the radial function for each set of samples
    for j in range(config['maskgal_nsamples']):
        indices = np.arange(j * config['maskgal_ngals'],
                            (j + 1) * config['maskgal_ngals'])

        # Radial function
        for i, rad in enumerate(radbins):
            inside, = np.where((maskgals.r[indices] <= rad) &
                               (maskgals.m[indices] < -2.5*np.log10(config['lval_reference'])))
            maskgals.nin_orig[indices, i] = inside.size

            if config['rsig'] <= 0.0:
                theta_r = np.ones(config['maskgal_ngals'])
            else:
                theta_r = 0.5 + 0.5*erf((rad - maskgals.r[indices]) / (np.sqrt(2.)*config['rsig']))
            maskgals.theta_r[indices, i] = theta_r

            inside2, = np.where(maskgals.m[indices] < -2.5*np.log10(config['lval_reference']))
            maskgals.nin[indices, i] = np.sum(theta_r[inside2], dtype=np.float64)

    if config['more_qa_plots']:
        import matplotlib.pyplot as plt

        if not os.path.exists(config['plotpath']):
            os.makedirs(config['plotpath'])

        # Plot spatial distribution
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        # Plot a subset
        sub = np.random.choice(maskgals.size, size=min(maskgals.size, 10000), replace=False)
        ax.plot(maskgals.x[sub], maskgals.y[sub], 'k.', markersize=1)
        ax.set_xlabel('X (Mpc)')
        ax.set_ylabel('Y (Mpc)')
        ax.set_title('Maskgals Spatial Distribution')
        ax.set_aspect('equal')
        fig.savefig(os.path.join(config['plotpath'], 'maskgals_spatial.png'))
        plt.close(fig)

        # Plot radial distribution
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(maskgals.r, bins=100, histtype='step')
        ax.set_xlabel('R (Mpc)')
        ax.set_title('Maskgals Radial Distribution')
        fig.savefig(os.path.join(config['plotpath'], 'maskgals_radial.png'))
        plt.close(fig)

        # Plot magnitude distribution
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(maskgals.m, bins=100, histtype='step')
        ax.set_xlabel('m - mstar')
        ax.set_title('Maskgals Magnitude Distribution')
        fig.savefig(os.path.join(config['plotpath'], 'maskgals_mag.png'))
        plt.close(fig)

        # Plot chisq distribution
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(maskgals.chisq, bins=100, histtype='step')
        ax.set_xlabel('Chisq')
        ax.set_title('Maskgals Chisq Distribution')
        fig.savefig(os.path.join(config['plotpath'], 'maskgals_chisq.png'))
        plt.close(fig)

    # And save it

    hdr = fitsio.FITSHDR()
    hdr['version'] = CURRENT_MASKGAL_VERSION
    hdr['r0'] = config['percolation_r0']
    hdr['beta'] = config['percolation_beta']
    hdr['stepsize'] = config['maskgal_rad_stepsize']
    hdr['nmag'] = config['nmag']
    hdr['ngals'] = config['maskgal_ngals']
    hdr['chisqmax'] = config['chisq_max']
    hdr['lvalref'] = config['lval_reference']
    hdr['extra'] = config['maskgal_dmag_extra']
    hdr['alpha'] = config['calib_lumfunc_alpha']
    hdr['rsig'] = config['rsig']
    hdr['zrederr'] = config['maskgal_zred_err']

    maskgals.to_fits_file(maskgalfile, clobber=True, header=hdr)

def read_mask(config, covpixels=None):
    """
    Read mask data from config.

    Parameters
    ----------
    config: `redmapper.Config`
        Configuration object
    covpixels: `np.array`, optional
        Coverage pixels to read. Default is None (read all).

    Returns
    -------
    mask_data: `healsparse.HealSparseMap` or None
        Mask data (None if mask_mode == 0)
    """
    if config['mask_mode'] == 0:
        return None
    
    if config['mask_mode'] == 3:
        maskfile = config['maskfile']
        hdr = fitsio.read_header(maskfile, ext=1)
        if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
            raise RuntimeError("Need to specify mask in healsparse format.  See redmapper_convert_mask_to_healsparse.py")

        if covpixels is None and len(config['hpix']) > 0:
            cov_hdr = fitsio.read_header(maskfile, ext='COV')
            nside_coverage = cov_hdr['NSIDE']
            covpixels = get_healsparse_subpix_indices(config['nside'], config['hpix'],
                                                      config['border'], nside_coverage)
        
        return healsparse.HealSparseMap.read(maskfile, pixels=covpixels)
    
    raise ValueError("Unsupported mask_mode: %d" % config['mask_mode'])

def get_mask_values(mask_data, ra, dec, rng=None, config=None):
    """
    Compute the geometric mask values at a list of positions.

    Parameters
    ----------
    mask_data: `healsparse.HealSparseMap` or None
        Mask data
    ra: `np.array`
       Float array of right ascensions
    dec: `np.array`
       Float array of declinations
    rng: `np.random.RandomState`, optional
        Random state for stochastic mask application. Default is None.
    config: `redmapper.Config`, optional
        Configuration object (used for random seed if rng is None). Default is None.

    Returns
    -------
    maskvals: `np.array`
       Bool array of True ("in the footprint") for each ra/dec.
    """
    _ra = np.atleast_1d(ra)
    _dec = np.atleast_1d(dec)

    if (_ra.size != _dec.size):
        raise ValueError("ra, dec must be same length")

    if mask_data is None:
        return np.ones(_ra.size, dtype=bool)

    gd, = np.where(np.abs(_dec) < 90.0)
    fracgood = np.zeros(_ra.size, dtype=np.float64)
    fracgood[gd] = mask_data.get_values_pos(_ra[gd], _dec[gd], lonlat=True)

    if rng is None:
        if config is not None:
            rng = np.random.RandomState(seed=config['randomseed'])
        else:
            rng = np.random.RandomState()

    radmask = np.zeros(_ra.size, dtype=bool)
    radmask[np.where(fracgood > rng.rand(_ra.size))] = True
    return radmask

def compute_maskgals_mark(mask_data, cluster, maskgals, rng=None, config=None):
    """
    Compute mark values for maskgals for a given cluster.

    Parameters
    ----------
    mask_data: `healsparse.HealSparseMap` or None
        Mask data
    cluster: `redmapper.Cluster`
        Cluster object
    maskgals: `redmapper.Catalog`
        Maskgals catalog
    rng: `np.random.RandomState`, optional
        Random state
    config: `redmapper.Config`, optional
        Configuration object

    Returns
    -------
    mark: `np.array`
        Boolean array of mark values
    """
    ras = cluster.ra + maskgals.x/(cluster.mpc_scale)/np.cos(np.radians(cluster.dec))
    decs = cluster.dec + maskgals.y/(cluster.mpc_scale)
    return get_mask_values(mask_data, ras, decs, rng=rng, config=config)

def calc_maskcorr(maskgals, mstar, maxmag, limmag, rng):
    """
    Calculate mask correction cpars, a third-order polynomial which describes the
    mask fraction of a cluster as a function of radius.

    Parameters
    ----------
    maskgals: `redmapper.Catalog`
        Maskgals catalog
    mstar: `float`
       mstar (mag) at cluster redshift
    maxmag: `float`
       maximum magnitude for use in luminosity function filter
    limmag: `float`
       Survey or local limiting magnitude
    rng: `np.random.RandomState`
        Random state

    Returns
    -------
    cpars: `np.array`
       Third-order polynomial parameters describing maskfrac as function of radius
    """
    mag_in = maskgals.m + mstar
    maskgals.refmag = mag_in

    if maskgals.limmag[0] > 0.0:
        mag, mag_err = apply_errormodels(maskgals, mag_in, rng=rng)

        maskgals.refmag_obs = mag
        maskgals.refmag_obs_err = mag_err
    else:
        mag = mag_in
        mag_err = 0*mag_in
        raise ValueError('Survey limiting magnitude <= 0!')

    if (maskgals.w[0] < 0) or (maskgals.w[0] == 0 and
            np.amax(maskgals.m50) == 0):
        theta_i = calc_theta_i(mag, mag_err, maxmag, limmag)
    elif (maskgals.w[0] == 0):
        theta_i = calc_theta_i(mag, mag_err, maxmag, maskgals.m50)
    else:
        raise Exception('Unsupported mode!')

    p_det = theta_i*maskgals.mark
    c = 1 - np.dot(p_det, maskgals.theta_r) / maskgals.nin[0]

    cpars = np.polyfit(maskgals.radbins[0], c, 3)

    return cpars

def convert_maskfile_to_healsparse(maskfile, healsparsefile, nsideCoverage, clobber=False):
    """
    Convert an old maskfile to a new healsparsefile

    Parameters
    ----------
    maskfile: `str`
       Input mask file
    healsparsefile: `str`
       Output healsparse file
    nsideCoverage: `int`
       Nside for sparse coverage map
    clobber: `bool`, optional
       Clobber existing healsparse file?  Default is false.
    """
    old_mask, old_hdr = fitsio.read(maskfile, ext=1, header=True, lower=True)

    nside = old_hdr['nside']

    sparseMap = healsparse.HealSparseMap.make_empty(nsideCoverage, nside, old_mask['fracgood'].dtype)
    sparseMap.update_values_pix(old_mask['hpix'], old_mask['fracgood'], nest=old_hdr['nest'])

    sparseMap.write(healsparsefile, clobber=clobber)

def get_mask(config, include_maskgals=True, rng=None):
    """
    Convenience function to look at a config file and load the appropriate type of mask.

    Parameters
    ----------
    config: `redmapper.Config`
       Configuration object
    include_maskgals : `bool`, optional
        Include maskgals in the mask?
    rng : `np.random.RandomState`, optional
        Use this RandomState?

    Returns
    -------
    mask: `dict`
        Dictionary containing mask data and metadata
    """
    if rng is None:
        rng = np.random.RandomState(seed=config['randomseed'])

    mask_data = read_mask(config)
    maskgals_all = None
    if include_maskgals:
        maskgals_all = read_maskgals(config['maskgalfile'])

    mask = {
        'mask_data': mask_data,
        'maskgals_all': maskgals_all,
        'maskgals': None,
        'maskgal_index': -1,
        'rng': rng,
        'config': config,
        'nside': mask_data.nside_sparse if mask_data is not None else -1
    }

    return mask
