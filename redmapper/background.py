"""Galaxy background classes for redmapper.

This file contains classes to describe the b(x) background terms for computing
richness and other redmapper likelihoods.
"""
import fitsio
import numpy as np
import hpgeom as hpg
import time
import copy
import os
import esutil
import multiprocessing

from .logger import logger

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .catalog import Entry
from .galaxy import GalaxyCatalog
from .redsequence import read_redsequence, redsequence_zindex, redsequence_mstar, compute_redsequence_chisq
from . import depthmap
from .utilities import interpol, cic
from .utilities import _pickle_method

copyreg.pickle(types.MethodType, _pickle_method)

def read_background(filename):
    """
    Read background data from a FITS file and interpolate to a standard fine grid.

    Parameters
    ----------
    filename: `str`
        Background filename

    Returns
    -------
    background_data: `dict`
        Dictionary containing background data and metadata
    """
    obkg = Entry.from_fits_file(filename, ext='CHISQBKG')

    # Set the bin size in redshift, chisq and refmag spaces
    zbinsize = 0.001
    chisqbinsize = 0.5
    refmagbinsize = 0.01

    # Create the refmag bins
    refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], refmagbinsize)
    nrefmagbins = refmagbins.size

    # Create the chisq bins
    nchisqbins = obkg.chisqbins.size
    nlnchisqbins = obkg.lnchisqbins.size

    # Read out the number of redshift bins from the object background
    nzbins = obkg.zbins.size

    # Set up some arrays to populate
    sigma_g_new = np.zeros((nrefmagbins, nchisqbins, nzbins))
    sigma_lng_new = np.zeros((nrefmagbins, nlnchisqbins, nzbins))

    # Do linear interpolation to get the sigma_g value
    # between the raw background points.
    # If any values are less than 0 then turn them into 0.
    for i in range(nzbins):
        for j in range(nchisqbins):
            sigma_g_new[:, j, i] = np.interp(refmagbins, obkg.refmagbins, obkg.sigma_g[:, j, i])
            sigma_g_new[:, j, i] = np.where(sigma_g_new[:, j, i] < 0, 0, sigma_g_new[:, j, i])
        for j in range(nlnchisqbins):
            sigma_lng_new[:, j, i] = np.interp(refmagbins, obkg.refmagbins, obkg.sigma_lng[:, j, i])
            sigma_lng_new[:, j, i] = np.where(sigma_lng_new[:, j, i] < 0, 0, sigma_lng_new[:, j, i])

    sigma_g = sigma_g_new
    sigma_lng = sigma_lng_new

    chisqbins = np.arange(obkg.chisqrange[0], obkg.chisqrange[1], chisqbinsize)
    nchisqbins_new = chisqbins.size

    sigma_g_new = np.zeros((nrefmagbins, nchisqbins_new, nzbins))

    # Now do the interpolation in chisq space
    for i in range(nzbins):
        for j in range(nrefmagbins):
            sigma_g_new[j, :, i] = np.interp(chisqbins, obkg.chisqbins, sigma_g[j, :, i])
            sigma_g_new[j, :, i] = np.where(sigma_g_new[j, :, i] < 0, 0, sigma_g_new[j, :, i])

    sigma_g = sigma_g_new

    zbins = np.arange(obkg.zrange[0], obkg.zrange[1], zbinsize)
    nzbins_new = zbins.size

    sigma_g_final = np.zeros((nrefmagbins, nchisqbins_new, nzbins_new))
    sigma_lng_final = np.zeros((nrefmagbins, nlnchisqbins, nzbins_new))

    # Now do the interpolation in redshift space
    for i in range(nchisqbins_new):
        for j in range(nrefmagbins):
            sigma_g_final[j, i, :] = np.interp(zbins, obkg.zbins, sigma_g[j, i, :])
            sigma_g_final[j, i, :] = np.where(sigma_g_final[j, i, :] < 0, 0, sigma_g_final[j, i, :])

    for i in range(nlnchisqbins):
        for j in range(nrefmagbins):
            sigma_lng_final[j, i, :] = np.interp(zbins, obkg.zbins, sigma_lng[j, i, :])
            sigma_lng_final[j, i, :] = np.where(sigma_lng_final[j, i, :] < 0, 0, sigma_lng_final[j, i, :])

    n_new = np.zeros((nrefmagbins, nzbins_new))
    for i in range(nzbins_new):
        n_new[:, i] = np.sum(sigma_g_final[:, :, i], axis=1, dtype=np.float64) * chisqbinsize

    background_data = {
        'refmagbins': refmagbins,
        'chisqbins': chisqbins,
        'lnchisqbins': obkg.lnchisqbins,
        'zbins': zbins,
        'sigma_g': sigma_g_final,
        'sigma_lng': sigma_lng_final,
        'n': n_new,
        'zbinsize': zbinsize,
        'chisqbinsize': chisqbinsize,
        'refmagbinsize': refmagbinsize
    }

    return background_data

def compute_background(background_data, z, chisq, refmag, allow0=False):
    """
    Look up the Sigma_g(z, chisq, refmag) background quantity.

    Parameters
    ----------
    background_data: `dict`
        Background data dictionary from read_background
    z: `np.array`
       redshifts of galaxies
    chisq: `np.array`
       chi-squared values of galaxies
    refmag: `np.array`
       reference magnitudes of galaxies
    allow0: `bool`, optional
       Flag to allow Sigma_g(x) to be zero. Otherwise will set to infinity
       where there is no data. Default is False.

    Returns
    -------
    sigma_g: `np.array`
       Sigma_g(x) for input values
    """
    zbins = background_data['zbins']
    chisqbins = background_data['chisqbins']
    refmagbins = background_data['refmagbins']
    sigma_g = background_data['sigma_g']
    chisqbinsize = background_data['chisqbinsize']
    refmagbinsize = background_data['refmagbinsize']

    zmin = zbins[0]
    chisqindex = np.searchsorted(chisqbins, chisq) - 1
    refmagindex = np.searchsorted(refmagbins, refmag) - 1
    
    ind = np.clip(np.round((z-zmin)/(zbins[1]-zmin)), 0, zbins.size-1).astype(np.int32)

    badchisq, = np.where((chisq < chisqbins[0]) |
                         (chisq > (chisqbins[-1] + chisqbinsize)))
    badrefmag, = np.where((refmag <= refmagbins[0]) |
                          (refmag > (refmagbins[-1] + refmagbinsize)))

    chisqindex[badchisq] = 0
    refmagindex[badrefmag] = 0

    zindex = np.full_like(chisqindex, ind)
    lookup_vals = sigma_g[refmagindex, chisqindex, zindex]
    lookup_vals[badchisq] = np.inf
    lookup_vals[badrefmag] = np.inf

    if not allow0:
        lookup_vals[lookup_vals == 0.0] = np.inf

    return lookup_vals

def read_zred_background(filename):
    """
    Read zred background data from a FITS file and interpolate to standard grid.

    Parameters
    ----------
    filename: `str`
        Zred background filename

    Returns
    -------
    zred_background_data: `dict`
        Dictionary containing zred background data and metadata
    """
    obkg = Entry.from_fits_file(filename, ext='ZREDBKG')

    refmagbinsize = 0.01
    zredbinsize = 0.001

    # Create the refmag bins
    refmagbins = np.arange(obkg.refmagrange[0], obkg.refmagrange[1], refmagbinsize)
    nrefmagbins = refmagbins.size

    # Leave the zred bins the same
    nzredbins = obkg.zredbins.size

    # Set up arrays to populate
    sigma_g_new = np.zeros((nrefmagbins, nzredbins))

    floor = np.min(obkg.sigma_g)

    for i in range(nzredbins):
        sigma_g_new[:, i] = np.clip(interpol(obkg.sigma_g[:, i], obkg.refmagbins, refmagbins), floor, None)

    sigma_g = sigma_g_new

    # And update zred
    zredbins = np.arange(obkg.zredrange[0], obkg.zredrange[1], zredbinsize)
    nzredbins = zredbins.size

    sigma_g_new = np.zeros((nrefmagbins, nzredbins))

    for i in range(nrefmagbins):
        sigma_g_new[i, :] = np.clip(interpol(sigma_g[i, :], obkg.zredbins, zredbins), floor, None)

    zred_background_data = {
        'zredbins': zredbins,
        'zredrange': obkg.zredrange,
        'refmagbins': refmagbins,
        'refmagrange': obkg.refmagrange,
        'sigma_g': sigma_g_new,
        'refmagbinsize': refmagbinsize,
        'zredbinsize': zredbinsize
    }

    return zred_background_data

def compute_zred_background(zred_background_data, zred, refmag):
    """
    Look up the Sigma_g(zred, refmag) background quantity for centering calculations.

    Parameters
    ----------
    zred_background_data: `dict`
        Zred background data dictionary from read_zred_background
    zred: `np.array`
       zred redshifts of galaxies
    refmag: `np.array`
       reference magnitudes of galaxies

    Returns
    -------
    sigma_g: `np.array`
       Sigma_g(x) for input values
    """
    zredbins = zred_background_data['zredbins']
    refmagbins = zred_background_data['refmagbins']
    sigma_g = zred_background_data['sigma_g']

    zredindex = np.searchsorted(zredbins, zred) - 1
    refmagindex = np.searchsorted(refmagbins, refmag) - 1

    badzred, = np.where((zredindex < 0) |
                        (zredindex >= zredbins.size))
    zredindex[badzred] = 0
    badrefmag, = np.where((refmagindex < 0) |
                          (refmagindex >= refmagbins.size))
    refmagindex[badrefmag] = 0

    lookup_vals = sigma_g[refmagindex, zredindex]

    lookup_vals[badzred] = np.inf
    lookup_vals[badrefmag] = np.inf

    return lookup_vals





def _make_qa_plots_background(config, sigma_g, sigma_lng, refmagrange, chisqrange, nzbins, zbins):
    """
    Generate QA plots for background calibration.

    Parameters
    ----------
    config: `redmapper.Configuration`
    sigma_g: `np.array`
        3D array of sigma_g(refmag, chisq, z)
    sigma_lng: `np.array`
        3D array of sigma_lng(refmag, lnchisq, z)
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not hasattr(config, 'plotpath') or config.plotpath is None:
        logger.warning("config.plotpath not set, skipping QA plots")
        return

    os.makedirs(config.plotpath, exist_ok=True)

    # Plot 1: sigma_g integrated over chisq, shown as 2D map in (refmag, z) plane
    fig, ax = plt.subplots(figsize=(12, 8))
    # Integrate over chisq axis (axis=1)
    sigma_refmag_z = np.sum(sigma_g, axis=1) * config.bkg_chisqbinsize
    sigma_plot = np.log10(sigma_refmag_z + 1e-10)
    im = ax.imshow(sigma_plot, origin='lower', aspect='auto',
                  extent=[refmagrange[0], refmagrange[1],
                          config.zrange[0], config.zrange[1]],
                  cmap='Blues')
    ax.set_xlabel('refmag')
    ax.set_ylabel('Redshift')
    ax.set_title(r'$\log_{10}(\Sigma_g)$ integrated over $\chi^2$')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}(\Sigma_g)$ [deg$^{-2}$]')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'bkg_sigma_g_refmag_z.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: bkg_sigma_g_refmag_z.png")

    # Plot 2: 2D maps of sigma_g(refmag, chisq) at different z (keep original)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    z_indices_2d = np.linspace(0, nzbins - 1, 6, dtype=int)

    for idx, (ax, zi) in enumerate(zip(axes.flat, z_indices_2d)):
        sigma_plot = np.log10(sigma_g[:, :, zi] + 1e-10)
        im = ax.imshow(sigma_plot.T, origin='lower', aspect='auto', 
                      extent=[refmagrange[0], refmagrange[1],
                             chisqrange[0], chisqrange[1]],
                      cmap='Reds')
        ax.set_xlabel('refmag')
        ax.set_ylabel(r'$\chi^2$')
        ax.set_title(f'z = {zbins[zi]:.3f}')
        plt.colorbar(im, ax=ax, label=r'$\log_{10}(\Sigma_g)$')

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'bkg_sigma_g_2d.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: bkg_sigma_g_2d.png")

    # Plot 3: chisq distribution at different z (all curves in one plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    nchisqbins = sigma_g.shape[1]
    chisqbins = np.arange(nchisqbins) * config.bkg_chisqbinsize + chisqrange[0]
    n_z_samples = 8
    z_indices = np.linspace(0, nzbins - 1, n_z_samples, dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_z_samples))
    
    for i, zi in enumerate(z_indices):
        # Integrate over refmag
        sigma_chisq = np.sum(sigma_g[:, :, zi], axis=0) * config.bkg_refmagbinsize
        ax.plot(chisqbins, sigma_chisq, color=colors[i], 
               linewidth=1.5, label=f'z={zbins[zi]:.2f}')
    
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel(r'$\Sigma_g$ [integrated over refmag, deg$^{-2}$]')
    ax.set_title(r'$\chi^2$ Distribution at Different Redshifts')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-5)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'bkg_chisq_distribution.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: bkg_chisq_distribution.png")

    # Plot 4: Total background density vs z
    fig, ax = plt.subplots(figsize=(10, 6))
    n_total = np.sum(sigma_g, axis=(0, 1)) * config.bkg_refmagbinsize * config.bkg_chisqbinsize
    ax.step(zbins, n_total, 'k-', linewidth=2, where='mid')
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Total $\Sigma_g$ [deg$^{-2}$]')
    ax.set_title('Total Background Density vs Redshift')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'bkg_total_vs_z.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: bkg_total_vs_z.png")

def _background_worker(worker_args):
    """
    Internal worker method for multiprocessing.

    Parameters
    ----------
    worker_args: `tuple`
       Tuple of (zbinmark, p)
    """
    zbinmark, p = worker_args
    
    config = p['config']
    zbins = p['zbins']
    refmagrange = p['refmagrange']
    nrefmagbins = p['nrefmagbins']
    refmagbins = p['refmagbins']
    chisqrange = p['chisqrange']
    nchisqbins = p['nchisqbins']
    chisqbins = p['chisqbins']
    lnchisqrange = p['lnchisqrange']
    nlnchisqbins = p['nlnchisqbins']
    lnchisqbinsize = p['lnchisqbinsize']
    lnchisqbins = p['lnchisqbins']
    areas = p['areas']
    deepmode = p['deepmode']
    natatime = p['natatime']

    starttime = time.time()

    zbins_use = zbins[zbinmark]
    zrange_use = np.array([zbins_use[0], zbins_use[-1] + config.bkg_zbinsize])

    # We need to load in the red sequence structure -- just in the specific redshift range
    zredstr = read_redsequence(config.parfile, zrange=zrange_use)

    zredstrbinsize = zredstr['z'][1] - zredstr['z'][0]
    zpos = np.searchsorted(zredstr['z'], zbins_use)

    # How many galaxies total?
    if config.galfile_pixelized:
        master = Entry.from_fits_file(config.galfile)

        if len(config.hpix) > 0:
            # We need to take a sub-region
            theta, phi = hpg.pixel_to_angle(master.nside, master.hpix, lonlat=False, nest=False)
            ipring_big = hpg.angle_to_pixel(config.nside, theta, phi, lonlat=False, nest=False)

            _, subreg_indices = esutil.numpy_util.match(config.hpix, ipring_big)
            subreg_indices = np.unique(subreg_indices)
        else:
            subreg_indices = np.arange(master.hpix.size)

        ngal = np.sum(master.ngals[subreg_indices])
        npix = subreg_indices.size
    else:
        hdr = fitsio.read_header(config.galfile, ext=1)

        ngal = hdr['NAXIS2']
        npix = 0

    nmag = config.nmag
    ncol = nmag - 1

    # default values are all guaranteed to be out of range
    chisqs = np.zeros((ngal, zbins_use.size), dtype=np.float32) + np.exp(np.max(lnchisqbins)) + 100.0
    refmags = np.zeros(ngal, dtype=np.float32)

    if (deepmode):
        zlimmag = np.atleast_1d(redsequence_mstar(zredstr, zbins_use + config.bkg_zbinsize) - 2.5 * np.log10(0.01))
    else:
        zlimmag = np.atleast_1d(redsequence_mstar(zredstr, zbins_use + config.bkg_zbinsize) - 2.5 * np.log10(0.1))

    bad, = np.where(zlimmag >= config.limmag_catalog)
    zlimmag[bad] = config.limmag_catalog - 0.01
    zlimmagpos = np.clip(((zlimmag - refmagrange[0]) * nrefmagbins / (refmagrange[1] - refmagrange[0])).astype(np.int32), 0, nrefmagbins - 1)
    zlimmag = refmagbins[zlimmagpos] + config.bkg_refmagbinsize

    zbinmid = np.median(np.arange(zredstr['z'].size - 1))

    # And the main loop
    ctr = 0
    p_idx = 0
    # This covers both loops
    while ((ctr < ngal) and (p_idx < npix)):
        # Read in a section of the galaxies, or the pixel
        if not config.galfile_pixelized:
            lo = ctr
            hi = np.clip(ctr + natatime, None, ngal)

            gals = GalaxyCatalog.from_fits_file(config.galfile, rows=np.arange(lo, hi))
            ctr = hi + 1
        else:
            if master.ngals[subreg_indices[p_idx]] == 0:
                p_idx += 1
                continue

            gals = GalaxyCatalog.from_galfile(config.galfile, nside=master.nside,
                                              hpix=master.hpix[subreg_indices[p_idx]], border=0.0)

            lo = ctr
            hi = ctr + gals.size

            ctr += master.ngals[subreg_indices[p_idx]]
            p_idx += 1

        inds = np.arange(lo, hi)

        refmags[inds] = gals.refmag

        for i, zbin in enumerate(zbins_use):
            use, = np.where((gals.refmag > refmagrange[0]) &
                            (gals.refmag < zlimmag[i]))

            if (use.size > 0):
                # Compute chisq at the redshift zbin
                chisqs[inds[use], i] = compute_redsequence_chisq(zredstr, gals[use], zbin)

    binsizes = config.bkg_refmagbinsize  * config.bkg_chisqbinsize
    lnbinsizes = config.bkg_refmagbinsize * lnchisqbinsize

    sigma_g_sub = np.zeros((nrefmagbins, nchisqbins, zbins_use.size))
    sigma_lng_sub = np.zeros((nrefmagbins, nlnchisqbins, zbins_use.size))

    for i, zbin in enumerate(zbins_use):
        use, = np.where((chisqs[:, i] >= chisqrange[0]) &
                        (chisqs[:, i] < chisqrange[1]) &
                        (refmags >= refmagrange[0]) &
                        (refmags < refmagrange[1]))
        chisqpos = (chisqs[use, i] - chisqrange[0]) * nchisqbins / (chisqrange[1] - chisqrange[0])
        refmagpos = (refmags[use] - refmagrange[0]) * nrefmagbins / (refmagrange[1] - refmagrange[0])

        value = np.ones(use.size)

        field = cic(value, chisqpos, nchisqbins, refmagpos, nrefmagbins, isolated=True)
        for j in range(nchisqbins):
            sigma_g_sub[:, j, i] = field[:, j] / (areas * binsizes)

        lnchisqs = np.log(chisqs[:, i])

        use, = np.where((lnchisqs >= lnchisqrange[0]) &
                        (lnchisqs < lnchisqrange[1]) &
                        (refmags >= refmagrange[0]) &
                        (refmags < refmagrange[1]))
        lnchisqpos = (lnchisqs[use] - lnchisqrange[0]) * nlnchisqbins / (lnchisqrange[1] - lnchisqrange[0])
        refmagpos = (refmags[use] - refmagrange[0]) * nrefmagbins / (refmagrange[1] - refmagrange[0])

        value = np.ones(use.size)

        field2 = cic(value, lnchisqpos, nlnchisqbins, refmagpos, nrefmagbins, isolated=True)

        for j in range(nlnchisqbins):
            sigma_lng_sub[:, j, i] = field2[:, j] / (areas * lnbinsizes)

    logger.info("Finished %.2f < z < %.2f in %.1f seconds" % (zbins_use[0], zbins_use[-1],
                                                                          time.time() - starttime))

    return (zbinmark, sigma_g_sub, sigma_lng_sub)


def generate_background(config, clobber=False, natatime=100000, deepmode=False):
    """
    Generate the galaxy background using multiprocessing.  The number of
    cores used is specified in config.calib_nproc, and the output
    filename is specified in config.bkgfile.

    Parameters
    ----------
    config: `redmapper.Configuration`
    clobber: `bool`, optional
       Overwrite any existing config.bkgfile file.  Default is False.
    natatime: `int`, optional
       Number of galaxies to read at a time.  Default is 100000.
    deepmode: `bool`, optional
       Run background to full depth of survey (rather than Lstar richness limit).
       Default is False.
    """

    if not clobber:
        if os.path.isfile(config.bkgfile):
            with fitsio.FITS(config.bkgfile) as fits:
                if 'CHISQBKG' in [ext.get_extname() for ext in fits[1: ]]:
                    logger.info("CHISQBKG already in %s and clobber is False" % (config.bkgfile))
                    return

    # get the ranges
    refmagrange = np.array([12.0, config.limmag_catalog])
    nrefmagbins = np.ceil((refmagrange[1] - refmagrange[0]) / config.bkg_refmagbinsize).astype(np.int32)
    refmagbins = np.arange(nrefmagbins) * config.bkg_refmagbinsize + refmagrange[0]

    chisqrange = np.array([0.0, config.chisq_max])
    nchisqbins = np.ceil((chisqrange[1] - chisqrange[0]) / config.bkg_chisqbinsize).astype(np.int32)
    chisqbins = np.arange(nchisqbins) * config.bkg_chisqbinsize + chisqrange[0]

    lnchisqbinsize = 0.2
    lnchisqrange = np.array([-2.0, 6.0])
    nlnchisqbins = np.ceil((lnchisqrange[1] - lnchisqrange[0]) / lnchisqbinsize).astype(np.int32)
    lnchisqbins = np.arange(nlnchisqbins) * lnchisqbinsize + lnchisqrange[0]

    nzbins = np.ceil((config.zrange[1] - config.zrange[0]) / config.bkg_zbinsize).astype(np.int32)
    zbins = np.arange(nzbins) * config.bkg_zbinsize + config.zrange[0]

    # this is the background hist
    sigma_g = np.zeros((nrefmagbins, nchisqbins, nzbins))
    sigma_lng = np.zeros((nrefmagbins, nlnchisqbins, nzbins))

    # We need the areas from the depth map
    if config.depthfile is not None:
        depth_data = depthmap.read_depth_map(config)
        areas = depthmap.compute_areas(depth_data, refmagbins)
    else:
        areas = np.zeros(refmagbins.size) + config.area

    # Split into bins for parallel running
    logrange = np.log(np.array([config.zrange[0] - 0.001, config.zrange[1] + 0.001]))
    logbinsize = (logrange[1] - logrange[0]) / config.calib_nproc
    zedges = (np.exp(logrange[0]) + np.exp(logrange[1])) - np.exp(logrange[0] + np.arange(config.calib_nproc + 1) * logbinsize)

    config_safe = config.copy()
    config_safe.cosmo = None

    worker_list = []
    for i in range(config.calib_nproc):
        ubins, = np.where((zbins < zedges[i]) & (zbins > zedges[i + 1]))
        gd, = np.where(ubins < zbins.size)

        # If we have more processes than bins, some of these will be empty
        # and this prevents us from adding them to the list
        if gd.size == 0:
            continue

        ubins = ubins[gd]

        zbinmark = np.zeros(zbins.size, dtype=bool)
        zbinmark[ubins] = True
        
        p = {
            'config': config_safe, 'zbins': zbins, 'refmagrange': refmagrange, 'nrefmagbins': nrefmagbins,
            'refmagbins': refmagbins, 'chisqrange': chisqrange, 'nchisqbins': nchisqbins,
            'chisqbins': chisqbins, 'lnchisqrange': lnchisqrange, 'nlnchisqbins': nlnchisqbins,
            'lnchisqbinsize': lnchisqbinsize, 'lnchisqbins': lnchisqbins, 'areas': areas,
            'deepmode': deepmode, 'natatime': natatime
        }

        worker_list.append((zbinmark, p))

    mp_ctx = multiprocessing.get_context("fork")
    pool = mp_ctx.Pool(processes=config.calib_nproc)
    retvals = pool.map(_background_worker, worker_list, chunksize=1)
    pool.close()
    pool.join()

    # And store the results
    for zbinmark, sigma_g_sub, sigma_lng_sub in retvals:
        sigma_g[:, :, zbinmark] = sigma_g_sub
        sigma_lng[:, :, zbinmark] = sigma_lng_sub

    # Generate QA plots if requested
    if hasattr(config, 'more_qa_plots') and config.more_qa_plots:
        _make_qa_plots_background(config, sigma_g, sigma_lng, refmagrange, chisqrange, nzbins, zbins)

    # And save them
    dtype = [('zbins', 'f4', zbins.size),
             ('zrange', 'f4', 2),
             ('zbinsize', 'f4'),
             ('chisq_index', 'i4'),
             ('refmag_index', 'i4'),
             ('chisqbins', 'f4', chisqbins.size),
             ('chisqrange', 'f4', 2),
             ('chisqbinsize', 'f4'),
             ('lnchisqbins', 'f4', lnchisqbins.size),
             ('lnchisqrange', 'f4', 2),
             ('lnchisqbinsize', 'f4'),
             ('areas', 'f4', areas.size),
             ('refmagbins', 'f4', refmagbins.size),
             ('refmagrange', 'f4', 2),
             ('refmagbinsize', 'f4'),
             ('sigma_g', 'f4', sigma_g.shape),
             ('sigma_lng', 'f4', sigma_lng.shape)]

    chisq_bkg = Entry(np.zeros(1, dtype=dtype))
    chisq_bkg.zbins[:] = zbins
    chisq_bkg.zrange[:] = config.zrange
    chisq_bkg.zbinsize = config.bkg_zbinsize
    chisq_bkg.chisq_index = 0
    chisq_bkg.refmag_index = 1
    chisq_bkg.chisqbins[:] = chisqbins
    chisq_bkg.chisqrange[:] = chisqrange
    chisq_bkg.chisqbinsize = config.bkg_chisqbinsize
    chisq_bkg.lnchisqbins[:] = lnchisqbins
    chisq_bkg.lnchisqrange[:] = lnchisqrange
    chisq_bkg.lnchisqbinsize = lnchisqbinsize
    chisq_bkg.areas[:] = areas
    chisq_bkg.refmagbins[:] = refmagbins
    chisq_bkg.refmagrange[:] = refmagrange
    chisq_bkg.refmagbinsize = config.bkg_refmagbinsize
    chisq_bkg.sigma_g[:, :] = sigma_g
    chisq_bkg.sigma_lng[:, :] = sigma_lng

    chisq_bkg.to_fits_file(config.bkgfile, extname='CHISQBKG', clobber=clobber)


def _make_qa_plots_zred_background(config, sigma_g, zredbins, refmagbins, areas, binsizes, galaxies=None):
    """
    Generate QA plots for zred background calibration.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not hasattr(config, 'plotpath') or config.plotpath is None:
        logger.warning("config.plotpath not set, skipping QA plots")
        return

    os.makedirs(config.plotpath, exist_ok=True)

    # Plot 1: sigma_g vs refmag at different zred slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nzredbins = zredbins.size
    zred_indices = np.linspace(0, nzredbins - 1, 6, dtype=int)
    
    for idx, (ax, zi) in enumerate(zip(axes.flat, zred_indices)):
        ax.step(refmagbins, sigma_g[:, zi], 'b-', linewidth=1.5, where='mid')
        ax.set_xlabel('refmag')
        ax.set_ylabel(r'$\Sigma_g$ [deg$^{-2}$]')
        ax.set_title(f'zred = {zredbins[zi]:.3f}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-2)

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'zredbkg_sigma_g_vs_refmag.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: zredbkg_sigma_g_vs_refmag.png")

    # Plot 2: sigma_g vs zred at different refmag slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    nrefmagbins = refmagbins.size
    refmag_indices = np.linspace(0, nrefmagbins - 1, 6, dtype=int)
    
    for idx, (ax, ri) in enumerate(zip(axes.flat, refmag_indices)):
        ax.step(zredbins, sigma_g[ri, :], 'r-', linewidth=1.5, where='mid')
        ax.set_xlabel('zred')
        ax.set_ylabel(r'$\Sigma_g$ [deg$^{-2}$]')
        ax.set_title(f'refmag = {refmagbins[ri]:.2f}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-2)

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'zredbkg_sigma_g_vs_zred.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: zredbkg_sigma_g_vs_zred.png")

    # Plot 3: Raw galaxy distribution (if provided)
    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(galaxies[0], galaxies[1], gridsize=50, bins='log', cmap='Reds',
                    extent=[refmagbins[0], refmagbins[-1], zredbins[0], zredbins[-1]])
    ax.set_xlabel('refmag')
    ax.set_ylabel('zred')
    ax.set_title('Raw Galaxy Distribution (Hexbin)')
    plt.colorbar(hb, ax=ax, label='log10(N)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'zredbkg_raw_hexbin.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: zredbkg_raw_hexbin.png")

    # Plot 4: Total background density vs zred
    fig, ax = plt.subplots(figsize=(10, 6))
    n_total = np.sum(sigma_g, axis=0) * config.bkg_refmagbinsize
    ax.step(zredbins, n_total, 'k-', linewidth=2, where='mid')
    ax.set_xlabel('zred')
    ax.set_ylabel(r'Total $\Sigma_g$ [deg$^{-2}$]')
    ax.set_title('Total Background Density vs Zred')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'zredbkg_total_vs_zred.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: zredbkg_total_vs_zred.png")

    # Plot 5: Area vs refmag
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(refmagbins, areas, 'g-', linewidth=2, where='mid')
    ax.set_xlabel('refmag')
    ax.set_ylabel('Area [deg$^2$]')
    ax.set_title('Effective Area vs Reference Magnitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, 'zredbkg_area_vs_refmag.png'), dpi=300)
    plt.close()
    logger.info("Saved QA plot: zredbkg_area_vs_refmag.png")

def generate_zred_background(config, clobber=False, natatime=100000):
    """
    Generate the zred galaxy background.  The output filename is specified
    in config.bkgfile.

    Parameters
    ----------
    config: `redmapper.Configuration`
    clobber: `bool`, optional
       Overwrite any existing config.bkgfile file.  Default is False.
    natatime: `int`, optional
       Number of galaxies to read at a time.  Default is 100000
    """

    if not os.path.isfile(config.zredfile):
        raise RuntimeError("Must run generate_zred_background with a zred file")

    if not clobber:
        if os.path.isfile(config.bkgfile):
            with fitsio.FITS(config.bkgfile) as fits:
                if 'ZREDBKG' in [ext.get_extname() for ext in fits[1: ]]:
                    logger.info("ZREDBKG already in %s and clobber is False" % (config.bkgfile))
                    return

    # Read in zred parameters
    zredstr = read_redsequence(config.parfile, fine=True, zrange=config.zrange)

    # Set ranges
    refmagrange = np.array([12.0, config.limmag_catalog])
    nrefmagbins = np.ceil((refmagrange[1] - refmagrange[0]) / config.bkg_refmagbinsize).astype(np.int32)
    refmagbins = np.arange(nrefmagbins) * config.bkg_refmagbinsize + refmagrange[0]

    zredrange = np.array([zredstr['z'][0], zredstr['z'][-2] + (zredstr['z'][1] - zredstr['z'][0])])
    nzredbins = np.ceil((zredrange[1] - zredrange[0]) / config.bkg_zredbinsize).astype(np.int32)
    zredbins = np.arange(nzredbins) * config.bkg_zredbinsize + zredrange[0]

    # Compute the areas...
    # This takes into account the configured sub-region
    if config.depthfile is not None:
        depth_data = depthmap.read_depth_map(config)
        areas = depthmap.compute_areas(depth_data, refmagbins)
    else:
        areas = np.zeros(refmagbins.size) + config.area

    maxchisq = config.wcen_zred_chisq_max

    # Prepare pixels (if necessary) and count galaxies

    if not config.galfile_pixelized:
        raise ValueError("Only pixelized galfiles are supported at this moment.")

    master = Entry.from_fits_file(config.galfile)

    if len(config.hpix) > 0:
        # We need to take a sub-region
        theta, phi = hpg.pixel_to_angle(master.nside, master.hpix, lonlat=False, nest=False)
        ipring_big = hpg.angle_to_pixel(config.nside, theta, phi, lonlat=False, nest=False)

        _, subreg_indices = esutil.numpy_util.match(config.hpix, ipring_big)
        subreg_indices = np.unique(subreg_indices)
    else:
        subreg_indices = np.arange(master.hpix.size)

    ngal = np.sum(master.ngals[subreg_indices])
    npix = subreg_indices.size

    starttime = time.time()

    nmag = config.nmag
    ncol = nmag - 1

    zreds = np.zeros(ngal, dtype=np.float32) - 1.0
    refmags = np.zeros(ngal, dtype=np.float32)

    zbinmid = np.median(np.arange(zredstr['z'].size, dtype=np.int32))

    # Loop
    ctr = 0
    p = 0
    while ((ctr < ngal) and (p < npix)):
        if master.ngals[subreg_indices[p]] == 0:
            p += 1
            continue

        gals = GalaxyCatalog.from_galfile(config.galfile, nside=master.nside,
                                          hpix=master.hpix[subreg_indices[p]],
                                          border=0.0,
                                          zredfile=config.zredfile)

        use, = np.where(gals.chisq < maxchisq)

        if use.size > 0:
            lo = ctr
            hi = ctr + use.size

            inds = np.arange(lo, hi, dtype=np.int64)

            refmags[inds] = gals.refmag[use]
            zreds[inds] = gals.zred[use]

        ctr += master.ngals[subreg_indices[p]]
        p += 1

    # Compute cic
    sigma_g = np.zeros((nrefmagbins, nzredbins))

    binsizes = config.bkg_refmagbinsize * config.bkg_zredbinsize

    use, = np.where((zreds >= zredrange[0]) & (zreds < zredrange[1]) &
                    (refmags > refmagrange[0]) & (refmags < refmagrange[1]))

    zredpos = (zreds[use] - zredrange[0]) * nzredbins / (zredrange[1] - zredrange[0])
    refmagpos = (refmags[use] - refmagrange[0]) * nrefmagbins / (refmagrange[1] - refmagrange[0])

    value = np.ones(use.size)

    field = cic(value, zredpos, nzredbins, refmagpos, nrefmagbins, isolated=True)

    sigma_g[:, :] = field

    for j in range(nzredbins):
        sigma_g[:, j] = np.clip(field[:, j], 0.1, None) / (areas * binsizes)

    logger.info("Finished zred background in %.2f seconds" % (time.time() - starttime))

    # Generate QA plots if requested
    if hasattr(config, 'more_qa_plots') and config.more_qa_plots:
        _make_qa_plots_zred_background(config, sigma_g, zredbins, refmagbins, areas, binsizes, galaxies=(refmags[use], zreds[use]))

    # save it

    dtype = [('zredbins', 'f4', zredbins.size),
             ('zredrange', 'f4', zredrange.size),
             ('zredbinsize', 'f4'),
             ('zred_index', 'i2'),
             ('refmag_index', 'i2'),
             ('refmagbins', 'f4', refmagbins.size),
             ('refmagrange', 'f4', refmagrange.size),
             ('refmagbinsize', 'f4'),
             ('areas', 'f4', areas.size),
             ('sigma_g', 'f4', sigma_g.shape)]

    zred_bkg = Entry(np.zeros(1, dtype=dtype))
    zred_bkg.zredbins[:] = zredbins
    zred_bkg.zredrange[:] = zredrange
    zred_bkg.zredbinsize = config.bkg_zredbinsize
    zred_bkg.zred_index = 0
    zred_bkg.refmag_index = 1
    zred_bkg.refmagbins[:] = refmagbins
    zred_bkg.refmagrange[:] = refmagrange
    zred_bkg.refmagbinsize = config.bkg_refmagbinsize
    zred_bkg.areas[:] = areas
    zred_bkg.sigma_g[:, :] = sigma_g

    zred_bkg.to_fits_file(config.bkgfile, extname='ZREDBKG', clobber=clobber)

