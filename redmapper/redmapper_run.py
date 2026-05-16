"""Function for performing a local redmapper run with multiprocessing.

This function is typically used during training.
"""
import numpy as np
import copy
import fitsio
import re
import os
import sys
import esutil
import hpgeom as hpg
from esutil.cosmology import Cosmo

import multiprocessing

from .catalog import Catalog, Entry
from .run_firstpass import run_firstpass
from .run_likelihoods import run_likelihoods
from .run_percolation import run_percolation
from .cluster_runner import output_cluster_catalog
from .utilities import getMemoryString
from .logger import logger

def _get_subpixels(config, nside_test, galtab):
    """
    Get all the pixels from a galaxy table corresponding to a given nside.

    Note that this takes into account the subregion of galtab that is being
    processed.

    Parameters
    ----------
    config: `redmapper.Configuration`
    nside_test: `int`
       Nside of the desired grouping of galtab.
    galtab: `redmapper.Entry`
       Galaxy table summary information.

    Returns
    -------
    pixels: `np.array`
       Integer array of pixels that cover galtab.
    """

    # generate all the pixels
    pixels = np.arange(hpg.nside_to_npixel(int(nside_test)))

    # Which of these match the parent?
    if len(config.hpix) > 0:
        theta, phi = hpg.pixel_to_angle(nside_test, pixels, lonlat=False, nest=False)
        hpix_test = hpg.angle_to_pixel(config.nside, theta, phi, lonlat=False, nest=False)
        a, b = esutil.numpy_util.match(config.hpix, hpix_test)
        pixels = pixels[b]

    # And which match the galaxies?
    theta, phi = hpg.pixel_to_angle(galtab.nside, galtab.hpix, lonlat=False, nest=False)
    hpix_test = hpg.angle_to_pixel(nside_test, theta, phi, lonlat=False, nest=False)
    a, b = esutil.numpy_util.match(pixels, hpix_test)

    return np.unique(pixels[a])

def _get_pixel_splits(config):
    """
    Get the subpixels on which to run to optimally split the input catalog
    based on the number of cores for the run.

    Returns
    -------
    nside_split: `int`
       Healpix nside for the split pixels
    pixels_split: `list`
       Integer list of healpix pixel numbers (ring format)
    """

    tab = Entry.from_fits_file(config.galfile, ext=1)

    if config.calib_run_nproc == 1:
        if config.nside > config.calib_run_min_nside:
            nside_test = config.run_min_nside
        else:
            nside_test = np.clip(config.nside, 1, None)
        subpixels = _get_subpixels(config, nside_test, tab)
        return (nside_test, subpixels)

    # start with the pixel and resolution in the config file
    if len(config.hpix) == 0:
        nside_splits = [config.calib_run_min_nside]
        pixels_splits = [_get_subpixels(config, nside_splits[0], tab)]
    else:
        if config.nside > config.calib_run_min_nside:
            nside_splits = [config.nside]
        else:
            nside_splits = [config.calib_run_min_nside]
        pixels_splits = [_get_subpixels(config, nside_splits[0], tab)]

    nsplit = [len(pixels_splits[0])]

    nside_test = nside_splits[0]
    while nside_test < tab.nside:
        # increment nside_test
        nside_test *= 2

        pixels_test = _get_subpixels(config, nside_test, tab)
        nsplit.append(pixels_test.size)
        nside_splits.append(nside_test)
        pixels_splits.append(pixels_test)

    test, = np.where(np.array(nsplit) <= config.calib_run_nproc*2)
    if test.size == 0:
        nside_split = nside_splits[0]
        pixels_split = pixels_splits[0]
    else:
        nside_split = nside_splits[test[-1]]
        pixels_split = pixels_splits[test[-1]]

    return (nside_split, pixels_split)


def _consolidate(config, hpixels, filenames, filetype, members=False, check=True):
    """
    Consolidate pixel run files.

    Parameters
    ----------
    config: `redmapper.Configuration`
    hpixels: `np.array`
       Integer array of healpix pixels to consolidate
    filenames: `list`
       List of strings of filenames to consolidate
    filetype: `str`
       Type of file (final, like)
    members: `bool`, optional
       Consolidate members as well?  Default is False.
    check: `bool`, optional
       Check to see if consolidated files exist (and exit if so).
       Default is False

    Returns
    -------
    outfile: `str`
       Output filename
    """

    outfile = config.redmapper_filename(filetype)
    memfile = config.redmapper_filename(filetype+'_members')

    if check:
        outfile_there = os.path.isfile(outfile)
        memfile_there = os.path.isfile(memfile)

        if (outfile_there and memfile_there and members):
            # All files are accounted for
            return outfile
        if outfile_there and not members:
            return outfile

    # How many clusters are there?  (This is the maximum before cuts)
    ncluster = 0
    for f in filenames:
        hdr = fitsio.read_header(f, ext=1)
        ncluster += hdr['NAXIS2']

    element = Entry.from_fits_file(filenames[0], ext=1, rows=0)
    dtype = element._ndarray.dtype

    ubercat = Catalog(np.zeros(ncluster, dtype=dtype))
    ctr = 0

    ubermem = None

    for hpix, f in zip(hpixels, filenames):
        cat = Catalog.from_fits_file(f, ext=1)

        # Cut to minlambda, maxfrac, and within a pixel
        if config.nside > 0:
            ipring = hpg.angle_to_pixel(config.nside, cat.ra, cat.dec, nest=False)
        else:
            # Set all the pixels to 0
            ipring = np.zeros(cat.size, dtype=np.int32)

        use, = np.where((ipring == hpix) &
                        (cat.maskfrac < config.max_maskfrac) &
                        (cat.Lambda / cat.scaleval > config.percolation_minlambda))

        # Make sure we have surviving clusters
        if use.size == 0:
            continue

        cat = cat[use]

        if members:
            parts = f.split('.fit')
            mem = Catalog.from_fits_file(parts[0] + '_members.fit')

            # We are going to replace the mem_match_ids in the consolidated catalog,
            # because the ones generated in the pixels aren't unique
            new_ids = np.arange(use.size, dtype=np.int32) + ctr + 1

            a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
            cat.mem_match_id = new_ids
            mem.mem_match_id[b] = cat.mem_match_id[a]

            # and we only want to store the members that matched!
            mem = mem[b]

        # And copy the fields
        for n in dtype.names:
            ubercat[n][ctr: ctr + cat.size] = cat[n]

        if members:
            if ubermem is None:
                ubermem = mem
            else:
                ubermem.append(mem)

        ctr += cat.size

    # Crop the catalog to the range that we had clusters
    ubercat = ubercat[0:ctr]

    # Now we need a final sorting by likelihood and mem_match_id replacement
    if members:
        st = np.argsort(ubercat.lnlike)[::-1]
        ubercat = ubercat[st]

        a, b = esutil.numpy_util.match(ubercat.mem_match_id, ubermem.mem_match_id)

        ubercat.mem_match_id = np.arange(ubercat.size, dtype=np.int32) + 1
        ubermem.mem_match_id[b] = ubercat.mem_match_id[a]

    # And write out...
    # We can clobber because if it was already there and we wanted to check
    # that already happened
    ubercat.to_fits_file(outfile, clobber=True)

    if members:
        ubermem.to_fits_file(memfile, clobber=True)

    return outfile

def _make_qa_plots(config, finalfile):
    """
    Generate QA plots for the redmapper run.

    Parameters
    ----------
    config: `redmapper.Configuration`
    finalfile: `str`
       Path to the final consolidated catalog file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping QA plots")
        return

    if not hasattr(config, 'plotpath') or config.plotpath is None:
        logger.warning("plotpath not set in config, skipping QA plots")
        return

    if not os.path.exists(config.plotpath):
        os.makedirs(config.plotpath)

    if finalfile is None or not os.path.isfile(finalfile):
        logger.warning("Final catalog file not found, skipping QA plots")
        return

    logger.info("Generating redmapper run QA plots...")

    # Read the catalog
    cat = Catalog.from_fits_file(finalfile)

    if cat.size == 0:
        logger.warning("Empty catalog, skipping QA plots")
        return

    # Plot 1: Lambda distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat.Lambda, bins=50, alpha=0.7, edgecolor='black', range=(0, 200))
    ax.set_xlabel(r'$\lambda$ (Richness)', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Cluster Richness Distribution', fontsize=14)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_lambda_dist.png'), dpi=300)
    plt.close()

    # Plot 2: Redshift distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat.z_lambda, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(r'$z_\lambda$ (Cluster Redshift)', fontsize=12)
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Cluster Redshift Distribution', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_z_dist.png'), dpi=300)
    plt.close()

    # Plot 3: Lambda vs z scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(cat.z_lambda, cat.Lambda, gridsize=(50, 50), cmap='Reds', mincnt=1, yscale='log')
    ax.set_xlabel(r'$z_\lambda$ (Cluster Redshift)', fontsize=12)
    ax.set_ylabel(r'$\lambda$ (Richness)', fontsize=12)
    ax.set_title('Richness vs Redshift', fontsize=14)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Number of Clusters', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_lambda_vs_z.png'), dpi=300)
    plt.close()

    # Plot 4: Spatial distribution
    # Calculate RA and Dec ranges to set aspect ratio = 1
    ra_range = cat.ra.max() - cat.ra.min()
    dec_range = cat.dec.max() - cat.dec.min()
    aspect_ratio = ra_range / dec_range if dec_range > 0 else 1.0
    fig_height = 8
    fig_width = fig_height * aspect_ratio
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(cat.ra, cat.dec, c=cat.z_lambda, s=10, alpha=0.6, cmap='Reds')
    ax.set_xlabel('RA (degrees)', fontsize=12)
    ax.set_ylabel('Dec (degrees)', fontsize=12)
    ax.set_title('Spatial Distribution of Clusters (colored by redshift)', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$z_\lambda$', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_spatial_dist.png'), dpi=300)
    plt.close()

    # Plot 5: Redshift error distribution
    if hasattr(cat, 'z_lambda_e'):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cat.z_lambda_e, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(r'$\sigma_{z_\lambda}$ (Redshift Error)', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Cluster Redshift Error Distribution', fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_z_err_dist.png'), dpi=300)
        plt.close()

    # Plot 6: Maskfrac distribution
    if hasattr(cat, 'maskfrac'):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cat.maskfrac, bins=50, alpha=0.7, edgecolor='black', range=(0, 1))
        ax.set_xlabel('Mask Fraction', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Mask Fraction Distribution', fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.plotpath, f'{config.outbase}_maskfrac_dist.png'), dpi=300)
        plt.close()

    logger.info("QA plots saved to %s" % config.plotpath)

def _worker_wrapper(args):
    """Wrapper to unpack arguments for _worker"""
    return _worker(*args)

def _worker(hpix, config, H0, omega_l, omega_m, specmode, check, keepz, cleaninput):
    """
    Do the run on one pixel (for multiprocessing).

    Outputs
    -------
    hpix: `int`
       Healpix ring number that was run.
    firstpass_filename: `str`
       Filename for firstpass file.
    like_filename: `str`
       Filename for likelihood file
    perc_filename: `str`
       Filename for percolation file.
    """
    logger.info("Running on pixel %d" % (hpix))

    config = config.copy()
    config.cosmo = Cosmo(H0=H0, omega_l=omega_l, omega_m=omega_m)

    config.hpix = [hpix]
    config.outbase = '%s_%d_%05d' % (config.outbase, config.nside, hpix)

    # run firstpass
    filetype_fp = 'firstpass_spec' if specmode else 'firstpass'
    firstpass_filename = config.redmapper_filename(filetype_fp + '_catalog')

    if not os.path.isfile(firstpass_filename) or not check:
        cat_fp, members_fp = run_firstpass(config, keepz=keepz, cleaninput=cleaninput, specmode=specmode)

        if cat_fp is None or cat_fp.size == 0:
            logger.info("Did not produce a firstpass catalog for pixel %d" % (hpix))
            return (hpix, None, None, None)

        output_cluster_catalog(cat_fp, None, config, filetype_fp, savemembers=False, withversion=False, clobber=True)
    else:
        logger.info("Firstpass file %s already present.  Skipping..." % (firstpass_filename))

    config.catfile = firstpass_filename

    # run likelihoods
    filetype_like = 'like'
    like_filename = config.redmapper_filename(filetype_like + '_catalog')

    if not os.path.isfile(like_filename) or not check:
        cat_like, members_like = run_likelihoods(config, keepz=keepz, cleaninput=cleaninput)

        if cat_like is None or cat_like.size == 0:
            logger.info("Did not produce a likelihood catalog for pixel %d" % (hpix))
            return (hpix, firstpass_filename, None, None)

        output_cluster_catalog(cat_like, None, config, filetype_like, savemembers=False, withversion=False, clobber=True)
    else:
        logger.info("Likelihood file %s already present.  Skipping..." % (like_filename))

    config.catfile = like_filename

    # run percolation
    filetype_perc = 'perc'
    perc_filename = config.redmapper_filename(filetype_perc + '_catalog')

    if not os.path.isfile(perc_filename) or not check:
        cat_perc, members_perc = run_percolation(config, keepz=keepz, cleaninput=cleaninput)

        if cat_perc is None or cat_perc.size == 0:
            logger.info("Did not produce a percolation catalog for pixel %d" % (hpix))
            return (hpix, firstpass_filename, like_filename, None)

        output_cluster_catalog(cat_perc, members_perc, config, filetype_perc, savemembers=True, withversion=False, clobber=True)
    else:
        logger.info("Percolation file %s already present.  Skipping..." % (perc_filename))

    return (hpix, firstpass_filename, like_filename, perc_filename)

def _percolation_only_worker_wrapper(args):
    """Wrapper to unpack arguments for _percolation_only_worker"""
    return _percolation_only_worker(*args)

def _percolation_only_worker(hpix, config, H0, omega_l, omega_m, check, keepz, cleaninput):
    """
    Do a percolation only run on one pixel (for multiprocessing).
    """
    logger.info("Running percolation on pixel %d" % (hpix))

    config = config.copy()
    config.cosmo = Cosmo(H0=H0, omega_l=omega_l, omega_m=omega_m)

    config.hpix = [hpix]
    config.outbase = '%s_%05d' % (config.outbase, hpix)

    filetype_perc = 'perc'
    perc_filename = config.redmapper_filename(filetype_perc + '_catalog')

    if not os.path.isfile(perc_filename) or not check:
        cat_perc, members_perc = run_percolation(config, keepz=keepz, cleaninput=cleaninput)

        if cat_perc is None or cat_perc.size == 0:
            logger.info("Did not produce a percolation catalog for pixel %d" % (hpix))
            return (hpix, None, None, None)
            
        output_cluster_catalog(cat_perc, members_perc, config, filetype_perc, savemembers=True, withversion=False, clobber=True)

    return (hpix, None, None, perc_filename)

def redmapper_run(config, specmode=False, seedfile=None, check=True,
                  percolation_only=False, consolidate_like=False, keepz=False, cleaninput=False,
                  consolidate=True):
    """
    Run the redmapper cluster finder using multiprocessing.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    specmode: `bool`, optional
       Run with spectroscopic mode (firstpass uses zspec as seeds).
       Default is False.
    seedfile: `str`, optional
       File containing spectroscopic seeds.  Default is None.
    check: `bool`, optional
       Check if files already exist, and skip if so.  Default is True.
    percolation_only: `bool`, optional
       Only run the percolation phase.  Default is False.
    consolidate_like: `bool`, optional
       Consolidate the pixel runs for the likelihood files?  Default is False.
    keepz: `bool`, optional
       Keep input redshifts or replace with z_lambda?  Default is False.
    cleaninput: `bool`, optional
       Processing stage should clean out bad clusters?  Default is False.
    consolidate: `bool`, optional
       Consolidate the pixel runs for the percolated files?  Default is True.

    Returns
    -------
    finalfile: `str`
       Filename for the final consolidated percolation file.
    likefile: `str`
       Filename for the final consolidated likelihood file
       (if consolidate_like == True)
    """
    config = config.copy()
    config.start_file_logging()

    if specmode and not keepz:
        raise RuntimeError("Must set keepz=True when specmode=True")
    if percolation_only and specmode:
        raise RuntimeError("Cannot set both percolation_only=True and specmode=True")

    nside_split, pixels_split = _get_pixel_splits(config)

    logger.info("Running on %d pixels" % (len(pixels_split)))

    # run each individual one
    nside_orig = config.nside
    config.nside = nside_split

    orig_seedfile = config.seedfile
    if seedfile is not None:
        # Use the specified seedfile if desired
        config.seedfile = seedfile
        
    H0 = config.cosmo.H0()
    omega_l = config.cosmo.omega_l()
    omega_m = config.cosmo.omega_m()
    config_copy = config.copy()
    config_copy.cosmo = None
        
    mp_ctx = multiprocessing.get_context("fork")
        
    pool = mp_ctx.Pool(processes=config.calib_run_nproc)
    
    if percolation_only:
        args_list = [(hp, config_copy, H0, omega_l, omega_m, check, keepz, cleaninput) for hp in pixels_split]
        retvals = pool.map(_percolation_only_worker_wrapper, args_list, chunksize=1)
    else:
        args_list = [(hp, config_copy, H0, omega_l, omega_m, specmode, check, keepz, cleaninput) for hp in pixels_split]
        retvals = pool.map(_worker_wrapper, args_list, chunksize=1)
        
    pool.close()
    pool.join()

    # Reset the seedfile
    config.seedfile = orig_seedfile

    # Consolidate (adds additional mask cuts)
    hpixels_like = [x[0] for x in retvals if x[2] is not None]
    likefiles = [x[2] for x in retvals if x[2] is not None]
    hpixels_perc = [x[0] for x in retvals if x[3] is not None]
    percfiles = [x[3] for x in retvals if x[3] is not None]

    # Allow for runs without consolidation
    if consolidate:
        finalfile = _consolidate(config, hpixels_perc, percfiles, 'final', members=True, check=check)
    else:
        finalfile = None

    if consolidate_like:
        likefile = _consolidate(config, hpixels_like, likefiles, 'like', members=False, check=check)

    # Reset the nside in the config file
    config.nside = nside_orig

    # Generate QA plots if requested
    if hasattr(config, 'more_qa_plots') and config.more_qa_plots and consolidate:
        _make_qa_plots(config, finalfile)

    # And done
    if consolidate_like:
        return (finalfile, likefile)
    else:
        return finalfile


