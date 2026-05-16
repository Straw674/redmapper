"""Functional interfaces for running a galaxy catalog through zred galaxy redshift computation,
using multiprocessing.
"""
import numpy as np
import copy
import fitsio
import re
import os
import time
import multiprocessing

from .zred_color import compute_zreds
from .galaxy import GalaxyCatalog, get_subpixel_indices, zred_extra_dtype
from .catalog import Catalog, Entry
from .redsequence import read_redsequence
from .logger import logger

def _zred_catalog_worker(ind_range, galaxyfile, config, zredstr, rng):
    """
    Worker function for run_zred_catalog.
    """
    in_cat = fitsio.read(galaxyfile, ext=1, upper=True,
                         rows=np.arange(ind_range[0], ind_range[1]))
    galaxies = GalaxyCatalog(in_cat)
    galaxies.add_zred_fields(config['zred_nsamp'])

    compute_zreds(zredstr, galaxies, rng=rng)

    return (ind_range,
            galaxies.zred, galaxies.zred_e,
            galaxies.zred2, galaxies.zred2_e,
            galaxies.zred_uncorr, galaxies.zred_uncorr_e,
            galaxies.zred_samp,
            galaxies.lkhd, galaxies.chisq)

def run_zred_catalog(config, galaxyfile, outfile, clobber=False, nperproc=None, maxperproc=500000, rng=None):
    """
    Run a galaxy file to compute zreds and output zreds to output file.

    Parameters
    ----------
    config: `redmapper.Config`
        Configuration object
    galaxyfile: `str`
       Galaxy input file
    outfile: `str`
       Output zred file
    clobber: `bool`, optional
       Clobber existing outfile?  Default is False.
    nperproc: `int`, optional
       Number of galaxies to run per processor.
    maxperproc: `int`, optional
       Maximum number to run per processor.
    rng: `np.random.RandomState`, optional
        Random number generator.
    """
    config_copy = config.copy()
    config_copy['cosmo'] = None

    if rng is None:
        rng = np.random.RandomState(config_copy['randomseed'])

    hdr = fitsio.read_header(galaxyfile, ext=1)
    ngal = hdr['NAXIS2']

    zredstr = read_redsequence(config_copy['parfile'])

    zreds = Catalog(np.zeros(ngal, dtype=zred_extra_dtype(config_copy['zred_nsamp'])))

    if nperproc is None:
        nperproc = int(float(ngal) / (config_copy['calib_nproc'] - 0.1))
        nperproc = np.clip(nperproc, None, maxperproc)

    inds = np.arange(0, ngal, nperproc)
    worker_list = [((ind, np.clip(ind + nperproc, None, ngal)), galaxyfile, config_copy, zredstr, rng) for ind in inds]

    mp_ctx = multiprocessing.get_context("fork")
    pool = mp_ctx.Pool(processes=config_copy['calib_nproc'])
    retvals = pool.starmap(_zred_catalog_worker, worker_list, chunksize=1)
    pool.close()
    pool.join()

    for ind_range, zred, zred_e, zred2, zred2_e, zred_uncorr, zred_uncorr_e, zred_samp, lkhd, chisq in retvals:
        zreds.zred[ind_range[0]: ind_range[1]] = zred
        zreds.zred_e[ind_range[0]: ind_range[1]] = zred_e
        zreds.zred2[ind_range[0]: ind_range[1]] = zred2
        zreds.zred2_e[ind_range[0]: ind_range[1]] = zred2_e
        zreds.zred_uncorr[ind_range[0]: ind_range[1]] = zred_uncorr
        zreds.zred_uncorr_e[ind_range[0]: ind_range[1]] = zred_uncorr_e
        zreds.zred_samp[ind_range[0]: ind_range[1], :] = zred_samp
        zreds.lkhd[ind_range[0]: ind_range[1]] = lkhd
        zreds.chisq[ind_range[0]: ind_range[1]] = chisq

    zreds.to_fits_file(outfile, clobber=clobber)

def _zred_pixels_worker(index, config, zredstr, outbase, zredpath):
    """
    Worker function for run_zred_pixels.
    """
    # Read in just one single pixel
    galtable = Entry.from_fits_file(config['galfile'])
    galaxies = GalaxyCatalog.from_galfile(config['galfile'],
                                          nside=galtable.nside,
                                          hpix=[galtable.hpix[index]],
                                          border=0.0)
    galaxies.add_zred_fields(config['zred_nsamp'])

    compute_zreds(zredstr, galaxies)

    # And write out the pixel file ... but just the zreds
    zreds = np.zeros(galaxies.size, dtype=zred_extra_dtype(config['zred_nsamp']))
    for dt in zred_extra_dtype(config['zred_nsamp']):
        zreds[dt[0]][:] = galaxies._ndarray[dt[0].lower()][:]

    outfile_nopath = '%s_zreds_%07d.fit' % (outbase, galtable.hpix[index])
    outfile = os.path.join(zredpath, outfile_nopath)

    fitsio.write(outfile, zreds, clobber=True)

    return (index, outfile)

def run_zred_pixels(config, single_process=False, no_zred_table=False, verbose=False):
    """
    Run all the galaxies in a pixelized config['galfile'] to compute
    zreds and save zreds.

    Parameters
    ----------
    config: `redmapper.Config`
        Configuration object
    single_process: `bool`, optional
       Run as a single process only.  Useful for testing.  Default is False.
    no_zred_table: `bool`, optional
       Do not output a final zred table, instead return numbers.
       Default is False.
    verbose: `bool`, optional
       Be verbose with output.  Default is False.

    Returns
    -------
    retvals: `list`
       Present only if no_zred_table is True.
       List of (index, outfile) tuples describing output files.
    """
    if not config['galfile_pixelized']:
        raise ValueError("Code only runs with a pixelized galfile.")

    config_copy = config.copy()
    config_copy['cosmo'] = None

    zredpath = os.path.dirname(config_copy['zredfile'])
    
    test = re.search('^(.*)_zreds_master_table.fit',
                     os.path.basename(config_copy['zredfile']))
    if test is None:
        raise ValueError("zredfile filename not in proper format (must end with _zreds_master_table.fit)")

    outbase = test.groups()[0]

    # Make the output directory if necessary
    if not os.path.exists(zredpath):
        os.makedirs(zredpath)

    zredstr = read_redsequence(config_copy['parfile'])

    galtable = Entry.from_fits_file(config_copy['galfile'])
    indices = list(get_subpixel_indices(galtable,
                                        hpix=config_copy['hpix'], 
                                        border=config_copy['border'], 
                                        nside=config_copy['nside']))

    starttime = time.time()

    if not single_process:
        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=config_copy['calib_nproc'])
        worker_list = [(index, config_copy, zredstr, outbase, zredpath) for index in indices]
        retvals = pool.starmap(_zred_pixels_worker, worker_list, chunksize=1)
        pool.close()
        pool.join()
    else:
        if (verbose):
            total_galaxies = np.sum(galtable.ngals[indices])
            logger.info("Computing zred for %d galaxies in %d pixels." % (total_galaxies, len(indices)))
        
        retvals = []
        for index in indices:
            retvals.append(_zred_pixels_worker(index, config_copy, zredstr, outbase, zredpath))

    logger.info("Done computing zreds in %.2f seconds" % (time.time() - starttime))

    if no_zred_table:
        return retvals

    make_zred_table(config_copy, retvals, galtable, outbase)

    # Generate QA plots if requested
    if config_copy.get('more_qa_plots'):
        _make_zred_qa_plots(config_copy, retvals, outbase)

def make_zred_table(config, indices_and_filenames, galtable, outbase):
    """
    Make a zred table from a list of indices and filenames

    Saves to config['zredfile'].
    """
    # figure out longest filename
    maxlen = 0
    for index, filename in indices_and_filenames:
        if len(os.path.basename(filename)) > maxlen:
            maxlen = len(os.path.basename(filename))

    gal_dtype = galtable.dtype

    zred_dtype = []
    for dt in gal_dtype.descr:
        if dt[0].lower() == 'filenames':
            dt = ('filenames', 'S%d' % (maxlen + 1), dt[2])
        zred_dtype.append(dt)

    zredtable = Entry(np.zeros(1, dtype=zred_dtype))
    for name in gal_dtype.names:
        if name.lower() != 'filenames':
            zredtable[name] = galtable._ndarray[name]

    zredtable.filenames = ''
    zredtable.ngals = 0

    for index, filename in indices_and_filenames:
        # Make sure file exists
        if not os.path.isfile(filename):
            raise ValueError("Could not find zredfile: %s" % (filename))
        # check size of file
        hdr = fitsio.read_header(filename, ext=1)
        if hdr['NAXIS2'] != galtable.ngals[index]:
            raise ValueError("Length mismatch for zredfile: %s" % (filename))

        zredtable.filenames[index] = os.path.basename(filename)
        zredtable.ngals[index] = galtable.ngals[index]

    hdr = fitsio.FITSHDR()
    hdr['PIXELS'] = 1

    zredtable.to_fits_file(config['zredfile'], header=hdr, clobber=True)

def _make_zred_qa_plots(config, indices_and_filenames, outbase):
    """
    Generate QA plots for zred computation results.
    """
    import matplotlib.pyplot as plt

    # Create plotpath if it doesn't exist
    plotpath = config.get('plotpath')
    if plotpath is None:
        logger.warning("plotpath not set in config, skipping QA plots")
        return

    full_plotpath = os.path.join(config['outpath'], plotpath)
    if not os.path.exists(full_plotpath):
        os.makedirs(full_plotpath)

    logger.info("Generating zred QA plots...")

    # Read all zred data
    all_zred = []
    all_zred_e = []
    all_chisq = []
    all_lkhd = []

    for index, filename in indices_and_filenames:
        if os.path.isfile(filename):
            data = fitsio.read(filename, ext=1, upper=True)
            all_zred.append(data["ZRED"])
            all_zred_e.append(data["ZRED_E"])
            all_chisq.append(data["CHISQ"])
            all_lkhd.append(data["LKHD"])

    if len(all_zred) == 0:
        logger.warning("No zred data found, skipping QA plots")
        return

    all_zred = np.concatenate(all_zred)
    all_zred_e = np.concatenate(all_zred_e)
    all_chisq = np.concatenate(all_chisq)
    all_lkhd = np.concatenate(all_lkhd)

    # Filter out bad values
    good = (
        (all_zred > 0)
        & (all_zred_e > 0)
        & np.isfinite(all_zred)
        & np.isfinite(all_zred_e)
    )

    # Plot 1: zred distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_zred[good], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("zred", fontsize=12)
    ax.set_ylabel("Number of Galaxies", fontsize=12)
    ax.set_title("Photometric Redshift Distribution", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_plotpath, f"{outbase}_zred_hist.png"), dpi=300
    )
    plt.close()

    # Plot 2: zred vs zred_e
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(
        all_zred[good],
        all_zred_e[good],
        gridsize=50,
        cmap="Reds",
        mincnt=1,
        bins="log",
    )
    ax.set_xlabel("zred", fontsize=12)
    ax.set_ylabel("zred_e", fontsize=12)
    ax.set_title("Photometric Redshift vs Error", fontsize=14)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Number of Galaxies", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_plotpath, f"{outbase}_zred_vs_err.png"),
        dpi=300,
    )
    plt.close()

    # Plot 3: chi-squared distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    chisq_good = all_chisq[good & (all_chisq > 0)]
    if len(chisq_good) > 0:
        ax.hist(np.log10(chisq_good), bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("log10(χ²)", fontsize=12)
        ax.set_ylabel("Number of Galaxies", fontsize=12)
        ax.set_title("Chi-squared Distribution", fontsize=14)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_plotpath, f"{outbase}_chisq_hist.png"),
        dpi=300,
    )
    plt.close()

    # Plot 4: likelihood distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    lkhd_good = all_lkhd[good]
    if len(lkhd_good) > 0:
        ax.hist(np.exp(lkhd_good), bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Likelihood", fontsize=12)
        ax.set_ylabel("Number of Galaxies", fontsize=12)
        ax.set_title("Likelihood Distribution", fontsize=14)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_plotpath, f"{outbase}_lkhd_hist.png"), dpi=300
    )
    plt.close()

    # Plot 5: zred error distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(all_zred_e[good], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("zred_e", fontsize=12)
    ax.set_ylabel("Number of Galaxies", fontsize=12)
    ax.set_title("Photometric Redshift Error Distribution", fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_plotpath, f"{outbase}_zred_e_hist.png"),
        dpi=300,
    )
    plt.close()

    logger.info("QA plots saved to %s" % full_plotpath)

