"""Functions to run redmapper on a single pixel, for distributed runs.
"""
import os
import numpy as np
import glob

from ..configuration import Configuration
from ..utilities import make_lockfile
from ..run_firstpass import run_firstpass
from ..run_likelihoods import run_likelihoods
from ..run_percolation import run_percolation
from ..cluster_runner import output_cluster_catalog
from ..run_randoms_zmask import run_randoms_zmask
from ..run_zscan import run_zscan
from ..runcat import run_catalog

from ..utilities import getMemoryString
from ..logger import logger

def run_redmapper_pixel_task(configfile, pixel, nside, path=None):
    """
    Run redmapper on a single healpix pixel, for distributed runs.

    Parameters
    ----------
    configfile: `str`
       Configuration yaml filename.
    pixel: `int`
       Healpix pixel to run on.
    nside: `int`
       Healpix nside associated with pixel.
    path: `str`, optional
       Output path.  Default is None, use same absolute
       path as configfile.
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=path)

    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

    # Compute the border size
    config.border = config.compute_border()

    config.hpix = [pixel]
    config.nside = nside
    config.outbase = '%s_%d_%05d' % (config.outbase, nside, pixel)

    # Do the run
    config.start_file_logging()
    logger.info("Running redMaPPer on pixel %d" % (pixel))

    firstpass_filename = config.redmapper_filename('firstpass_catalog')
    if not os.path.isfile(firstpass_filename):
        cat_fp, members_fp = run_firstpass(config)
        output_cluster_catalog(cat_fp, members_fp, config, 'firstpass', savemembers=False, withversion=False)
    else:
        logger.info("Firstpass file %s already present.  Skipping..." % (firstpass_filename))

    config.catfile = firstpass_filename

    like_filename = config.redmapper_filename('like_catalog')
    if not os.path.isfile(like_filename):
        cat_like, members_like = run_likelihoods(config)
        output_cluster_catalog(cat_like, members_like, config, 'like', savemembers=False, withversion=False)
    else:
        logger.info("Likelihood file %s already present.  Skipping..." % (like_filename))

    config.catfile = like_filename

    perc_filename = config.redmapper_filename('final_catalog')
    if not os.path.isfile(perc_filename):
        cat_perc, members_perc = run_percolation(config)
        output_cluster_catalog(cat_perc, members_perc, config, 'final', savemembers=True, withversion=False)
    else:
        logger.info("Percolation file %s already present.  Skipping..." % (perc_filename))

    config.stop_file_logging()

def run_runcat_pixel_task(configfile, pixel, nside, path=None):
    """
    Run richness computation (runcat) on a single healpix pixel, for
    distributed runs.

    Parameters
    ----------
    configfile: `str`
       Configuration yaml filename.
    pixel: `int`
       Healpix pixel to run on.
    nside: `int`
       Healpix nside associated with pixel.
    path: `str`, optional
       Output path.  Default is None, use same absolute
       path as configfile.
    percolation_masking: `bool`, optional
       Do percolation masking when computing richnesses
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=path)

    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    config.check_files(check_zredfile=False, check_bkgfile=True, check_bkgfile_components=False, check_parfile=True, check_zlambdafile=True)

    # Compute the border size
    config.border = config.compute_border()

    config.hpix = [pixel]
    config.nside = nside
    config.outbase = '%s_%d_%05d' % (config.outbase, nside, pixel)

    # Do the run
    config.start_file_logging()

    logger.info("Running runcat on pixel %d" % (pixel))

    runcat_filename = config.redmapper_filename('runcat_catalog', withversion=True)
    if not os.path.isfile(runcat_filename):
        cat_runcat, members_runcat = run_catalog(config, do_percolation_masking=config.runcat_percolation_masking)
        output_cluster_catalog(cat_runcat, members_runcat, config, 'runcat', savemembers=True, withversion=True)

    config.stop_file_logging()

def run_zmask_pixel_task(configfile, pixel, nside, path=None):
    """
    Run redmapper zmask randoms on a single healpix pixel, for
    distributed runs.

    Parameters
    ----------
    configfile: `str`
       Configuration yaml filename.
    pixel: `int`
       Healpix pixel to run on.
    nside: `int`
       Healpix nside associated with pixel.
    path: `str`, optional
       Output path.  Default is None, use same absolute
       path as configfile.
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=path)

    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    config.check_files(check_zredfile=False, check_bkgfile=True,
                            check_parfile=True, check_randfile=True)

    # Compute the border size
    config.border = config.compute_border()

    config.hpix = [pixel]
    config.nside = nside
    config.outbase = '%s_%d_%05d' % (config.outbase, nside, pixel)

    config.start_file_logging()
    logger.info("Running zmask on pixel %d" % (pixel))

    filetype = 'randoms_zmask'
    filename = config.redmapper_filename(filetype + '_catalog')
    if not os.path.isfile(filename):
        cat, members = run_randoms_zmask(config)
        output_cluster_catalog(cat, members, config, filetype, savemembers=False, withversion=False)

    # All done
    config.stop_file_logging()

def run_zscan_pixel_task(configfile, pixel, nside, path=None):
    """Run redshift-scanning (zscan) on a single healpix pixel, for
    distributed runs.

    Parameters
    ----------
    configfile: `str`
       Configuration yaml filename.
    pixel: `int`
       Healpix pixel to run on.
    nside: `int`
       Healpix nside associated with pixel.
    path: `str`, optional
       Output path.  Default is None, use same absolute
       path as configfile.
    percolation_masking: `bool`, optional
       Do percolation masking when computing richnesses
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=path)

    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

    # Compute the border size
    config.border = config.compute_border()

    config.hpix = [pixel]
    config.nside = nside
    config.outbase = '%s_%d_%05d' % (config.outbase, nside, pixel)

    # Do the run
    config.start_file_logging()

    logger.info("Running zscan on pixel %d" % (pixel))

    zscan_filename = config.redmapper_filename('zscan_catalog', withversion=True)
    if not os.path.isfile(zscan_filename):
        cat_zscan, members_zscan = run_zscan(config)
        output_cluster_catalog(cat_zscan, members_zscan, config, 'zscan', savemembers=True, withversion=True)

    config.stop_file_logging()
