"""Functions to run redmagic, scanning over all pixels"""
import os
import numpy as np
import glob
import fitsio

from ..configuration import Configuration
from .redmagic_selector import read_redmagic_calibration, select_redmagic_galaxies
from ..catalog import Entry
from ..galaxy import GalaxyCatalog
from ..plotting import plot_spec_comparison, plot_redmagic_nz
from .redmagic_randoms import generate_redmagic_randoms
from ..volumelimit import get_volume_limit_areas
from ..logger import logger

def run_redmagic_task(configfile, path=None, modes=None, clobber=False, do_plots=True, n_randoms=None, rng=None):
    """
    Run redMaGiC selection over a full catalog.

    The modes are optional, if not specified all the modes
    will be run.

    Parameters
    ----------
    configfile: `str`
       Configuration yaml filename
    path: `str`, optional
       Output path.  Default is None, use same absolute
       path as configfile.
    modes: `list`, optional
       List of strings of modes to run
    clobber: `bool`, optional
       Overwrite any existing files.  Default is False.
    do_plots: `bool`, optional
       Make the output plots
    n_randoms: `int`, optional
       If None, then 10x the number of redmagic galaxies are generated.
       If 0, then no randoms are generated.
       If >0, then that many randoms are generated.
    rng : `np.random.RandomState`, optional
       Pre-set random number generator.  Default is None.
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=outpath)
    config.start_file_logging()

    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    # Prepare the redMaGiC selector
    selector_state = read_redmagic_calibration(config)

    if modes is None:
        modes = selector_state['modes']

    n_modes = len(modes)

    # Check if files exist, clobber is False
    filenames = [''] * n_modes
    for i, mode in enumerate(modes):
        filenames[i] = config.redmapper_filename('redmagic_%s' % (mode), withversion=True)

        if os.path.isfile(filenames[i]) and not clobber:
            raise RuntimeError("redMaGiC file %s already exists, and clobber is False" % (filenames[i]))

    # Loop over all pixels in the galaxy table
    tab = Entry.from_fits_file(config.galfile)

    started = [False] * n_modes

    logger.info("Making redMaGiC selection for %d modes and %d pixels" % (n_modes, tab.hpix.size))
    if config.has_truth:
        logger.info("Using truth information for zspec")

    for i, pix in enumerate(tab.hpix):
        gals = GalaxyCatalog.from_galfile(config.galfile,
                                          zredfile=config.zredfile,
                                          nside=tab.nside,
                                          hpix=pix,
                                          border=0.0,
                                          truth=config.has_truth)

        # Loop over all modes
        for j, mode in enumerate(modes):
            # Select the red galaxies
            red_gals, spec = select_redmagic_galaxies(selector_state, config, gals, mode, rng=rng)

            # Spool out redMaGiC galaxies
            if not started[j]:
                # write a new file (and overwrite if necessary, since we
                # already did the clobber check)
                red_gals.to_fits_file(filenames[j], clobber=True)
                started[j] = True
            else:
                with fitsio.FITS(filenames[j], mode='rw') as fits:
                    fits[1].append(red_gals._ndarray)

    # Load in catalogs and make plots!
    if do_plots:
        import matplotlib.pyplot as plt

        for j, mode in enumerate(modes):
            gals = GalaxyCatalog.from_fits_file(filenames[j])

            plot_redmagic_nz(config, gals, mode, selector_state['calib_data'][mode].etamin,
                                         selector_state['calib_data'][mode].n0,
                                         get_volume_limit_areas(selector_state['vlim_masks'][mode]),
                                         zrange=selector_state['calib_data'][mode].zrange,
                                         withversion=True,
                                         binsize=config.redmagic_calib_zbinsize)

            okspec, = np.where(gals.zspec > 0.0)
            if okspec.size > 0:
                fig = plot_spec_comparison(config, gals.zspec[okspec], gals.zredmagic[okspec],
                                           gals.zredmagic_e[okspec],
                                           name=r'z_{\mathrm{redmagic}}',
                                           title='%s: %3.1f-%02d' %
                                           (mode, selector_state['calib_data'][mode].etamin,
                                            int(selector_state['calib_data'][mode].n0)),
                                           figure_return=True)
                fig.savefig(config.redmapper_filename('redmagic_zspec_%s_%3.1f-%02d' %
                                                           (mode, selector_state['calib_data'][mode].etamin,
                                                            int(selector_state['calib_data'][mode].n0)),
                                                           paths=(config.plotpath,),
                                                           withversion=True,
                                                           filetype='png'))
                plt.close(fig)

    # Need a check on when to kick out
    if (n_randoms is not None and n_randoms == 0):
        # We are not generating randoms
        config.stop_file_logging()
        return

    # Random generation
    for j, mode in enumerate(modes):
        if selector_state['vlim_masks'][mode]['type'] == 'fixed':
            logger.info("Cannot construct randoms for %s, because we don't have geometry/depth info." % (mode))
            continue

        gals = GalaxyCatalog.from_fits_file(filenames[j])

        if n_randoms is None:
            _n_randoms = gals.size * 10
        else:
            _n_randoms = n_randoms

        randfile = config.redmapper_filename('redmagic_%s_randoms' % (mode), withversion=True)

        generate_redmagic_randoms(config, selector_state['vlim_masks'][mode], gals, _n_randoms, randfile, clobber=clobber, rng=rng)

    config.stop_file_logging()
