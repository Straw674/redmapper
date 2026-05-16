"""Functional interface to run the full red-sequence calibration
"""
import os
import numpy as np
import fitsio
import copy
import re

from ..configuration import Configuration, Config
from ..color_background import generate_color_background
from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from .selectspecred import select_spec_red_galaxies_wrapper
from .selectspecseeds import select_spec_seeds_wrapper
from .redsequencecal import calibrate_red_sequence
from .centeringcal import calibrate_wcen
from .zlambdacal import calibrate_zlambda
from .prepmembers import prep_members
from ..zred_runner import run_zred_catalog, run_zred_pixels
from ..background import generate_background, generate_zred_background
from ..redmapper_run import redmapper_run
from ..zlambda import read_zlambda_correction, apply_zlambda_correction
from ..plotting import plot_spec_comparison
from ..mask import get_mask, gen_maskgals
from ..run_colormem import run_colormem
from ..utilities import getMemoryString
from .._version import __version__
from ..logger import logger

def calibrate_redmapper(conf):
    """
    Run the full red-sequence calibration.

    Parameters
    ----------
    conf: `str` or `redmapper.Config`
       Configuration yaml file or configuration object
    """
    if not isinstance(conf, Config):
        config = Configuration(conf)
    else:
        config = conf

    logger.info("Calibrating with version %s" % (__version__))
    rng = np.random.RandomState(seed=config.randomseed)

    # 1. Select the red galaxies to start
    config.redgalfile = config.redmapper_filename('zspec_redgals')
    config.redgalmodelfile = config.redmapper_filename('zspec_redgals_model')

    if os.path.isfile(config.redgalfile):
        logger.info("%s already there.  Skipping..." % (config.redgalfile))
    else:
        logger.info("Selecting red galaxies from spectra...")
        select_spec_red_galaxies_wrapper(config)

    # 2. Make a color background
    config.bkgfile_color = config.redmapper_filename('bkg_color')

    if os.path.isfile(config.bkgfile_color):
        logger.info("%s already there.  Skipping..." % (config.bkgfile_color))
    else:
        logger.info("Constructing color background...")
        generate_color_background(config)

    # 3. Generate maskgals
    config.maskgalfile = config.redmapper_filename('maskgals')

    if os.path.isfile(config.maskgalfile):
        logger.info("%s already there.  Skipping..." % (config.maskgalfile))
    else:
        logger.info("Constructing maskgals...")
        gen_maskgals(config, config.maskgalfile, rng=rng)

    # 4. Do the color-lambda training
    config.zmemfile = config.redmapper_filename('iter0_colormem_pgt%4.2f_lamgt%02d' % 
                                                (config.calib_pcut, config.calib_colormem_minlambda))

    if os.path.isfile(config.zmemfile):
        logger.info("%s already there.  Skipping..." % (config.zmemfile))
    else:
        logger.info("Doing color-lambda training...")
        cat, members = run_colormem(config)
        use, = np.where(members.pcol > config.calib_pcut)
        savemem = members[use]
        savemem.to_fits_file(config.zmemfile)
        cat.to_fits_file(config.redmapper_filename('colorcat'))

    # 5. Generate the spec seed file
    config.seedfile = config.redmapper_filename('specseeds_train')

    if os.path.isfile(config.seedfile):
        logger.info("%s already there.  Skipping..." % (config.seedfile))
    else:
        logger.info("Generating spectroscopic seeds (training spec)...")
        select_spec_seeds_wrapper(config, usetrain=True)

    # Save original outbase for iterations
    outbase_orig = config.outbase

    # 6. Run calibration iterations
    for iteration in range(1, config.calib_niter + 1):
        # Run the calibration iteration
        _run_calibration_iteration(config, iteration, rng)

        # Clean out the members
        redmapper_name = 'zmem_pgt%4.2f_lamgt%02d' % (config.calib_pcut, int(config.calib_minlambda))
        config.zmemfile = config.redmapper_filename(redmapper_name)
        if os.path.isfile(config.zmemfile):
            logger.info("%s already there.  Skipping..." % (config.zmemfile))
        else:
            logger.info("Preparing members for next calibration...")
            prep_members(config, 'z_init', rng=rng)

        # Reset outbase for next iteration
        config.outbase = outbase_orig

        if iteration == 1:
            # If this is the first iteration, generate a new seedfile
            new_seedfile = config.redmapper_filename('cut_specseeds')
            if os.path.isfile(new_seedfile):
                logger.info("%s already there.  Skipping..." % (new_seedfile))
            else:
                logger.info("Generating cut specseeds...")
                seeds = GalaxyCatalog.from_fits_file(config.seedfile)
                cat = GalaxyCatalog.from_fits_file(config.catfile)
                use, = np.where(cat.Lambda > config.percolation_minlambda)
                i0, i1, dd = seeds.match_many(cat.ra[use], cat.dec[use], 0.5/3600., maxmatch=1)
                seeds.to_fits_file(new_seedfile, indices=i1)
            config.seedfile = new_seedfile

    # 7. Prep for final iteration
    config.seedfile = config.redmapper_filename('specseeds')

    if os.path.isfile(config.seedfile):
        logger.info("%s already there.  Skipping..." % (config.seedfile))
    else:
        logger.info("Generating spectroscopic seeds (full spec)...")
        select_spec_seeds_wrapper(config, usetrain=False)

    _run_final_calibration_iteration(config, config.calib_niter)

    # Reset outbase after final iteration (it was changed to *_iterNb inside)
    config.outbase = outbase_orig

    # 8. Output a configuration file
    new_bkgfile, new_zreds = _output_calibration_config(config)

    # 9. Generate a full background if needed
    if new_bkgfile:
        logger.info("Running full background...")
        config.hpix = []
        config.nside = 0
        config.area = config.galfile_area
        generate_background(config, deepmode=True)
        logger.info("Remember to run zreds and zred background before running the full cluster finder.")
    else:
        if new_zreds:
            logger.info("Remember to run zreds before running the full cluster finder.  No need to recompute the background.")
        else:
            logger.info("Calibration done on full footprint, so background and zreds are already available.")

def _run_calibration_iteration(config, iteration, rng):
    """
    Internal helper to run a single iteration of the redmapper calibration.
    """
    # Generate the name of the parfile
    outbase_orig = config.outbase
    config.outbase = '%s_iter%d' % (outbase_orig, iteration)
    config.parfile = config.redmapper_filename('pars')

    # 1. Run the red sequence calibration
    if os.path.isfile(config.parfile):
        logger.info("%s already there.  Skipping..." % (config.parfile))
    else:
        logger.info("Running red sequence calibration...")
        calibrate_red_sequence(config, config.zmemfile, rng=rng)

    # 2. Compute zreds
    if config.galfile_pixelized:
        config.zredfile = config.redmapper_filename('zreds_master_table', paths=(config.outbase,))
    else:
        config.zredfile = config.redmapper_filename('zreds')

    if os.path.isfile(config.zredfile):
        logger.info("%s already there.  Skipping..." % (config.zredfile))
    else:
        logger.info("Computing zreds for all galaxies in the training region...")
        if config.galfile_pixelized:
            run_zred_pixels(config)
        else:
            run_zred_catalog(config, config.galfile, config.zredfile)

    # 3. Compute the chisq background
    config.bkgfile = config.redmapper_filename('bkg')
    calc_bkg = False
    calc_zred_bkg = False
    if not os.path.isfile(config.bkgfile):
        calc_bkg = True
        calc_zred_bkg = True
    else:
        with fitsio.FITS(config.bkgfile) as fits:
            extnames = [ext.get_extname() for ext in fits[1: ]]
            if 'CHISQBKG' not in extnames:
                calc_bkg = True
            else:
                logger.info("Found CHISQBKG in %s.  Skipping..." % (config.bkgfile))
            if 'ZREDBKG' not in extnames:
                calc_zred_bkg = True
            else:
                logger.info("Found ZREDBKG in %s.  Skipping..." % (config.bkgfile))

    if calc_bkg:
        logger.info("Generating chisq background...")
        generate_background(config)
    if calc_zred_bkg:
        logger.info("Generating zred background...")
        generate_zred_background(config)

    # 4. Set the centering function
    centerclass_orig = config.centerclass
    if iteration == 1:
        config.centerclass = config.firstpass_centerclass
    else:
        config.centerclass = 'CenteringWcenZred'

    # 5. Generate the zreds for the specseeds
    iter_seedfile = config.redmapper_filename('specseeds')
    if os.path.isfile(iter_seedfile):
        logger.info('%s already there.  Skipping...' % (iter_seedfile))
    else:
        logger.info("Generating iteration seedfile...")
        seedzredfile = config.redmapper_filename('specseeds_zreds')
        run_zred_catalog(config, config.seedfile, seedzredfile)
        seeds = Catalog.from_fits_file(config.seedfile, ext=1)
        zreds = Catalog.from_fits_file(seedzredfile, ext=1)
        seeds.zred = zreds.zred
        seeds.zred_e = zreds.zred_e
        seeds.zred_chisq = zreds.chisq
        seeds.to_fits_file(iter_seedfile)

    # 6. Run the cluster finder in specmode
    finalfile = config.redmapper_filename('final')
    if os.path.isfile(finalfile):
        logger.info('%s already there.  Skipping...' % (finalfile))
    else:
        logger.info("Running redmapper in specmode with seeds...")
        config.zlambdafile = None
        catfile, likefile = redmapper_run(config, specmode=True, keepz=True, 
                                          consolidate_like=True, seedfile=iter_seedfile, cleaninput=True)
        if catfile != finalfile:
            raise RuntimeError("The output catfile %s should be the same as finalfile %s" % (catfile, finalfile))

    # 7. Calibrate random and satellite w functions if first iteration
    if iteration == 1:
        sublikefile = config.redmapper_filename('sub_like')
        if not os.path.isfile(sublikefile):
            lcat = GalaxyCatalog.from_fits_file(config.redmapper_filename('like'))
            pcat = GalaxyCatalog.from_fits_file(finalfile)
            use, = np.where(pcat.Lambda > config.percolation_minlambda)
            i0, i1, dd = lcat.match_many(pcat.ra[use], pcat.dec[use], 0.5/3600., maxmatch=1)
            sublcat = lcat[i1]
            sublcat.to_fits_file(sublikefile)

        outbase_iter = config.outbase
        config.outbase = '%s_rand' % (outbase_iter)
        catfile_for_rand_calib = config.redmapper_filename('final')
        if os.path.isfile(catfile_for_rand_calib):
            logger.info('%s already there.  Skipping...' % (catfile_for_rand_calib))
        else:
            logger.info("Running percolation for random centers...")
            config.catfile = sublikefile
            config.centerclass = 'CenteringRandom'
            redmapper_run(config, check=True, percolation_only=True, keepz=True, cleaninput=True)

        config.outbase = '%s_randsat' % (outbase_iter)
        catfile_for_randsat_calib = config.redmapper_filename('final')
        if os.path.isfile(catfile_for_randsat_calib):
            logger.info('%s already there.  Skipping...' % (catfile_for_randsat_calib))
        else:
            logger.info("Running percolation for random satellite centers...")
            config.catfile = sublikefile
            config.centerclass = 'CenteringRandomSatellite'
            redmapper_run(config, check=True, percolation_only=True, keepz=True, cleaninput=True)
        
        config.outbase = outbase_iter
    else:
        catfile_for_rand_calib = None
        catfile_for_randsat_calib = None

    # 8. Calibrate wcen
    config.centerclass = centerclass_orig
    config.catfile = finalfile
    config.wcenfile = config.redmapper_filename('wcen')
    if os.path.isfile(config.wcenfile):
        logger.info('%s already there.  Skipping...' % (config.wcenfile))
    else:
        logger.info("Calibrating Wcen")
        calibrate_wcen(config, iteration,
                       randcatfile=catfile_for_rand_calib,
                       randsatcatfile=catfile_for_randsat_calib,
                       rng=rng)

    # We need a way to set wcen vals in config dictionary
    from ..configuration import get_wcen_vals
    wcen_vals = get_wcen_vals(config.wcenfile)
    for key, value in wcen_vals.items():
        config[key] = value

    # 9. Calibrate zlambda correction
    config.zlambdafile = config.redmapper_filename('zlambda')
    if os.path.isfile(config.zlambdafile):
        logger.info('%s already there.  Skipping...' % (config.zlambdafile))
    else:
        logger.info("Calibrating zlambda corrections...")
        calibrate_zlambda(config, corrslope=False)

    # 10. Make pretty plots
    plot_filename = config.redmapper_filename('zspec', paths=(config.plotpath,), filetype='png')
    if os.path.isfile(plot_filename):
        logger.info("%s already there.  Skipping..." % (plot_filename))
    else:
        logger.info("Correcting redshifts and making spec plot...")
        cat = Catalog.from_fits_file(config.catfile)
        use, = np.where(cat.Lambda > config.calib_minlambda)
        cat = cat[use]
        zlambda_corr_data = read_zlambda_correction(parfile=config.zlambdafile,
                                                    zlambda_pivot=config.zlambda_pivot)
        for cluster in cat:
            zlam, zlam_e = apply_zlambda_correction(zlambda_corr_data, cluster.Lambda, 
                                                    cluster.z_lambda, cluster.z_lambda_e)
            cluster.z_lambda = zlam
            cluster.z_lambda_e = zlam_e

        test, = np.where(cat.z_lambda < 0.0)
        if test.size > 0:
            raise RuntimeError("z_lambda correction calibration totally failed yielding negative redshifts.")
        
        use, = np.where(cat.Lambda > config.calib_zlambda_minlambda)
        plot_spec_comparison(config, cat.z_spec_init[use], cat.z_lambda[use], cat.z_lambda_e[use], 
                             title=config.outbase)

def _run_final_calibration_iteration(config, iteration):
    """
    Internal helper to run the final iteration of the red-sequence calibration.
    """
    outbase_orig = config.outbase
    config.outbase = '%s_iter%db' % (outbase_orig, iteration)

    iter_seedfile = config.redmapper_filename('specseeds')
    if os.path.isfile(iter_seedfile):
        logger.info('%s already there.  Skipping...'  % (iter_seedfile))
    else:
        logger.info("Creating final iteration seeds...")
        seedzredfile = config.redmapper_filename('specseeds_zreds')
        run_zred_catalog(config, config.seedfile, seedzredfile)
        seeds = Catalog.from_fits_file(config.seedfile, ext=1)
        zreds = Catalog.from_fits_file(seedzredfile, ext=1)
        seeds.zred = zreds.zred
        seeds.zred_e = zreds.zred_e
        seeds.zred_chisq = zreds.chisq
        seeds.to_fits_file(iter_seedfile)

    finalfile = config.redmapper_filename('final')
    if os.path.isfile(finalfile):
        logger.info('%s already there.  Skipping...' % (finalfile))
    else:
        logger.info("Doing final iteration run")
        catfile = redmapper_run(config, seedfile=iter_seedfile, cleaninput=True)
        if catfile != finalfile:
            raise RuntimeError("The output catfile %s should be the same as finalfile %s" % (catfile, finalfile))

    config.catfile = finalfile
    plot_filename = config.redmapper_filename('zspec', paths=(config.plotpath,), filetype='png')
    if os.path.isfile(plot_filename):
        logger.info("%s already there.  Skipping..." % (plot_filename))
    else:
        logger.info("Making final iteration spec plot...")
        cat = Catalog.from_fits_file(config.catfile)
        use, = np.where(cat.Lambda > config.calib_zlambda_minlambda)
        cat = cat[use]
        plot_spec_comparison(config, cat.z_spec_init, cat.z_lambda, cat.z_lambda_e, title=config.outbase)

def _output_calibration_config(config):
    """
    Internal helper to output a configuration yaml file.
    """
    new_zreds = False
    new_bkgfile = False

    calpath = os.path.abspath(config.outpath)
    calparent = os.path.normpath(os.path.join(calpath, os.pardir))
    calpath_only = os.path.basename(os.path.normpath(calpath))

    if calpath_only == 'cal':
        runpath_only = 'run'
    elif 'cal_' in calpath_only:
        runpath_only = calpath_only.replace('cal_', 'run_')
    elif '_cal' in calpath_only:
        runpath_only = calpath_only.replace('_cal', '_run')
    else:
        runpath_only = '%s_run' % (calpath_only)

    runpath = os.path.join(calparent, runpath_only)
    if not os.path.isdir(runpath):
        os.makedirs(runpath)

    config.galfile = os.path.abspath(config.galfile)
    config.specfile = os.path.abspath(config.specfile)

    outbase_cal = config.outbase
    iterstr = '%s_iter%d' % (outbase_cal, config.calib_niter)

    if '_cal' in outbase_cal:
        outbase_run = outbase_cal.replace('_cal', '_run')
    else:
        outbase_run = '%s_run' % (outbase_cal)

    config.outbase = outbase_run
    config.parfile = os.path.abspath(os.path.join(config.outpath, '%s_pars.fit' % (iterstr)))
    config.bkgfile = os.path.abspath(os.path.join(calpath, '%s_bkg.fit' % (iterstr)))

    if config.nside == 0:
        config.zredfile = os.path.abspath(os.path.join(calpath, '%s' % (iterstr), 
                                                        '%s_zreds_master_table.fit' % (iterstr)))
    else:
        new_zreds = True
        galfile_base = os.path.basename(config.galfile)
        zredfile = galfile_base.replace('_master', '_zreds_master')
        config.zredfile = os.path.abspath(os.path.join(runpath, 'zreds', zredfile))
        if config.calib_make_full_bkg:
            new_bkgfile = True
            config.bkgfile = os.path.abspath(os.path.join(runpath, '%s_bkg.fit' % (outbase_run)))

    config.zlambdafile = os.path.abspath(os.path.join(calpath, '%s_zlambda.fit' % (iterstr)))
    config.wcenfile = os.path.abspath(os.path.join(calpath, '%s_wcen.fit' % (iterstr)))
    config.bkgfile_color = os.path.abspath(config.bkgfile_color)
    config.catfile = None
    config.maskgalfile = os.path.abspath(config.maskgalfile)
    config.redgalfile = os.path.abspath(config.redgalfile)
    config.redgalmodelfile = os.path.abspath(config.redgalmodelfile)
    config.seedfile = None
    config.zmemfile = None
    config.nside = 0
    config.hpix = []
    config.border = 0.0
    config.area = None

    config.output_yaml(os.path.join(runpath, 'run_default.yml'))

    return (new_bkgfile, new_zreds)

