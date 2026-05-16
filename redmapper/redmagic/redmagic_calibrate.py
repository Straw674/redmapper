"""Classes for calibrating a redMaGiC parameter file."""
from collections import OrderedDict
import os
import numpy as np
import fitsio
import time
import scipy.optimize
import esutil
import copy

from ..configuration import Configuration
from ..fitters import fit_med_z
from ..redsequence import read_redsequence, redsequence_mstar
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry
from ..utilities import (decode_string, make_nodes, cubic_spline_compute_y2,
                         cubic_spline_interpolate, interpol, read_members)
from ..plotting import plot_spec_comparison, plot_redmagic_nz
from ..volumelimit import create_volume_limit_mask, create_volume_limit_mask_fixed, calc_zmax, get_volume_limit_areas
from .redmagic_selector import read_redmagic_calibration, select_redmagic_galaxies
from .redmagictask import run_redmagic_task
from ..logger import logger


def redmagic_cost(pars, state):
    """
    Cost function for redMaGiC parameters.
    
    Parameters
    ----------
    pars: `np.array`
        Float array of parameters
    state: `dict`
        Dictionary of data arrays and configuration
        
    Returns
    -------
    t: `float`
        Cost-function at pars
    """
    # chi2max is computed at the raw redshift
    y2 = cubic_spline_compute_y2(state['nodes'], pars)
    chi2max = np.clip(cubic_spline_interpolate(state['z'], state['nodes'], pars, y2), 0.1, state['maxchi'])

    gd, = np.where((state['chisq'] < chi2max) &
                   (state['refmag'] < (state['mstar'] - 2.5 * np.log10(state['etamin']))) &
                   (state['zredmagic'] < state['zmax']))

    if gd.size == 0:
        return 1e11

    # Histogram the galaxies into bins
    # This is done with the sampled redshifts
    h = esutil.stat.histogram(state['zredmagic_samp'][gd],
                              min=state['zrange'][0], max=state['zrange'][1] - 0.0001,
                              binsize=state['zbinsize'])

    # Compute density and error
    den = h.astype(np.float64) / state['volume']
    den_err = np.sqrt(state['n0'] * 1e-4 * state['volume']) / state['volume']

    # Compute cost function
    t = np.sum(((den - state['n0'] * 1e-4) / den_err)**2.)

    # Penalize (arbitrarily) if any of them are negative
    test, = np.where(pars < 0.1)
    if (test.size > 0):
        t += 10000.0

    return t

def fit_redmagic_parameters(p0_cval, state, biaspars=None, eratiopars=None, afterburner=False):
    """
    Fit the redMaGiC parameters.
    
    Parameters
    ----------
    p0_cval: `np.array`
       Float array of starting values for chi2(z) at nodes
    state: `dict`
       Data dict with redmagic properties
    biaspars: `np.array`, optional
       Parameters of afterburner bias.  Default is None
    eratiopars: `np.array`, optional
       Parameters of afterburner eratio.  Default is None
    afterburner: `bool`, optional
       Run the afterburner?  Default is False.

    Returns
    -------
    pars: `np.array`
       Float array of best-fit chi2(z) parameters
    """
    if state['ab_use'] is None and afterburner:
        raise RuntimeError("Must set afterburner_use if using the afterburner")

    if afterburner:
        if biaspars is None or eratiopars is None:
            raise RuntimeError("Must set biaspars, eratiopars if using the afterburner")

        # Set zredmagic based on the afterburner values
        y2 = cubic_spline_compute_y2(state['corrnodes'], biaspars)
        bias_vals = cubic_spline_interpolate(state['z'], state['corrnodes'], biaspars, y2)
        state['zredmagic'] = state['z'] - bias_vals
        state['zredmagic_samp'] = state['zsamp'] - bias_vals

        y2 = cubic_spline_compute_y2(state['corrnodes'], eratiopars)
        state['zredmagic_e'] = state['z_err'] * cubic_spline_interpolate(state['z'], state['corrnodes'], eratiopars, y2)
    else:
        state['zredmagic'][:] = state['z']
        state['zredmagic_e'][:] = state['z_err']
        state['zredmagic_samp'][:] = state['zsamp']

    # Compute here...
    state['mstar'] = redsequence_mstar(state['zredstr'], state['zredmagic'])

    # For whatever reason, the nelder-meade minimizer works better here.
    pars = scipy.optimize.fmin(redmagic_cost, p0_cval, args=(state,), disp=False, xtol=1e-8, ftol=1e-8)

    return pars

def fit_redmagic_bias_eratio(cval, p0_bias, p0_eratio, state):
    """
    Fit the bias and eratio afterburner parameters.
    
    Parameters
    ----------
    cval: `np.array`
       Float array of chi2(z) parameters
    p0_bias: `np.array`
       Float array of initial guess of bias parameters.
    p0_eratio: `np.array`
       Float array of initial guess of eratio parameters
    state: `dict`
       Dictionary with redmagic properties

    Returns
    -------
    pars_bias: `np.array`
       Float array of best-fit bias parameters
    pars_eratio: `np.array`
       Float array of best-fit eratio parameters
    """
    y2 = cubic_spline_compute_y2(state['nodes'], cval)
    chi2max = np.clip(cubic_spline_interpolate(state['z'], state['nodes'], cval, y2), 0.1, state['maxchi'])

    ab_mask = ((state['chisq'][state['ab_use']] < chi2max[state['ab_use']]) &
               (state['refmag'][state['ab_use']] < (state['mstar'][state['ab_use']] - 2.5 * np.log10(state['etamin']))))
    if state['zmax'].size > 1:
        # This is an array of zmax values
        ab_mask &= (state['zredmagic'][state['ab_use']] < state['zmax'][state['ab_use']])
    else:
        # This is a single value
        ab_mask &= (state['zredmagic'][state['ab_use']] < state['zmax'])

    ab_gd, = np.where(ab_mask)
    ab_gd = state['ab_use'][ab_gd]

    delta_gd = state['z'][ab_gd] - state['zcal'][ab_gd]

    pars_bias = fit_med_z(state['corrnodes'], state['z'][ab_gd], delta_gd, p0_bias, min_val=-0.1, max_val=0.1)

    y2 = cubic_spline_compute_y2(state['corrnodes'], pars_bias)
    delta_med_gd = cubic_spline_interpolate(state['z'][ab_gd], state['corrnodes'], pars_bias, y2)

    y = 1.4826 * np.abs(delta_gd - delta_med_gd) / state['z_err'][ab_gd]

    pars_eratio = fit_med_z(state['corrnodes'], state['z'][ab_gd], y, p0_eratio, min_val=0.5, max_val=1.5)

    return pars_bias, pars_eratio

def calibrate_redmagic(config, gals=None, do_run=True):
    """
    Run the redMaGiC calibration
    
    Parameters
    ----------
    config: `redmapper.Configuration` or `str`
       Configuration object or config filename
    gals: `redmapper.GalaxyCatalog`, optional
       Galaxy catalog to calibrate.  Default is None.  Used for testing.
    do_run: `bool`, optional
       Do the full run after calibration.  Default is True.
    """

    if not isinstance(config, Configuration):
        config = Configuration(config)
    import matplotlib.pyplot as plt

    # set gals for testing purposes...

    if gals is None:
        # make sure that we have pixelized file, zreds, etc.
        if not config.galfile_pixelized:
            raise RuntimeError("Code only runs with pixelized galfile.")

        if config.zredfile is None or not os.path.isfile(config.zredfile):
            raise RuntimeError("Must have zreds available.")

        if config.catfile is None or not os.path.isfile(config.catfile):
            raise RuntimeError("Must have a cluster catalog available.")

    # this is the number of calibration runs we have
    nruns = len(config.redmagic_etas)

    # check for vlim files

    vlim_masks = OrderedDict()
    vlim_areas = OrderedDict()

    if config.depthfile is not None and os.path.isfile(config.depthfile):
        # Best way: generate volume-limit mask from depth map
        for i, vlim_lstar in enumerate(config.redmagic_etas):
            logger.info("Reading/creating volume-limit mask from depth maps for %.2f" % (vlim_lstar))
            vlim_masks[config.redmagic_names[i]] = create_volume_limit_mask(config, vlim_lstar)
            vlim_areas[config.redmagic_names[i]] = get_volume_limit_areas(vlim_masks[config.redmagic_names[i]])
    elif config.maskfile is not None and os.path.isfile(config.maskfile):
        # Okay way: generate volume-limit mask from geometry mask
        for i, vlim_lstar in enumerate(config.redmagic_etas):
            logger.info("Reading/creating volume-limit mask from geometry map for %.2f" % (vlim_lstar))
            logger.info("NOTE: this is not optimal if the high redshift end is near the depth of the survey.")
            vlim_masks[config.redmagic_names[i]] = create_volume_limit_mask(config, vlim_lstar, use_geometry=True)
            vlim_areas[config.redmagic_names[i]] = get_volume_limit_areas(vlim_masks[config.redmagic_names[i]])
    else:
        # Just simulate it.
        for i, vlim_lstar in enumerate(config.redmagic_etas):
            logger.info("Simulating volume-limit mask for %.2f" % (vlim_lstar))
            logger.info("NOTE: You will not be able to create randoms without any geometry/depth information.")
            vlim_masks[config.redmagic_names[i]] = create_volume_limit_mask_fixed(config)
            vlim_areas[config.redmagic_names[i]] = get_volume_limit_areas(vlim_masks[config.redmagic_names[i]])

    # Note that the area is already scaled properly!

    if gals is None:
        # Read in galaxies with zreds
        gals = GalaxyCatalog.from_galfile(config.galfile,
                                          nside=config.nside,
                                          hpix=config.hpix,
                                          border=config.border,
                                          zredfile=config.zredfile,
                                          truth=config.redmagic_mock_truthspec)

    # Add redmagic fields
    gals.add_fields([('zuse', 'f4'),
                     ('zuse_e', 'f4'),
                     ('zspec', 'f4'),
                     ('zcal', 'f4'),
                     ('zcal_e', 'f4'),
                     ('zredmagic', 'f4'),
                     ('zredmagic_e', 'f4'),
                     ('zredmagic_samp', 'f4', config.zred_nsamp)])

    gals.zuse = gals.zred_uncorr
    gals.zuse_e = gals.zred_uncorr_e
    gals.zredmagic_samp = gals.zred_samp

    zredstr = read_redsequence(config.parfile, fine=True)

    mstar_init = redsequence_mstar(zredstr, gals.zuse)

    # This is the initial check of everything

    # modify zrange for even bins, including cushion
    cost_zrange = np.copy(config.redmagic_zrange)
    cost_zrange = [config.redmagic_zrange[0] + config.redmagic_calib_redshift_buffer,
                   config.redmagic_zrange[1] - config.redmagic_calib_redshift_buffer]
    nbin = np.ceil((cost_zrange[1] - cost_zrange[0]) / config.redmagic_calib_zbinsize).astype(np.int32)
    cost_zrange[1] = nbin * config.redmagic_calib_zbinsize + cost_zrange[0]

    lstar_cushion = 0.05
    z_cushion = 0.05

    # Cut input galaxies
    cut_zrange = [cost_zrange[0] - z_cushion - config.redmagic_calib_redshift_buffer,
                  cost_zrange[1] + z_cushion + config.redmagic_calib_redshift_buffer]
    minlstar = np.clip(np.min(config.redmagic_etas) - lstar_cushion, 0.1, None)

    use, = np.where((gals.zuse > cut_zrange[0]) & (gals.zuse < cut_zrange[1]) &
                    (gals.chisq < config.redmagic_calib_chisqcut) &
                    (gals.refmag < (mstar_init - 2.5*np.log10(minlstar))))

    if use.size == 0:
        raise RuntimeError("No galaxies in redshift range/chisq range/eta range.")

    # This selects all *possible* redmagic galaxies
    gals = gals[use]
    mstar_init = mstar_init[use]

    # Run the galaxy cleaner
    # FIXME: implement cleaner

    # match to spectra
    if not config.redmagic_mock_truthspec:
        logger.info("Reading and matching spectra...")

        spec = Catalog.from_fits_file(config.specfile)
        use, = np.where(spec.z_err < config.calib_spec_max_zerr)
        spec = spec[use]

        i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3./3600., maxmatch=1)
        gals.zspec[i1] = spec.z[i0]
    else:
        logger.info("Using truth spectra for reference...")
        gals.zspec = gals.ztrue

    # Match to cluster catalog

    cat = Catalog.from_fits_file(config.catfile)
    mem = read_members(config.catfile)

    mem.add_fields([('z_err', 'f4')])

    a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
    mem.z_err[b] = cat.z_lambda_e[a]

    use, = np.where(mem.p > config.calib_corr_pcut)
    mem = mem[use]

    gals.zcal[:] = -1.0
    gals.zcal_e[:] = -1.0

    a, b = esutil.numpy_util.match(gals.id, mem.id)
    gals.zcal[a] = mem.z[b]
    gals.zcal_e[a] = mem.z_err[b]

    # Clear out members
    del mem

    config.redmagicfile = config.redmapper_filename('redmagic_calib')

    # loop over cuts...

    for i in range(nruns):
        logger.info("Working on %s: etamin = %.3f, n0 = %.3f" % (config.redmagic_names[i], config.redmagic_etas[i], config.redmagic_n0s[i]))

        # This is the full redshift range for redshift selection and plots
        redmagic_zrange = [config.redmagic_zrange[0],
                           config.redmagic_zmaxes[i]]

        # This is the redshift range where the cost function is computed
        cost_zrange = [redmagic_zrange[0] + config.redmagic_calib_redshift_buffer,
                       redmagic_zrange[1] - config.redmagic_calib_redshift_buffer]
        # And make sure that we have even sized bins
        nbin = np.ceil((cost_zrange[1] - cost_zrange[0]) / config.redmagic_calib_zbinsize).astype(np.int32)
        cost_zrange[1] = nbin * config.redmagic_calib_zbinsize + cost_zrange[0]

        # Plot over the full range
        plot_zrange = [redmagic_zrange[0], redmagic_zrange[1]]
        nbin_plot = np.ceil((plot_zrange[1] - plot_zrange[0]) / config.redmagic_calib_zbinsize).astype(np.int32)
        plot_zrange[1] = nbin_plot * config.redmagic_calib_zbinsize + plot_zrange[0]

        # The nodes only cover the range of the cost function
        nodes = make_nodes(cost_zrange, config.redmagic_calib_nodesize)
        corrnodes = make_nodes(cost_zrange, config.redmagic_calib_corr_nodesize)

        # Prepare calibration structure
        vmaskfile = ''
        if vlim_masks[config.redmagic_names[i]]['type'] == 'sparse':
            vmaskfile = vlim_masks[config.redmagic_names[i]]['vlimfile']

        calstr = Entry(np.zeros(1, dtype=[('zrange', 'f4', 2),
                                          ('cost_zrange', 'f4', 2),
                                          ('lstar_cushion', 'f4'),
                                          ('z_cushion', 'f4'),
                                          ('name', 'S%d' % (len(config.redmagic_names[i]) + 1)),
                                          ('maxchi', 'f4'),
                                          ('nodes', 'f8', nodes.size),
                                          ('etamin', 'f8'),
                                          ('n0', 'f8'),
                                          ('cmax', 'f8', nodes.size),
                                          ('corrnodes', 'f8', corrnodes.size),
                                          ('run_afterburner', 'i2'),
                                          ('apply_afterburner', 'i2'),
                                          ('buffer', 'f4'),
                                          ('bias', 'f8', corrnodes.size),
                                          ('eratio', 'f8', corrnodes.size),
                                          ('vmaskfile', 'S%d' % (len(vmaskfile) + 1))]))

        calstr.zrange[:] = redmagic_zrange
        calstr.cost_zrange[:] = cost_zrange
        calstr.lstar_cushion = lstar_cushion
        calstr.z_cushion = z_cushion
        calstr.name = config.redmagic_names[i]
        calstr.maxchi = config.redmagic_calib_chisqcut
        calstr.nodes[:] = nodes
        calstr.corrnodes[:] = corrnodes
        calstr.etamin = config.redmagic_etas[i]
        calstr.n0 = config.redmagic_n0s[i]
        calstr.vmaskfile = vmaskfile
        calstr.run_afterburner = config.redmagic_run_afterburner
        calstr.apply_afterburner = config.redmagic_apply_afterburner_zsamp
        calstr.buffer = config.redmagic_calib_redshift_buffer

        # Take the first sample of each galaxy
        zsamp = gals.zred_samp[:, 0].copy()

        # Initial histogram, using sampled redshifts
        # This histogram isn't actually used for anything except to confirm the size
        h = esutil.stat.histogram(zsamp,
                                  min=cost_zrange[0], max=cost_zrange[1] - 0.0001,
                                  binsize=config.redmagic_calib_zbinsize)
        zbins = np.arange(h.size, dtype=np.float64) * config.redmagic_calib_zbinsize + cost_zrange[0] + config.redmagic_calib_zbinsize / 2.

        etamin_ref = np.clip(config.redmagic_etas[i] - lstar_cushion, 0.1, None)

        # These are possible redmagic galaxies for this selection
        red_poss, = np.where(gals.refmag < (mstar_init - 2.5*np.log10(etamin_ref)))

        # Determine which of the galaxies to use in the afterburner
        gd, = np.where(gals.zcal[red_poss] > 0.0)
        ntrain = int(config.redmagic_calib_fractrain * gd.size)
        r = np.random.random(gd.size)
        st = np.argsort(r)
        afterburner_use = gd[st[0: ntrain]]

        # Compute the volume
        vmask = vlim_masks[config.redmagic_names[i]]
        astr = vlim_areas[config.redmagic_names[i]]

        aind = np.searchsorted(astr.z, zbins)
        z_areas = astr.area[aind]

        volume = np.zeros(zbins.size)
        for j in range(zbins.size):
            volume[j] = (config.cosmo.V(zbins[j] - config.redmagic_calib_zbinsize/2.,
                                             zbins[j] + config.redmagic_calib_zbinsize/2.) *
                              (z_areas[j] / 41252.961))

        zmax = calc_zmax(vmask, gals.ra[red_poss], gals.dec[red_poss])

        # Compute the density based on the histogram above and the volume
        dens = h.astype(np.float64) / volume

        if not config.redmagic_use_constchi:
            bad, = np.where(dens < config.redmagic_n0s[i] * 1e-4)
            if bad.size > 0:
                logger.info("Warning: not enough galaxies at z=%s" % (zbins[bad].__str__()))

        # get starting values
        cmaxvals = np.zeros(nodes.size)

        if config.redmagic_use_constchi:
            cmaxvals[:] = config.redmagic_constchis[i]
            _constchi = config.redmagic_constchis[i]
        else:
            _constchi = None
            aind = np.searchsorted(astr.z, nodes)
            test_areas = astr.area[aind]

            test_vol = np.zeros(nodes.size)
            for j in range(nodes.size):
                test_vol[j] = (config.cosmo.V(nodes[j] - config.redmagic_calib_zbinsize/2.,
                                                   nodes[j] + config.redmagic_calib_zbinsize/2.) *
                               (test_areas[j] / 41252.961))

            for j in range(nodes.size):
                zrange = [nodes[j] - config.redmagic_calib_zbinsize/2.,
                          nodes[j] + config.redmagic_calib_zbinsize/2.]
                if j == 0:
                    zrange = [nodes[j],
                              nodes[j] + config.redmagic_calib_zbinsize]
                elif j == nodes.size - 1:
                    zrange = [nodes[j] - config.redmagic_calib_zbinsize,
                              nodes[j]]

                u, = np.where((gals.zuse[red_poss] > zrange[0]) &
                              (gals.zuse[red_poss] < zrange[1]))

                st = np.argsort(gals.chisq[red_poss[u]])
                test_den = np.arange(u.size) / test_vol[j]
                ind = np.clip(np.searchsorted(test_den, config.redmagic_n0s[i] * 1e-4), 0, test_den.size - 1)
                cmaxvals[j] = gals.chisq[red_poss[u[st[ind]]]]

        # setup state for fitting
        z = gals.zuse[red_poss].copy()
        state = {
            'nodes': np.atleast_1d(nodes).astype(np.float64),
            'corrnodes': np.atleast_1d(corrnodes).astype(np.float64),
            'z': np.atleast_1d(z).astype(np.float64),
            'z_err': np.atleast_1d(gals.zuse_e[red_poss]).astype(np.float64),
            'zredmagic': np.atleast_1d(z).astype(np.float64),
            'zredmagic_e': np.atleast_1d(gals.zuse_e[red_poss]).astype(np.float64),
            'chisq': np.atleast_1d(gals.chisq[red_poss]).astype(np.float64),
            'mstar': np.atleast_1d(mstar_init[red_poss]).astype(np.float64),
            'zcal': np.atleast_1d(gals.zcal[red_poss]).astype(np.float64),
            'zcal_err': np.atleast_1d(gals.zcal_e[red_poss]).astype(np.float64),
            'refmag': np.atleast_1d(gals.refmag[red_poss]).astype(np.float64),
            'zsamp': np.atleast_1d(zsamp[red_poss]).astype(np.float64),
            'zredmagic_samp': np.atleast_1d(zsamp[red_poss]).astype(np.float64),
            'zmax': np.atleast_1d(zmax).astype(np.float64),
            'etamin': config.redmagic_etas[i],
            'n0': config.redmagic_n0s[i],
            'volume': volume,
            'zrange': cost_zrange,
            'zbinsize': config.redmagic_calib_zbinsize,
            'zredstr': zredstr,
            'ab_use': afterburner_use,
            'ab_apply': config.redmagic_apply_afterburner_zsamp,
            'constchi': _constchi,
            'maxchi': config.redmagic_calib_chisqcut
        }

        if not config.redmagic_use_constchi:
            logger.info("Fitting first pass...")
            cmaxvals = fit_redmagic_parameters(cmaxvals, state, afterburner=False)

        # default is no bias or eratio
        biasvals = np.zeros(corrnodes.size)
        eratiovals = np.ones(corrnodes.size)

        # run with afterburner
        if config.redmagic_run_afterburner:
            logger.info("Fitting with afterburner...")

            if not config.redmagic_use_constchi:
                # Let's look at 5 iterations here...
                for k in range(5):
                    logger.info("Afterburner iteration %d" % (k))
                    # Fit the bias and eratio...
                    biasvals, eratiovals = fit_redmagic_bias_eratio(cmaxvals, biasvals, eratiovals, state)
                    cmaxvals = fit_redmagic_parameters(cmaxvals, state, biaspars=biasvals, eratiopars=eratiovals, afterburner=True)

            # And a last fit of bias/eratio
            biasvals, eratiovals = fit_redmagic_bias_eratio(cmaxvals, biasvals, eratiovals, state)

        rmfitter = None

        # Record the calibrations
        calstr.cmax[:] = cmaxvals
        calstr.bias[:] = biasvals
        calstr.eratio[:] = eratiovals

        calstr.to_fits_file(config.redmagicfile, clobber=False, extname=config.redmagic_names[i])

        # Do the redmagic selection on our calibration galaxies
        selector_state = read_redmagic_calibration(config, vlim_masks=vlim_masks)
        redgals, gd, _ = select_redmagic_galaxies(selector_state, config, gals, config.redmagic_names[i], return_indices=True)
        gals.zredmagic[gd] = redgals.zredmagic
        gals.zredmagic_e[gd] = redgals.zredmagic_e
        gals.zredmagic_samp[gd, :] = redgals.zredmagic_samp

        # make pretty plots
        name = calstr.name
        if hasattr(name, 'decode'):
            name = decode_string(name)
        plot_redmagic_nz(config, gals[gd], name, calstr.etamin, calstr.n0,
                                     vlim_areas[config.redmagic_names[i]],
                                     zrange=plot_zrange,
                                     calib_zrange=cost_zrange,
                                     extraname='calib',
                                     binsize=config.redmagic_calib_zbinsize)

        # This stringification can be streamlined, I think.

        # spectroscopic comparison
        okspec, = np.where(gals.zspec[gd] > 0.0)
        if okspec.size > 10:
            fig = plot_spec_comparison(config, gals.zspec[gd[okspec]], gals.zredmagic[gd[okspec]],
                                       gals.zredmagic_e[gd[okspec]],
                                       name=r'z_{\mathrm{redmagic}}',
                                       title='%s: %3.1f-%02d' %
                                       (name, calstr.etamin, int(calstr.n0)),
                                       calib_zrange=cost_zrange,
                                       figure_return=True)
            fig.savefig(config.redmapper_filename('redmagic_calib_zspec_%s_%3.1f-%02d' %
                                                       (name, calstr.etamin,
                                                        int(calstr.n0)),
                                                       paths=(config.plotpath,),
                                                       filetype='png'))
            plt.close(fig)

        okcal, = np.where(gals.zcal[gd] > 0.0)
        if okcal.size > 10:
            name = calstr.name
            if hasattr(name, 'decode'):
                name = decode_string(name)
            fig = plot_spec_comparison(config, gals.zcal[gd[okcal]], gals.zredmagic[gd[okcal]],
                                       gals.zredmagic_e[gd[okcal]],
                                       name=r'z_{\mathrm{redmagic}}',
                                       specname=r'z_{\mathrm{cal}}',
                                       title='%s: %3.1f-%02d' %
                                       (name, calstr.etamin, int(calstr.n0)),
                                       calib_zrange=cost_zrange,
                                       figure_return=True)
            fig.savefig(config.redmapper_filename('redmagic_calib_zcal_%s_%3.1f-%02d' %
                                                       (name, calstr.etamin,
                                                        int(calstr.n0)),
                                                       paths=(config.plotpath,),
                                                       filetype='png'))
            plt.close(fig)


    # Output config file here!
    runfile = config.configfile.replace('cal', 'run')
    if runfile == config.configfile:
        # We didn't find anything to replace
        parts = config.configfile.split('.y')
        runfile = parts[0] + '_run.yaml'

    runfile_out = os.path.join(config.outpath, runfile)

    # Reset the pixel to do the full sky when we create a catalog.
    config.hpix = []
    config.nside = 0
    config.area = None
    config.output_yaml(runfile_out)

    if do_run:
        # Call the redmagic runner here.
        run_redmagic_task(runfile)

    return runfile_out
