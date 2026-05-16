"""Functions for making a redMaGiC galaxy selection."""
from collections import OrderedDict
import os
import numpy as np

import fitsio
import time
import scipy.optimize
import esutil

from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from ..configuration import Configuration
from ..volumelimit import create_volume_limit_mask, create_volume_limit_mask_fixed, calc_zmax
from ..redsequence import read_redsequence, redsequence_mstar
from ..utilities import decode_string, cubic_spline_compute_y2, cubic_spline_interpolate
from ..logger import logger

def read_redmagic_calibration(config, vlim_masks=None):
    """
    Read redMaGiC calibration data and setup masks.

    Parameters
    ----------
    config: `redmapper.Configuration` or `str`
       Configuration object or config filename
    vlim_masks: `OrderedDict`, optional
       Dictionary of vlim_masks. Will read in if not set.

    Returns
    -------
    state: `dict`
        Dictionary containing calibration data, modes, zredstr, and vlim_masks.
    """
    if not isinstance(config, Configuration):
        config = Configuration(config)

    redmagicfilepath = os.path.dirname(config.redmagicfile)

    calib_data = OrderedDict()
    with fitsio.FITS(config.redmagicfile) as fits:
        # Number of modes is number of binary extentions
        n_modes = len(fits) - 1

        for ext in range(n_modes):
            data = Entry(fits[ext + 1].read())

            try:
                name = decode_string(data.name).rstrip()
            except AttributeError:
                name = data.name.rstrip()

            calib_data[name] = data

    modes = list(calib_data.keys())

    zredstr = read_redsequence(config.parfile, fine=True)

    if vlim_masks is None:
        vlim_masks = OrderedDict()

        for mode in modes:
            try:
                vmaskfile = decode_string(calib_data[mode].vmaskfile).rstrip()
            except AttributeError:
                vmaskfile = calib_data[mode].vmaskfile.rstrip()

            if vmaskfile == '':
                # There is no vmaskfile, we need to do a fixed area one
                vlim_masks[mode] = create_volume_limit_mask_fixed(config)
            else:
                vmaskfile = os.path.join(redmagicfilepath,
                                         os.path.basename(vmaskfile))
                if not os.path.isfile(vmaskfile):
                    raise RuntimeError("Could not find vmaskfile %s.  Must be in same path as redmagic calibration file %s." % (vmaskfile, os.path.abspath(config.redmagicfile)))

                vlim_masks[mode] = create_volume_limit_mask(config,
                                                        calib_data[mode].etamin,
                                                        vlimfile=vmaskfile)
    else:
        vlim_masks = vlim_masks

    return {
        'calib_data': calib_data,
        'modes': modes,
        'zredstr': zredstr,
        'vlim_masks': vlim_masks
    }


def select_redmagic_galaxies(state, config, gals, mode, rng=None, spec=None, return_indices=False):
    """
    Select redMaGiC galaxies from a galaxy catalog, according to the mode.

    Parameters
    ----------
    state: `dict`
       State dictionary returned by read_redmagic_calibration.
    config: `redmapper.Configuration`
       Configuration object.
    gals: `redmapper.GalaxyCatalog`
       Catalog of galaxies for redMaPPer
    mode: `str`
       redMaGiC mode to select
    rng : `np.random.RandomState`, optional
        Random number generator.
    spec: `redmapper.GalaxyCatalog`, optional
        Spectroscopic catalog.
    return_indices: `bool`, optional
       Return the indices of the galaxies selected.  Default is False.

    Returns
    -------
    redmagic_catalog: `redmapper.GalaxyCatalog`
       Catalog of redMaGiC galaxies
    indices: `np.ndarray`
       Integer array of selection (if return_indices is True)
    spec: `redmapper.GalaxyCatalog`
       Spectroscopic catalog used (can be passed back in to subsequent calls).
    """

    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    # Check if we have to decode mode (py2/py3)
    if hasattr(mode, 'decode'):
        _mode = decode_string(mode)
    else:
        _mode = mode

    modes = state['modes']
    if _mode not in modes:
        raise RuntimeError("Requested redMaGiC mode %s not available." % (_mode))

    calstr = state['calib_data'][_mode]

    # Takes in galaxies...
    # Which are the possibly red galaxies?
    lstar_cushion = calstr.lstar_cushion
    z_cushion = calstr.z_cushion
    z_buffer = calstr.buffer

    mstar_init = redsequence_mstar(state['zredstr'], gals.zred_uncorr)

    cut_zrange = [calstr.cost_zrange[0] - z_cushion - z_buffer,
                  calstr.cost_zrange[1] + z_cushion + z_buffer]
    minlstar = np.clip(np.min(calstr.etamin) - lstar_cushion, 0.1, None)

    red_poss_mask = ((gals.zred_uncorr > cut_zrange[0]) &
                     (gals.zred_uncorr < cut_zrange[1]) &
                     (gals.chisq < calstr.maxchi) &
                     (gals.refmag < (mstar_init - 2.5*np.log10(minlstar))))

    # Creates a new catalog...
    zredmagic = np.copy(gals.zred_uncorr)
    zredmagic_e = np.copy(gals.zred_uncorr_e)
    try:
        zredmagic_samp = np.copy(gals.zred_samp)
    except (ValueError, AttributeError) as e:
        # Sample from zred + zred_e (not optimal, for old catalogs)
        zredmagic_samp = np.zeros((zredmagic.size, 1))
        zredmagic_samp[:, 0] = rng.normal(loc=zredmagic,
                                          scale=zredmagic_e,
                                          size=zredmagic.size)

    y2 = cubic_spline_compute_y2(calstr.nodes, calstr.cmax)
    chi2max = np.clip(cubic_spline_interpolate(gals.zred_uncorr, calstr.nodes, calstr.cmax, y2, fixextrap=True), 0.1, calstr.maxchi)

    if calstr.run_afterburner:
        y2 = cubic_spline_compute_y2(calstr.corrnodes, calstr.bias)
        offset = cubic_spline_interpolate(gals.zred_uncorr, calstr.corrnodes, calstr.bias, y2, fixextrap=True)
        zredmagic -= offset

        if calstr.apply_afterburner:
            for i in range(zredmagic_samp.shape[1]):
                zredmagic_samp[:, i] -= offset

        y2 = cubic_spline_compute_y2(calstr.corrnodes, calstr.eratio)
        zredmagic_e *= cubic_spline_interpolate(gals.zred_uncorr, calstr.corrnodes, calstr.eratio, y2, fixextrap=True)

    # Compute mstar
    mstar = redsequence_mstar(state['zredstr'], zredmagic)

    # Compute the maximum redshift
    vmask = state['vlim_masks'][_mode]
    zmax = calc_zmax(vmask, gals.ra, gals.dec)

    # Do the redmagic selection
    gd, = np.where((gals.chisq < chi2max) &
                   (gals.refmag < (mstar - 2.5 * np.log10(calstr.etamin))) &
                   (zredmagic < zmax) &
                   (red_poss_mask))

    redmagic_catalog = GalaxyCatalog(np.zeros(gd.size, dtype=[('id', 'i8'),
                                                              ('ra', 'f8'),
                                                              ('dec', 'f8'),
                                                              ('refmag', 'f4'),
                                                              ('refmag_err', 'f4'),
                                                              ('mag', 'f4', config.nmag),
                                                              ('mag_err', 'f4', config.nmag),
                                                              ('lum', 'f4'),
                                                              ('zredmagic', 'f4'),
                                                              ('zredmagic_e', 'f4'),
                                                              ('zredmagic_samp', 'f4', config.zred_nsamp),
                                                              ('chisq', 'f4'),
                                                              ('zspec', 'f4')]))

    if gd.size == 0:
        if return_indices:
            return redmagic_catalog, gd, spec
        else:
            return redmagic_catalog, spec

    redmagic_catalog.id = gals.id[gd]
    redmagic_catalog.ra = gals.ra[gd]
    redmagic_catalog.dec = gals.dec[gd]
    redmagic_catalog.refmag = gals.refmag[gd]
    redmagic_catalog.refmag_err = gals.refmag_err[gd]
    redmagic_catalog.mag[:, :] = gals.mag[gd, :]
    redmagic_catalog.mag_err[:, :] = gals.mag_err[gd, :]
    redmagic_catalog.zredmagic = zredmagic[gd]
    redmagic_catalog.zredmagic_e = zredmagic_e[gd]
    redmagic_catalog.zredmagic_samp = zredmagic_samp[gd, :]
    redmagic_catalog.chisq = gals.chisq[gd]

    # Compute the luminosity
    redmagic_catalog.lum = 10.**((mstar[gd] - redmagic_catalog.refmag) / 2.5)

    # In the future, add absolute magnitude calculations, but that will
    # require some k-corrections.

    # Compute the zspec (check this)
    if 'ztrue' in gals.dtype.names:
        # We have truth zspec
        redmagic_catalog.zspec = gals.ztrue[gd]
    elif 'zspec' in gals.dtype.names:
        # We have already done a zspec match
        redmagic_catalog.zspec = gals.zspec[gd]
    else:
        # We need to do a zspec match here
        if spec is None:
            logger.info("Reading in spectroscopic information...")

            spec = GalaxyCatalog.from_fits_file(config.specfile)
            use, = np.where(spec.z_err < 0.001)
            spec = spec[use]
            logger.info("Done reading in spectroscopic information.")

        redmagic_catalog.zspec[:] = -1.0
        i0, i1, dists = spec.match_many(redmagic_catalog.ra, redmagic_catalog.dec, 3./3600., maxmatch=1)
        redmagic_catalog.zspec[i0] = spec.z[i1]

    if return_indices:
        return redmagic_catalog, gd, spec
    else:
        return redmagic_catalog, spec
