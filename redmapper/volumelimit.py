"""Functions to describe the volume limit mask.
"""
import fitsio
import hpgeom as hpg
import numpy as np
import esutil
import os
import healsparse

from .catalog import Catalog, Entry
from .redsequence import read_redsequence, redsequence_mstar
from .utilities import astro_to_sphere, get_healsparse_subpix_indices
from .logger import logger

def create_volume_limit_mask(config, vlim_lstar, vlimfile=None, use_geometry=False, withversion=True):
    """
    Create a volume limit mask dictionary.

    If the mask described by maskfile already exists, it will be read in
    directly.  If it does not exist, it will be generated from the depth
    files described in the config parameters, and then stored in vlimfile.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    vlim_lstar: `float`
       Luminosity cutoff (units of L*) for red-sequence volume limit
       computation
    vlimfile: `str`, optional
       Filename to store volume limit mask.  Default is None, which
       means generate the filename of the 'vlim_zmask' type.
    use_geometry: `bool`, optional
       Use the geometric mask info only.  Only use if necessary.
       Default is False.
    withversion: `bool`, optional
       Output filename with redmapper version string.
       
    Returns
    -------
    dict
       Volume limit mask data dictionary.
    """
    if vlimfile is None:
        vlimfile = config.redmapper_filename('vl%02d_vlim_zmask' %
                                                        (int(vlim_lstar*10)),
                                                       withversion=withversion)

    if os.path.isfile(vlimfile):
        return _read_mask(vlimfile, config)
    else:
        if use_geometry:
            _build_geometry_mask(config, vlimfile)
        else:
            _build_mask(config, vlim_lstar, vlimfile)
        return _read_mask(vlimfile, config)

def _read_mask(vlimfile, config):
    """
    Read an existing volume-limit mask into the data structure.
    """
    hdr = fitsio.read_header(vlimfile, ext=1)
    if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
        raise RuntimeError("Need to specify vlimfile in healsparse format.")

    cov_hdr = fitsio.read_header(vlimfile, ext='COV')
    nside_coverage = cov_hdr['NSIDE']

    if len(config.hpix) > 0:
        covpixels = get_healsparse_subpix_indices(config.nside, config.hpix,
                                                  config.border, nside_coverage)
    else:
        covpixels = None

    sparse_vlimmap = healsparse.HealSparseMap.read(vlimfile, pixels=covpixels)

    return {
        'type': 'sparse',
        'vlimfile': vlimfile,
        'sparse_vlimmap': sparse_vlimmap,
        'nside': sparse_vlimmap.nside_sparse,
        'subpix_nside': config.hpix,
        'subpix_hpix': config.nside,
        'subpix_border': config.border,
        'zrange': config.zrange,
        'area_finebin': config.area_finebin
    }

def _build_mask(config, vlim_lstar, vlimfile):
    """
    Build a VolumeLimitMask from the parameters in the config file, and
    store the mask in vlimfile
    """

    # Make some checks to make sure we can build a volume limit mask
    if config.depthfile is None or not os.path.isfile(config.depthfile):
        raise RuntimeError("Cannot create a volume limit mask without a depth file")
    for fname in config.vlim_depthfiles:
        if not os.path.isfile(fname):
            raise RuntimeError("Could not find specified vlim_depthfile %s" % (fname))

    # Read in the red-sequence parameters
    zredstr = read_redsequence(config.parfile, fine=True)

    # create the redshift bins
    zbinsize = 0.001 # arbitrary fine bin
    nzbins = int(np.ceil((config.zrange[1] - config.zrange[0]) / zbinsize))
    # Note that we want to start one step above the low redshift range
    zbins = np.arange(nzbins) * zbinsize + config.zrange[0] + zbinsize

    # magnitude limits
    limmags = redsequence_mstar(zredstr, zbins) - 2.5 * np.log10(vlim_lstar)

    # get the reference index
    ref_ind = config.bands.index(config.refmag)

    # Read in the primary depth structure
    sparse_depthmap = healsparse.HealSparseMap.read(config.depthfile)

    dtype_vlimmap = [('fracgood', 'f4'),
                     ('zmax', 'f4')]

    sparse_vlimmap = healsparse.HealSparseMap.make_empty(sparse_depthmap.nside_coverage,
                                                         sparse_depthmap.nside_sparse,
                                                         dtype=dtype_vlimmap,
                                                         primary='fracgood')

    validPixels = sparse_depthmap.valid_pixels
    depthValues = sparse_depthmap.get_values_pix(validPixels)
    vlimmap = np.zeros(validPixels.size, dtype=dtype_vlimmap)
    vlimmap['fracgood'] = depthValues['fracgood']

    lo, = np.where(depthValues['m50'] <= limmags.min())
    vlimmap['zmax'][lo] = zbins.min()
    hi, = np.where(depthValues['m50'] >= limmags.max())
    vlimmap['zmax'][hi] = zbins.max()
    mid, = np.where((depthValues['m50'] > limmags.min()) & (depthValues['m50'] < limmags.max()))
    if mid.size > 0:
        l = np.searchsorted(limmags, depthValues['m50'][mid], side='right')
        vlimmap['zmax'][mid] = zbins[l]

    # Read in any additional depth maps
    for i, depthfile in enumerate(config.vlim_depthfiles):
        sparse_depthmap2, hdr2 = healsparse.HealSparseMap.read(depthfile, header=True)

        validPixels2 = sparse_depthmap2.valid_pixels
        depthValues2 = sparse_depthmap2.get_values_pix(validPixels2)

        nsig = hdr2['NSIG']
        zp = hdr2['ZP']

        # find mag name thing...
        # Note this is validated in the config read
        map_ind = config.bands.index(config.vlim_bands[i])

        # match pixels
        a, b = esutil.numpy_util.match(validPixels, validPixels2)

        n2 = config.vlim_nsigs[i]**2.
        flim_in = 10.**((depthValues2['limmag'][b] - zp) / (-2.5))
        fn = np.clip((flim_in**2. * depthValues2['exptime'][b]) / (nsig**2.) - flim_in, 0.001, None)
        flim_mask = (n2 + np.sqrt(n2**2. + 4.*depthValues2['exptime'][b] * n2 * fn)) / (2.*depthValues2['exptime'][b])
        lim_mask = np.zeros(vlimmap.size)
        lim_mask[a] = zp - 2.5*np.log10(flim_mask)

        zinds = np.searchsorted(zredstr['z'], zbins, side='right')

        limmags_temp = redsequence_mstar(zredstr, zbins) - 2.5*np.log10(vlim_lstar)
        refmag_lim = limmags_temp.copy()

        if (map_ind == ref_ind):
            logger.info('Warning: vlim_band %s is the same as the reference band!  Skipping...' % (config.vlim_bands[i]))
        else:
            if map_ind < ref_ind:
                # Need to go blueward
                for jj in range(ref_ind - 1, map_ind - 1, -1):
                    limmags_temp += (zredstr['c'][zinds, jj] + zredstr['slope'][zinds, jj] * (refmag_lim - zredstr['pivotmag'][zinds]))
            else:
                # Need to go redward
                for jj in range(ref_ind, map_ind):
                    limmags_temp -= (zredstr['c'][zinds, jj] + zredstr['slope'][zinds, jj] * (refmag_lim - zredstr['pivotmag'][zinds]))

        # adjust zmax with zmax_temp
        zmax_temp = np.zeros(vlimmap.size)

        lo, = np.where(lim_mask <= limmags_temp.min())
        zmax_temp[lo] = zbins.min()
        hi, = np.where(lim_mask >= limmags_temp.max())
        zmax_temp[hi] = zbins.max()
        mid, = np.where((lim_mask > limmags_temp.min()) & (lim_mask < limmags_temp.max()))
        if mid.size > 0:
            l = np.clip(np.searchsorted(limmags_temp, lim_mask[mid], side='right'), 0, zbins.size - 1)
            zmax_temp[mid] = zbins[l]

        limited, = np.where(zmax_temp < vlimmap['zmax'])
        vlimmap['zmax'][limited] = zmax_temp[limited]


    gd, = np.where(vlimmap['zmax'] > zbins[0])

    sparse_vlimmap.update_values_pix(validPixels[gd], vlimmap[gd])

    sparse_vlimmap.write(vlimfile)

def _build_geometry_mask(config, vlimfile):
    """
    Build a VolumeLimitMask from the geometric mask from the config file,
    and store the mask in vlimfile
    """

    if config.maskfile is None or not os.path.isfile(config.maskfile):
        raise RuntimeError("Cannot create a geometry volume limit mask without a mask file")

    sparse_mask = healsparse.HealSparseMap.read(config.maskfile)

    dtype_vlimmap = [('fracgood', 'f4'),
                     ('zmax', 'f4')]

    sparse_vlimmap = healsparse.HealSparseMap.make_empty(sparse_mask.nside_coverage,
                                                         sparse_mask.nside_sparse,
                                                         dtype=dtype_vlimmap,
                                                         primary='fracgood')

    validPixels = sparse_mask.valid_pixels
    maskValues = sparse_mask.get_values_pix(validPixels)
    vlimmap = np.zeros(validPixels.size, dtype=dtype_vlimmap)
    vlimmap['fracgood'] = maskValues
    vlimmap['zmax'] = config.zrange[1]

    sparse_vlimmap.update_values_pix(validPixels, vlimmap)

    sparse_vlimmap.write(vlimfile)

def calc_zmax(vlim_mask_data, ras, decs, get_fracgood=False):
    """
    Calculate the maximum redshifts associated with a set of ra/decs.

    Parameters
    ----------
    vlim_mask_data: `dict`
       Volume limit mask data dictionary
    ras: `np.array` or `float`
       Float array of right ascensions
    decs: `np.array` or `float`
       Float array of declinations
    get_fracgood: `bool`, optional
       Also retrieve the fracgood pixel coverage.  Default is False.

    Returns
    -------
    zmax: `np.array` or `float`
       Float array of maximum redshifts
    fracgood: `np.array`, optional
       Float array of fracgood, if get_fracgood=True
    """
    if vlim_mask_data['type'] == 'fixed':
        if np.isscalar(ras):
            zmax = vlim_mask_data['z_max']
        else:
            zmax = np.full_like(ras, vlim_mask_data['z_max'])
        if not get_fracgood:
            return zmax
        else:
            if np.isscalar(ras):
                return (zmax, 1.0)
            else:
                return (zmax, np.ones_like(ras))

    if (len(np.atleast_1d(ras)) != len(np.atleast_1d(decs))):
        raise ValueError("ras, decs must be same length")

    values = vlim_mask_data['sparse_vlimmap'].get_values_pos(ras, decs, lonlat=True)

    bad, = np.where(np.abs(decs) > 90.0)
    values['zmax'][bad] = hpg.UNSEEN
    values['fracgood'][bad] = 0.0

    if not get_fracgood:
        return np.clip(values['zmax'], 0.0, None)
    else:
        return (np.clip(values['zmax'], 0.0, None), values['fracgood'])

def get_volume_limit_areas(vlim_mask_data):
    """
    Retrieve the area structure (area as a function of redshift) associated
    with the volume-limit mask.

    Parameters
    ----------
    vlim_mask_data: `dict`
       Volume limit mask data dictionary
       
    Returns
    -------
    astr: `redmapper.Catalog`
       Area structure catalog, with .z and .area
    """
    if vlim_mask_data['type'] == 'fixed':
        zbins = np.arange(vlim_mask_data['zrange'][0], vlim_mask_data['zrange'][1], vlim_mask_data['area_finebin'])

        astr = Catalog(np.zeros(zbins.size, dtype=[('z', 'f4'),
                                                   ('area', 'f4')]))
        astr.z = zbins
        astr.area = vlim_mask_data['area']
        return astr
        
    zbinsize = vlim_mask_data['area_finebin']
    zbins = np.arange(vlim_mask_data['zrange'][0], vlim_mask_data['zrange'][1], zbinsize)

    astr = Catalog(np.zeros(zbins.size, dtype=[('z', 'f4'),
                                               ('area', 'f4')]))
    astr.z = zbins

    pixsize = hpg.nside_to_pixel_area(vlim_mask_data['nside'], degrees=True)

    validPixels = vlim_mask_data['sparse_vlimmap'].valid_pixels
    zmax = vlim_mask_data['sparse_vlimmap'].get_values_pix(validPixels)['zmax']
    st = np.argsort(zmax)

    fracgoods = vlim_mask_data['sparse_vlimmap'].get_values_pix(validPixels)['fracgood'][st]

    inds = np.searchsorted(zmax[st], zbins, side='right')

    lo = (inds <= 0)
    astr.area[lo] = np.sum(fracgoods.astype(np.float64)) * pixsize

    if np.sum(~lo) > 0:
        carea = pixsize * np.cumsum(fracgoods, dtype=np.float64)
        astr.area[~lo] = carea[carea.size - inds[~lo]]

    return astr

def create_volume_limit_mask_fixed(config):
    """
    Create a volume limit mask dictionary with a fixed redshift maximum.

    This is used as a placeholder when there is no depth information to
    construct a true volume limit mask.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
       
    Returns
    -------
    dict
       Fixed volume limit mask data dictionary.
    """
    return {
        'type': 'fixed',
        'z_max': config.zrange[1],
        'zrange': config.zrange,
        'area_finebin': config.area_finebin,
        'area': config.area
    }
