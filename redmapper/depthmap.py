"""Functions to describe a redmapper depth map.
"""
import fitsio
import hpgeom as hpg
import numpy as np
import healsparse

from .utilities import get_healsparse_subpix_indices
from .logger import logger

def read_depth_map(config, depthfile=None):
    """
    Read a healpix-based redmapper depth map.

    Parameters
    ----------
    config: `redmapper.Config`
       Configuration object, with config['depthfile'] set
    depthfile: `str`, optional
       Name of depthfile to use instead of config['depthfile']

    Returns
    -------
    depth_data: `dict`
        Dictionary containing depth map data and metadata
    """
    if depthfile is None:
        depthfile = config['depthfile']
    else:
        depthfile = depthfile

    hdr = fitsio.read_header(depthfile, ext=1)
    if 'PIXTYPE' not in hdr or hdr['PIXTYPE'] != 'HEALSPARSE':
        raise RuntimeError("Need to specify depthfile in healsparse format.  See redmapper_convert_depthfile_to_healsparse.py")

    cov_hdr = fitsio.read_header(depthfile, ext='COV')
    nside_coverage = cov_hdr['NSIDE']

    if len(config['hpix']) > 0:
        covpixels = get_healsparse_subpix_indices(config['nside'], config['hpix'],
                                                  config['border'], nside_coverage)
    else:
        covpixels = None

    sparse_depthmap = healsparse.HealSparseMap.read(depthfile, pixels=covpixels)

    depth_data = {
        'depthfile': depthfile,
        'nsig': cov_hdr['NSIG'],
        'zp': cov_hdr['ZP'],
        'nband': cov_hdr['NBAND'],
        'w': cov_hdr['W'],
        'eff': cov_hdr['EFF'],
        'sparse_depthmap': sparse_depthmap,
        'galfile_nside': config['galfile_nside'],
        'nside': sparse_depthmap.nside_sparse,
        'config_area': config['area'],
        'subpix_nside': config['nside'],
        'subpix_hpix': config['hpix'],
        'subpix_border': config['border']
    }

    return depth_data

def get_depth_values(depth_data, ras, decs):
    """
    Get the depth values for a set of positions.

    Parameters
    ----------
    depth_data: `dict`
       Depth data dictionary from read_depth_map
    ras: `np.array`
       Float array of right ascensions
    decs: `np.array`
       Float array of declinations

    Returns
    -------
    limmag: `np.array`
       Limiting magnitude values
    exptime: `np.array`
       Effective exposure times
    m50: `np.array`
       50% completeness depth values.  Should be same as limmag for now.
    """

    if ras.size != decs.size:
        raise ValueError("ra, dec must be the same length")

    values = depth_data['sparse_depthmap'].get_values_pos(ras, np.clip(decs, -90.0, 90.0), lonlat=True)

    bad, = np.where(np.abs(decs) > 90.0)
    values['limmag'][bad] = hpg.UNSEEN
    values['exptime'][bad] = hpg.UNSEEN
    values['m50'][bad] = hpg.UNSEEN

    return (values['limmag'],
            values['exptime'],
            values['m50'])

def get_fracgoods(depth_data, ras, decs):
    """
    Get the fraction of good coverage of each pixel

    Parameters
    ----------
    depth_data: `dict`
       Depth data dictionary from read_depth_map
    ras: `np.array`
       Float array of right ascensions
    decs: `np.array`
       Float array of declinations

    Returns
    -------
    fracgoods: `np.array`
       Float array of fracgoods
    """

    if (ras.size != decs.size):
        raise ValueError("ra, dec must be the same length")

    values = depth_data['sparse_depthmap'].get_values_pos(ras, np.clip(decs, -90.0, 90.0), lonlat=True)

    bad, = np.where(np.abs(decs) > 90.0)
    values['fracgood'][bad] = 0.0

    return values['fracgood']

def compute_maskdepth(depth_data, maskgals, ra, dec, mpc_scale):
    """
    Calculate depth for maskgals structure.

    This will modify maskgals.limmag, maskgals.exptime, maskgals.zp,
    maskgals.nsig.

    Parameters
    ----------
    depth_data: `dict`
       Depth data dictionary from read_depth_map
    maskgals: `redmapper.Catalog` or `astropy.table.Table`
       maskgals catalog
    ra: `float`
       Right ascension to center maskgals
    dec: `float`
       Declination ot center maskgals
    mpc_scale: `float`
       Scaling in Mpc / degree at cluster redshift
    """
    unseen = hpg.UNSEEN

    # compute ra and dec based on maskgals
    ras = ra + (maskgals.x/mpc_scale)/np.cos(dec*np.pi/180.)
    decs = dec + maskgals.y/mpc_scale

    maskgals.w[:] = depth_data['w']
    # Note: in original code it was maskgals.eff = None, but maskgals might be a Table where eff is a column.
    # However, Catalog wrapper handles it.
    try:
        maskgals.eff = None
    except:
        pass
    maskgals.limmag[:] = unseen
    maskgals.zp[0] = depth_data['zp']
    maskgals.nsig[0] = depth_data['nsig']

    # Make sure the dec is within range, if we're going toward the pole (in sims)
    gd, = np.where(np.abs(decs) < 90.0)

    maskgals.limmag[gd], maskgals.exptime[gd], maskgals.m50[gd] = get_depth_values(depth_data, ras[gd], decs[gd])

    bd = (maskgals.limmag < 0.0)
    ok = ~bd
    nok = ok.sum()

    if (bd.sum() > 0):
        if (nok >= 3):
            # fill them in
            maskgals.limmag[bd] = np.median(maskgals.limmag[ok])
            maskgals.exptime[bd] = np.median(maskgals.exptime[ok])
            maskgals.m50[bd] = np.median(maskgals.m50[ok])
        elif (nok > 0):
            # fill with mean
            maskgals.limmag[bd] = np.mean(maskgals.limmag[ok])
            maskgals.exptime[bd] = np.mean(maskgals.exptime[ok])
            maskgals.m50[bd] = np.mean(maskgals.m50[ok])
        else:
            # very bad (nok == 0)
            # Set this to 1.0 so it'll get used but will give giant errors.
            # And the cluster should be filtered
            maskgals.limmag[:] = 1.0
            maskgals.exptime[:] = 1000.0
            maskgals.m50[:] = 0.0
            logger.info("Warning: Bad cluster in bad region...")

def compute_areas(depth_data, mags):
    """
    Calculate total area from the depth map as a function of magnitude.

    Parameters
    ----------
    depth_data: `dict`
       Depth data dictionary from read_depth_map
    mags: `np.array`
       Float array of magnitudes at which to compute area

    Returns
    -------
    areas: `np.array`
       Float array of total areas for each of the mags
    """

    pixsize = hpg.nside_to_pixel_area(depth_data['nside'], degrees=True)

    if (depth_data['w'] < 0.0):
        # This is just constant area
        areas = np.zeros(mags.size) + depth_data['config_area']
        return areas

    if len(depth_data['subpix_hpix']) > 0:
        # for the subregion, we need the area covered in the main pixel
        # I'm not sure what to do about border...but you shouldn't
        # be running this with a subregion with a border
        if depth_data['subpix_border'] > 0.0:
            raise RuntimeError("Cannot run calc_areas() with a subregion with a border")

        bitShift = 2 * int(np.round(np.log(depth_data['nside'] / depth_data['subpix_nside']) / np.log(2)))
        nFinePerSub = 2**bitShift
        ipnest = np.zeros(0, dtype=np.int64)
        for hpix in depth_data['subpix_hpix']:
            ipnest_temp = np.left_shift(hpg.ring_to_nest(depth_data['subpix_nside'], hpix), bitShift) + np.arange(nFinePerSub)
            ipnest = np.append(ipnest, ipnest_temp)
    else:
        ipnest = depth_data['sparse_depthmap'].valid_pixels

    areas = np.zeros(mags.size)

    values = depth_data['sparse_depthmap'].get_values_pix(ipnest)

    gd, = np.where(values['m50'] > 0.0)

    depths = values['m50'][gd]
    st = np.argsort(depths)
    depths = depths[st]

    fracgoods = values['fracgood'][gd[st]]

    inds = np.clip(np.searchsorted(depths, mags) - 1, 1, depths.size - 1)

    lo = (inds < 0)
    areas[lo] = np.sum(fracgoods, dtype=np.float64) * pixsize
    carea = pixsize * np.cumsum(fracgoods, dtype=np.float64)
    areas[~lo] = carea[carea.size - inds[~lo]]

    return areas

def convert_depthfile_to_healsparse(depthfile, healsparsefile, nsideCoverage, clobber=False):
    """
    Convert an old depthfile to a new healsparsefile

    Parameters
    ----------
    depthfile: `str`
       Input depth file
    healsparsefile: `str`
       Output healsparse file
    nsideCoverage: `int`
       Nside for sparse coverage map
    clobber: `bool`, optional
       Clobber existing healsparse file?  Default is False.
    """

    old_depth, old_hdr = fitsio.read(depthfile, ext=1, header=True, lower=True)

    nside = old_hdr['nside']

    # Need to remove the HPIX from the dtype

    dtype_new = []
    names = []
    for d in old_depth.dtype.descr:
        if d[0] != 'hpix':
            dtype_new.append(d)
            names.append(d[0])

    sparseMap = healsparse.HealSparseMap.make_empty(nsideCoverage, nside, dtype_new, primary='fracgood')

    old_depth_sub = np.zeros(old_depth.size, dtype=dtype_new)
    for name in names:
        old_depth_sub[name] = old_depth[name]

    sparseMap.update_values_pix(old_depth['hpix'], old_depth_sub, nest=old_hdr['nest'])

    hdr = fitsio.FITSHDR()
    hdr['NSIG'] = old_hdr['NSIG']
    hdr['ZP'] = old_hdr['ZP']
    hdr['NBAND'] = old_hdr['NBAND']
    hdr['W'] = old_hdr['W']
    hdr['EFF'] = old_hdr['EFF']

    sparseMap.metadata = hdr

    sparseMap.write(healsparsefile, clobber=clobber)
