"""Classes for generating redmapper randoms.
"""

import fitsio
import esutil
import re
import copy
import numpy as np
import healsparse
import warnings
import os

from .catalog import Catalog, Entry
from .galaxy import GalaxyCatalog, GalaxyCatalogMaker
from .cluster import ClusterCatalog
from .utilities import make_nodes, cubic_spline_compute_y2, cubic_spline_interpolate
from .fitters import fit_med_z
from .volumelimit import create_volume_limit_mask, calc_zmax, get_volume_limit_areas
from .logger import logger

def generate_randoms(config, nrandoms, vlim_mask=None, vlim_lstar=None, redmapper_cat=None, rng=None):
    """
    Generate seed randoms for use in redmapper, applying redshift mask.

    Parameters
    ----------
    config : `redmapper.Configuration`
    nrandoms : `int`
       Number of random points to generate.
    vlim_mask : `dict`, optional
       Volume limit mask, or else it will be read/generated from config
    vlim_lstar : `float`, optional
       Volume limit lstar, or else it is from config.vlim_lstar
    redmapper_cat : `redmapper.ClusterCatalog`, optional
       Redmapper catalog, or else it will be read from config.catfile
    rng : `np.random.RandomState`, optional
       Pre-set random number generator.  Default is None.
    """
    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    if config.randfile is None:
        raise RuntimeError("Must set randfile in config to run GenerateRandoms.")

    if vlim_lstar is None:
        vlim_lstar = config.vlim_lstar

    if vlim_mask is None:
        vlim_mask = create_volume_limit_mask(config, vlim_lstar)

    if redmapper_cat is None:
        redmapper_cat = ClusterCatalog.from_fits_file(config.catfile)

    min_gen = 10000
    max_gen = 1000000

    n_left = copy.copy(nrandoms)
    ctr = 0

    dtype = [('id', 'i4'),
             ('ra', 'f8'),
             ('dec', 'f8'),
             ('z', 'f4'),
             ('lambda', 'f4'),
             ('id_input', 'i4')]

    info_dict = {}
    # Get outbase from config.randfile
    m = re.search(r'(.*)\_master\_table.fit$', config.randfile)
    if m is None:
        raise RuntimeError("Config has randfile of incorrect format.  Must end in _master_table.fit")
    outbase = m.groups()[0]
    maker = RandomCatalogMaker(outbase, info_dict, nside=config.galfile_nside)

    logger.info("Generating %d randoms to %s" % (n_left, outbase))

    while (n_left > 0):
        n_gen = np.clip(n_left * 3, min_gen, max_gen)
        ra_rand, dec_rand = healsparse.make_uniform_randoms(vlim_mask['sparse_vlimmap'],
                                                            n_gen, rng=rng)

        zmax, fracgood = calc_zmax(vlim_mask, ra_rand, dec_rand, get_fracgood=True)

        r = rng.uniform(size=n_gen)
        gd, = np.where(r < fracgood)

        if gd.size == 0:
            continue

        tempcat = Catalog(np.zeros(gd.size, dtype=dtype))
        tempcat.ra = ra_rand[gd]
        tempcat.dec = dec_rand[gd]
        tempcat.z = -1.0

        r = rng.choice(np.arange(redmapper_cat.size), size=gd.size, replace=True)
        zz = redmapper_cat.z_lambda[r]
        ll = redmapper_cat.Lambda[r]
        ii = redmapper_cat.mem_match_id[r]

        # zctr counts the number of successfully placed randoms
        # while i counts index through tempcat, many of which will be rejected.
        zctr = 0
        for i in range(tempcat.size):
            if (zz[zctr] < zmax[i]):
                # This is in a location that is within the volume limit.
                tempcat.z[i] = zz[zctr]
                tempcat.Lambda[i] = ll[zctr]
                tempcat.id_input[i] = ii[zctr]
                zctr += 1

        # Which of the tempcat were actually placed?
        gd, = np.where(tempcat.z > 0.0)
        n_good = gd.size

        if n_good == 0:
            continue

        if n_good > n_left:
            n_good = n_left
            gd = gd[0: n_good]

        tempcat = tempcat[gd]
        tempcat.id = np.arange(ctr + 1, ctr + n_good + 1)

        maker.append_randoms(tempcat._ndarray[: n_good])

        ctr += n_good
        n_left -= n_good
        logger.info("There are %d randoms remaining..." % (n_left))

    maker.finalize_catalog()


class RandomCatalog(GalaxyCatalog):
    """
    """

    @classmethod
    def from_randfile(cls, filename, nside=0, hpix=[], border=0.0):
        """
        """

        return super(RandomCatalog, cls).from_galfile(filename, nside=nside, hpix=hpix, border=border)

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=[], border=0.0, truth=False):
        raise NotImplementedError("Cannot call from_galfile on a RandomCatalog")

    @property
    def galcol(self):
        raise NotImplementedError("Cannot call galcol on a RandomCatalog")

    @property
    def galcol_err(self):
        raise NotImplementedError("Cannot call galcol_err on a RandomCatalog")

    @property
    def add_zred_fields(self):
        raise NotImplementedError("Cannot call add_zred_fields on a RandomCatalog")


class RandomCatalogMaker(GalaxyCatalogMaker):
    """
    """

    def __init__(self, outbase, info_dict, nside=32, maskfile=None, mask_mode=0, parallel=False):
        """
        """

        if 'LIM_REF' not in info_dict:
            info_dict['LIM_REF'] = 0.0
        if 'REF_IND' not in info_dict:
            info_dict['REF_IND'] = 0
        if 'AREA' not in info_dict:
            info_dict['AREA'] = 0.0
        if 'NMAG' not in info_dict:
            info_dict['NMAG'] = 0
        if 'MODE' not in info_dict:
            info_dict['MODE'] = 'NONE'
        if 'ZP' not in info_dict:
            info_dict['ZP'] = 0.0

        super(RandomCatalogMaker, self).__init__(outbase, info_dict, nside=nside, maskfile=maskfile, mask_mode=mask_mode, parallel=parallel)

    def split_randoms(self, rands):
        """
        """

        if self.is_finalized:
            raise RuntimeError("Cannot split randoms for an already finalized catalog.")
        if os.path.isfile(self.filename):
            raise RuntimeError("Cannot split randoms when final file %s already exists." % (self.filename))

        self.append_randoms(rands)
        self.finalize_catalog()

    def append_randoms(self, rands):
        """
        """

        self.append_galaxies(rands)

    def _check_galaxies(self, rands):
        # These always come back true.
        return True


def weight_randoms(config, randcatfile, minlambda, zrange=None, lambdabin=None, vlim_mask=None, vlim_lstar=None, redmapper_cat=None):
    """
    Compute random weights.

    Parameters
    ----------
    config : `redmapper.Configuration`
    randcatfile : `str`
       Consolidated random catalog with scaleval, maskfrac
    minlambda : `float`
       Minimum lambda to use in computations
    zrange : `np.ndarray`, optional
       2-element list of redshift range.  Default is full range.
    lambdabin : `np.ndarray`, optional
       2-element list of lambda range.  Default is full range.
    vlim_mask : `dict`, optional
       Volume limit mask, or else it will be read/generated from config
    vlim_lstar : `float`, optional
       Volume limit lstar, or else it is from config.vlim_lstar
    redmapper_cat : `redmapper.ClusterCatalog`, optional
       Redmapper catalog, or else it will be read from config.catfile
    """
    randcat = Catalog.from_fits_file(randcatfile)

    if vlim_lstar is None:
        vlim_lstar = config.vlim_lstar

    if vlim_mask is None:
        vlim_mask = create_volume_limit_mask(config, vlim_lstar)

    if redmapper_cat is None:
        redmapper_cat = ClusterCatalog.from_fits_file(config.catfile)

    if zrange is None:
        zrange = np.array([config.zrange[0], config.zrange[1]])

    zname = 'z%03d-%03d' % (int(config.zrange[0]*100),
                            int(config.zrange[1]*100))
    vlimname = 'vl%02d' % (int(vlim_lstar*10))
    if lambdabin is None:
        lamname = 'lgt%03d' % (int(minlambda))
        lambdabin = np.array([0.0, 1000.0])
    else:
        lamname = 'lgt%03d_l%03d-%03d' % (int(minlambda), int(lambdabin[0]), int(lambdabin[1]))

    zuse, = np.where((randcat.z > zrange[0]) &
                     (randcat.z < zrange[1]))

    if zuse.size == 0:
        raise RuntimeError("No random points in specified redshift range %.2f < z < %.2f" %
                           (zrange[0], zrange[1]))

    st = np.argsort(randcat.id_input[zuse])
    uid = np.unique(randcat.id_input[zuse[st]])

    a, b = esutil.numpy_util.match(redmapper_cat.mem_match_id, uid)

    if b.size < uid.size:
        raise RuntimeError("IDs in randcat do not match those of corresponding redmapper catalog.")

    a, b = esutil.numpy_util.match(redmapper_cat.mem_match_id, randcat.id_input[zuse])

    if config.select_scaleval:
        luse, = np.where((redmapper_cat.Lambda[a]/redmapper_cat.scaleval[a] > minlambda) &
                         (redmapper_cat.Lambda[a] > lambdabin[0]) &
                         (redmapper_cat.Lambda[a] <= lambdabin[1]))
    else:
        luse, = np.where((redmapper_cat.Lambda[a] > minlambda) &
                         (redmapper_cat.Lambda[a] > lambdabin[0]) &
                         (redmapper_cat.Lambda[a] <= lambdabin[1]))

    if luse.size == 0:
        raise RuntimeError("No random points in specified richness range %0.2f < lambda < %0.2f and lambda > %.2f" %
                           (lambdabin[0], lambdabin[1], minlambda))

    alluse = zuse[b[luse]]

    randpoints = Catalog.zeros(luse.size, dtype=[('ra', 'f8'),
                                                 ('dec', 'f8'),
                                                 ('ztrue', 'f4'),
                                                 ('lambda_in', 'f4'),
                                                 ('avg_lambdaout', 'f4'),
                                                 ('weight', 'f4')])
    randpoints.ra = randcat.ra[alluse]
    randpoints.dec = randcat.dec[alluse]
    randpoints.ztrue = randcat.z[alluse]
    randpoints.lambda_in = randcat.lambda_in[alluse]
    randpoints.avg_lambdaout = randcat.lambda_in[alluse]

    h, rev = esutil.stat.histogram(randcat.id_input[alluse], rev=True)
    ok, = np.where(h > 0)

    for i in ok:
        i1a = rev[rev[i]: rev[i + 1]]

        if config.select_scaleval:
            gd, = np.where((randcat.lambda_in[alluse[i1a]]/randcat.scaleval[alluse[i1a]] > minlambda) &
                           (randcat.maskfrac[alluse[i1a]] < config.max_maskfrac) &
                           (randcat.lambda_in[alluse[i1a]] > lambdabin[0]) &
                           (randcat.lambda_in[alluse[i1a]] <= lambdabin[1]))
        else:
            gd, = np.where((randcat.lambda_in[alluse[i1a]] > minlambda) &
                           (randcat.maskfrac[alluse[i1a]] < config.max_maskfrac) &
                           (randcat.lambda_in[alluse[i1a]] > lambdabin[0]) &
                           (randcat.lambda_in[alluse[i1a]] <= lambdabin[1]))

        if gd.size > 0:
            randpoints.weight[i1a[gd]] = float(i1a.size)/float(gd.size)

    # And only save the randpoints with weight > 0.0
    use, = np.where(randpoints.weight > 0.0)

    fname_base = 'weighted_randoms_%s_%s_%s' % (zname, lamname, vlimname)
    randfile_out = config.redmapper_filename(fname_base, withversion=True)

    randpoints.to_fits_file(randfile_out, indices=use)

    # And now we need to compute the associated area.

    # area = full_area(z < zmax) * (P/(P+Q)) where
    #   P = number of good points with z < zmax
    #   Q = number of bad points with z < zmax

    # get the default area structure

    astr = get_volume_limit_areas(vlim_mask)

    # Make the fitting nodes
    nodes = make_nodes(config.zrange, config.area_nodesize)

    zbinsize = config.area_coarsebin
    zbins = np.arange(config.zrange[0], config.zrange[1], zbinsize)

    st = np.argsort(randcat.z[alluse])
    ind1 = np.searchsorted(randcat.z[alluse[st]], zbins)
    if config.select_scaleval:
        gd, = np.where((randcat.lambda_in[alluse[st]]/randcat.scaleval[alluse[st]] > minlambda) &
                       (randcat.maskfrac[alluse[st]] < config.max_maskfrac) &
                       (randcat.lambda_in[alluse[st]] > lambdabin[0]) &
                       (randcat.lambda_in[alluse[st]] < lambdabin[1]))
    else:
        gd, = np.where((randcat.lambda_in[alluse[st]] > minlambda) &
                       (randcat.maskfrac[alluse[st]] < config.max_maskfrac) &
                       (randcat.lambda_in[alluse[st]] > lambdabin[0]) &
                       (randcat.lambda_in[alluse[st]] < lambdabin[1]))
    ind2 = np.searchsorted(randcat.z[alluse[st[gd]]], zbins)

    xvals = (zbins[0: -2] + zbins[1: -1])/2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yvals = np.nan_to_num(ind2[1: -1].astype(np.float64) / ind1[1: -1].astype(np.float64))

    p0 = np.ones(nodes.size)
    # Do an extra fit here for stability
    pars0 = fit_med_z(nodes, xvals, yvals, p0)
    pars = fit_med_z(nodes, xvals, yvals, pars0)

    y2 = cubic_spline_compute_y2(nodes, pars)
    corrs = np.clip(cubic_spline_interpolate(astr.z, nodes, pars, y2), 0.0, 1.0)
    astr.area = corrs*astr.area

    areafile_out = config.redmapper_filename(fname_base + '_area', withversion=True)
    astr.to_fits_file(areafile_out)

    return (randfile_out, areafile_out)