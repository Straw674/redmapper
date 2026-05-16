"""Functions to generate redmagic randoms"""
import os
import copy
import numpy as np
import healsparse

from ..configuration import Configuration
from ..catalog import Catalog
from ..galaxy import GalaxyCatalog
from ..volumelimit import create_volume_limit_mask, calc_zmax, get_volume_limit_areas

def generate_redmagic_randoms(config, vlim_mask_or_file, redmagic_cat_or_file, nrandoms, filename, clobber=False, rng=None):
    """
    Generate random points, and save to filename

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    vlim_mask_or_file: `str` or `dict`
       Name of a file with the volume-limited mask information or
       a volume-limit mask.
    redmagic_cat_or_file: `str` or `redmapper.Catalog`
       Name of redmagic file or redmagic catalog.
    nrandoms: `int`
       Number of randoms to generate
    filename: `str`
       Output filename
    clobber: `bool`
       Clobber output file?  Default is False.
    rng : `np.random.RandomState`, optional
        Random number generator.
    """
    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    if isinstance(vlim_mask_or_file, dict) and 'type' in vlim_mask_or_file:
        vlim_mask = vlim_mask_or_file
    elif isinstance(vlim_mask_or_file, str):
        # This 0.2 is a dummy value
        vlim_mask = create_volume_limit_mask(config, 0.2, vlimfile=vlim_mask_or_file)
    else:
        raise RuntimeError("vlim_mask_or_file must be a dict or a filename")

    if isinstance(redmagic_cat_or_file, GalaxyCatalog):
        redmagic_cat = redmagic_cat_or_file
    elif isinstance(redmagic_cat_or_file, str):
        redmagic_cat = GalaxyCatalog.from_fits_file(redmagic_cat_or_file)
    else:
        raise RuntimeError("redmagic_cat_or_file must be a redmapper.GalaxyCatalog")

    if not clobber and os.path.isfile(filename):
        raise RuntimeError("Random file %s already exists and clobber is False." % (filename))

    min_gen = 10000
    max_gen = 1000000

    n_left = copy.copy(nrandoms)
    ctr = 0

    dtype = [('ra', 'f8'),
             ('dec', 'f8'),
             ('z', 'f4'),
             ('weight', 'f4')]

    randcat = Catalog(np.zeros(nrandoms, dtype=dtype))

    # Not used at the moment
    randcat.weight[:] = 1.0

    while (n_left > 0):
        n_gen = np.clip(n_left * 3, min_gen, max_gen)
        ra_rand, dec_rand = healsparse.make_uniform_randoms(vlim_mask['sparse_vlimmap'],
                                                            n_gen, rng=rng)

        # What are the associated z_max and fracgood?
        zmax, fracgood = calc_zmax(vlim_mask, ra_rand, dec_rand, get_fracgood=True)

        # Down-select from fracgood
        r = rng.uniform(size=n_gen)
        gd, = np.where(r < fracgood)

        # Go back and generate more if all bad
        if gd.size == 0:
            continue

        tempcat = Catalog(np.zeros(gd.size, dtype=dtype))
        tempcat.ra = ra_rand[gd]
        tempcat.dec = dec_rand[gd]
        tempcat.z[:] = -1.0

        zz = rng.choice(redmagic_cat.zredmagic, size=gd.size)
        zmax = zmax[gd]

        # This essentially takes each redshift and then finds a random
        # point where it fits within the redshift envelope.  It's a bit
        # inefficient, but it preserves the redshift distribution.

        zctr = 0
        for i in range(tempcat.size):
            if zz[zctr] < zmax[i]:
                # This redshift is okay!
                tempcat.z[i] = zz[zctr]
                zctr += 1

        gd, = np.where(tempcat.z > 0.0)
        n_good = gd.size

        if n_good == 0:
            continue

        tempcat = tempcat[gd]

        if n_good > n_left:
            n_good = n_left

        randcat[ctr: ctr + n_good] = tempcat._ndarray[: n_good]

        ctr += n_good
        n_left -= n_good

    randcat.to_fits_file(filename, clobber=True)
