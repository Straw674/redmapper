"""Function to predict memory usage for runs at various nsides.
"""
import os
import numpy as np
import hpgeom as hpg
import esutil
import fitsio

from ..configuration import Configuration
from ..catalog import Entry
from ..utilities import decode_string

def predict_memory(configfile, include_zreds=True, border_factor=2.0):
    """Predict the per-tile memory usage from the galaxy/zred catalog
    for all nsides between 2 and the nside used for the galaxy catalog.

    Parameters
    ----------
    configfile : `str`
       Configuration yaml filename.
    include_zreds : `bool`, optional
       Include zreds in memory prediction.
    border_factor : `float`, optional
       Factor to inflate memory to approximate outside-boundary pixels.

    Returns
    -------
    galmem_usage : `np.ndarray`
       Array with nside/galaxy memory usage for various nsides.
    """
    config = Configuration(configfile)
    
    tab = Entry.from_fits_file(config.galfile, ext=1)
    nside_tab = tab.nside

    path = os.path.dirname(os.path.abspath(config.galfile))

    try:
        first_fname = os.path.join(path, decode_string(tab.filenames[0]))
    except AttributeError:
        first_fname = os.path.join(path, tab.filenames[0])

    elt = fitsio.read(first_fname, ext=1, rows=0, lower=True)
    dtype_in = elt.dtype.descr
    # Remove truth information
    mark = []
    for dt in dtype_in:
        if (dt[0] != 'ztrue' and dt[0] != 'm200' and dt[0] != 'central' and
            dt[0] != 'halo_id'):
            mark.append(True)
        else:
            mark.append(False)

    dtype = [dt for i, dt in enumerate(dtype_in) if mark[i]]

    test = np.zeros(1, dtype=dtype)
    nbytes = test.nbytes

    if include_zreds:
        ztab = Entry.from_fits_file(config.zredfile, ext=1)
        zpath = os.path.dirname(config.zredfile)
        try:
            fname = os.path.join(zpath, decode_string(ztab.filenames[0]))
        except AttributeError:
            fname = os.path.join(zpath, ztab.filenames[0])

        zelt = fitsio.read(fname, ext=1, rows=0, lower=True)
        nbytes += zelt[0].nbytes

    # Now compute the number of bytes per galaxy tile
    hpix_tab = tab.hpix

    tab_mb = tab.ngals*nbytes/(1000.0*1000.0)

    # Now break apart into different nsides ...
    nsides = [2]
    while nsides[-1] < tab.nside:
        nsides.append(nsides[-1]*2)

    retstr = np.zeros(len(nsides), dtype=[('nside', 'i4'),
                                          ('npix', 'i4'),
                                          ('max_memory_mb', 'f4')])
    retstr['nside'] = nsides

    theta, phi = hpg.pixel_to_angle(tab.nside, tab.hpix, lonlat=False, nest=False)
    for i, nside in enumerate(nsides):
        hpix_run = hpg.pixel_to_angle(nside, theta, phi, lonlat=False, nest=False)

        h, rev = esutil.stat.histogram(hpix_run, rev=True)

        use, = np.where(h > 0)
        retstr['npix'][i] = use.size
        for ind in use:
            i1a = rev[rev[ind]: rev[ind + 1]]
            mem = np.sum(tab_mb[i1a])*border_factor
            if mem > retstr['max_memory_mb'][i]:
                retstr['max_memory_mb'][i] = mem

    for i in range(retstr.size):
        print('nside = %d, max memory = %.2f Mb on %d pixels.' %
              (retstr['nside'][i], retstr['max_memory_mb'][i], retstr['npix'][i]))
    return retstr
