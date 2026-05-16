"""Function to compute zreds on a single pixel, for distributed runs.
"""
import os
import numpy as np
import glob
import re

from ..configuration import Configuration
from ..zred_runner import run_zred_pixels, make_zred_table
from ..catalog import Entry
from ..utilities import make_lockfile
from ..logger import logger

def run_zred_pixel_task(configfile, pixel, nside, path=None):
    """
    Calculate zreds on a single healpix pixel, for distributed runs.

    All files will be placed in the path in config.zredfile, and when
    the final pixel is run the config.zredfile master table will be
    created.

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
       path as configfile.  I think this is unused.
    """
    if path is None:
        outpath = os.path.dirname(os.path.abspath(configfile))
    else:
        outpath = path

    config = Configuration(configfile, outpath=outpath)

    # Make sure galaxy file exists, and is pixelized
    if not config.galfile_pixelized:
        raise ValueError("Code only runs with pixelized galfile.")

    # Create output path if necessary (with checks)
    zredpath = os.path.dirname(config.zredfile)
    galpath = os.path.dirname(config.galfile)

    test = re.search('^(.*)_zreds_master_table.fit',
                     os.path.basename(config.zredfile))
    if test is None:
        raise ValueError("zredfile filename not in proper format (must end with _zreds_master_table.fit)")

    config.outbase = test.groups()[0]

    if not os.path.exists(zredpath):
        try:
            os.makedirs(zredpath)
        except OSError:
            # Make sure that the path exists (From another run), if so we're good
            if not os.path.exists(zredpath):
                raise IOError("Could not create %s directory" % (zredpath))

    # Configure the config to run only this pixel
    config.hpix = [pixel]
    config.nside = nside
    config.outbase = '%s_%05d' % (config.outbase, pixel)
    config.border = 0.0

    # Create a pixel lockfile
    # Note that the pixel number will probably contain many sub-pixels, but
    # this is fine because we just don't want these jobs to have the possibility
    # of stepping on each other
    writelock = '%s/%s_zreds_%07d.lock' % (zredpath, config.outbase, pixel)
    test = make_lockfile(writelock, block=False)
    if not test:
        raise IOError("Failed to get lock on pixel %d" % (pixel))

    # Compute all the zreds and output pixels
    run_zred_pixels(config, single_process=True, no_zred_table=True, verbose=True)

    # We are done writing, so we can clear the lockfile
    os.unlink(writelock)

    # Make a lockfile and check what's been output already.

    lockfile = '%s.lock' % (config.zredfile)
    locktest = make_lockfile(lockfile, block=True, maxtry=60, waittime=2)
    if locktest:
        logger.info("Created lock file: %s" % (lockfile))
        logger.info("Checking for zred completion...")

        test_files = glob.glob('%s/%s_zreds_???????.fit' % (zredpath, config.outbase))
        test_locks = glob.glob('%s/%s_zreds_???????.lock' % (zredpath, config.outbase))
        
        galtable = Entry.from_fits_file(config.galfile)

        if (len(test_files) == len(galtable.filenames) and
            len(test_locks) == 0):
            # We have written all the files, and there are no locks left.
            logger.info("All zred files have been found!  Creating master table.")

            indices = np.arange(len(galtable.filenames))
            filenames = []
            for i in indices:
                filenames.append('%s/%s_zreds_%07d.fit' % (zredpath, config.outbase, galtable.hpix[i]))

            indices_and_filenames = list(zip(indices, filenames))

            make_zred_table(config, indices_and_filenames, galtable, config.outbase)
        elif len(test_locks) > 0:
            pass

        # clear the lockfile
        os.unlink(lockfile)
    else:
        logger.info("Failed to get a consolidate lock.  That's okay.")
