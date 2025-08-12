"""Classes related to preparing members for the next calibration iteration
"""
import os
import numpy as np
import esutil

from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from ..utilities import read_members
from ..configuration import Configuration

class PrepMembers(object):
    """
    Class to prepare members for input to the next calibration iteration.
    """

    def __init__(self, conf, rng=None):
        """
        Instantiate a PrepMembers object.

        Parameters
        ----------
        conf: `str` or `redmapper.Configuration`
           Config filename or configuration object
        rng : `np.random.RandomState`, optional
            Random number generator.
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        if rng is None:
            rng = np.random.RandomState(self.config.randomseed)
        self.rng = rng

    def run(self, mode):
        """
        Run the member preparation.

        Output members are put into self.config.zmemfile.

        Parameters
        ----------
        mode: `str`
           May be "z_init": use initial cluster seed redshift as member redshift or
           may be "cg": use the most likely central spec_z as member redshift

        Raises
        ------
        RuntimeError: If mode is not "z_init" or "cg".
        """

        cat = Catalog.from_fits_file(self.config.catfile)

        if mode == 'z_init':
            cat_z = cat.z_init
        elif mode == 'cg':
            cat_z = cat.cg_spec_z
        else:
            raise RuntimeError("Unsupported mode %s" % (mode))

        mem = read_members(self.config.catfile)

        # Cut the clusters
        use, = np.where((cat.Lambda / cat.scaleval > self.config.calib_minlambda) &
                        (cat.scaleval > 0.0) &
                        (np.abs(cat_z - cat.z_lambda) < self.config.calib_zlambda_clean_nsig * cat.z_lambda_e))
        cat = cat[use]
        cat_z = cat_z[use]

        # Cut the members
        use, = np.where((mem.p * mem.theta_i * mem.theta_r > self.config.calib_pcut) |
                        (mem.pcol > self.config.calib_pcut))

        mem = mem[use]

        # Match cut clusters to members
        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)

        newmem = Catalog(np.zeros(b.size, dtype=[('z', 'f4'),
                                                 ('z_lambda', 'f4'),
                                                 ('p', 'f4'),
                                                 ('pcol', 'f4'),
                                                 ('central', 'i2'),
                                                 ('ra', 'f8'),
                                                 ('dec', 'f8'),
                                                 ('mag', 'f4', self.config.nmag),
                                                 ('mag_err', 'f4', self.config.nmag),
                                                 ('refmag', 'f4'),
                                                 ('refmag_err', 'f4'),
                                                 ('ebv', 'f4')]))

        newmem.ra[:] = mem.ra[b]
        newmem.dec[:] = mem.dec[b]
        newmem.p[:] = mem.p[b]
        newmem.pcol[:] = mem.pcol[b]
        newmem.mag[:, :] = mem.mag[b, :]
        newmem.mag_err[:, :] = mem.mag_err[b, :]
        newmem.refmag[:] = mem.refmag[b]
        newmem.refmag_err[:] = mem.refmag_err[b]
        newmem.ebv[:] = mem.ebv[b]

        cent, = np.where(mem.r[b] < 0.0001)
        newmem.central[cent] = 1

        newmem.z[:] = cat_z[a]
        newmem.z_lambda = cat.z_lambda[a]

        if self.config.calib_smooth > 0.0:
            newmem.z[:] += self.config.calib_smooth * self.rng.normal(size=newmem.size)

        newmem.to_fits_file(self.config.zmemfile)


