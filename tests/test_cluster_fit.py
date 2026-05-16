import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import esutil
from numpy import random

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper.mask import get_mask, select_maskgals_sample, compute_maskgals_mark
from redmapper import depthmap
from redmapper import read_color_background
from redmapper import Cluster
from redmapper import read_redsequence

class ClusterFitTestCase(unittest.TestCase):
    """
    Tests for computing richness by fitting the red sequence (used in calibration).
    """
    def runTest(self):
        """
        Run the ClusterFit test.
        """
        rng = np.random.RandomState(12345)

        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        gals = GalaxyCatalog.from_galfile(config.galfile)

        # temporary hack...
        dist = esutil.coords.sphdist(142.12752, 65.103898, gals.ra, gals.dec)
        mpc_scale = np.radians(1.) * config.cosmo.Da(0, 0.227865)
        r = np.clip(mpc_scale * dist, 1e-6, None)
        use, = np.where(r < 0.75)

        st = np.argsort(r[use])

        cbkg = read_color_background(config.bkgfile_color, usehdrarea=True)
        zredstr = read_redsequence(None, config=config)

        cluster = Cluster(r0=0.5, beta=0.0, config=config, cbkg=cbkg, neighbors=gals[use[st]], zredstr=zredstr)
        cluster.ra = 142.12752
        cluster.dec = 65.103898
        cluster.redshift = 0.227865
        cluster.update_neighbors_dist()

        mask = get_mask(cluster.config, rng=rng)
        mask['maskgals'], mask['maskgal_index'] = select_maskgals_sample(cluster.config, mask['maskgals_all'], mask['rng'], maskgal_index=0)
        depth_data = depthmap.read_depth_map(cluster.config)
        mask['maskgals'].mark = compute_maskgals_mark(mask['mask_data'], cluster, mask['maskgals'], rng=mask['rng'], config=cluster.config)
        depthmap.compute_maskdepth(depth_data, mask['maskgals'], cluster.ra, cluster.dec, cluster.mpc_scale)

        lam = cluster.calc_richness_fit(mask, 1, centcolor_in=1.36503, calc_err=False)
        testing.assert_almost_equal(lam, 16.26602, decimal=5)
        testing.assert_almost_equal(cluster.neighbors.pcol[0:4], np.array([0.94273, 0., 0.06941, 0.16621]), 5)

        lam = cluster.calc_richness_fit(mask, 1, calc_err=False)
        testing.assert_almost_equal(lam, 16.29653, decimal=5)


if __name__=='__main__':
    unittest.main()
