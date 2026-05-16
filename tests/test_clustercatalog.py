import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
from esutil.cosmology import Cosmo

from redmapper import Cluster
from redmapper import ClusterCatalog, Catalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import read_redsequence
from redmapper import read_background
from redmapper.mask import get_mask, select_maskgals_sample, compute_maskgals_mark
from redmapper import depthmap

from redmapper.cluster import cluster_dtype_base


class ClusterCatalogTestCase(unittest.TestCase):
    """
    Tests for features of redmapper.ClusterCatalog, including reading and matching
    galaxies.
    """
    def runTest(self):
        """
        Run the ClusterCatalog tests.
        """

        rng = np.random.RandomState(12345)

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'

        config = Configuration(file_path + '/' + conffile)

        gals_all = GalaxyCatalog.from_galfile(config.galfile)

        zred_filename = 'test_dr8_pars.fit'
        zredstr = read_redsequence(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        bkg = read_background('%s/%s' % (file_path, bkg_filename))

        mask = get_mask(config, rng=rng)
        mask['maskgals'], mask['maskgal_index'] = select_maskgals_sample(config, mask['maskgals_all'], mask['rng'], maskgal_index=0)
        depth_data = depthmap.read_depth_map(config)

        testcatfile = 'test_cluster_pos.fit'
        cat = ClusterCatalog.from_catfile(file_path + '/' + testcatfile,
                                          zredstr=zredstr,
                                          config=config,
                                          bkg=bkg)

        # test single neighbors...
        c0 = cat[0]
        c0.find_neighbors(0.2, gals_all)
        c1 = cat[1]
        c1.find_neighbors(0.2, gals_all)

        testing.assert_equal(c0.neighbors.size, 580)
        testing.assert_equal(c1.neighbors.size, 298)
        testing.assert_array_less(c0.neighbors.dist, 0.2)
        testing.assert_array_less(c1.neighbors.dist, 0.2)

        # and multi-match...
        i0, i1, dist = gals_all.match_many(cat.ra, cat.dec, 0.2)

        u0, = np.where(i0 == 0)
        testing.assert_equal(c0.neighbors.size, u0.size)

        u1, = np.where(i0 == 1)
        testing.assert_equal(c1.neighbors.size, u1.size)

        # and compute the richness on the first one...
        mask['maskgals'].mark = compute_maskgals_mark(mask['mask_data'], c0, mask['maskgals'], rng=mask['rng'], config=config)

        depthmap.compute_maskdepth(depth_data, mask['maskgals'], c0.ra, c0.dec, c0.mpc_scale)
        richness = c0.calc_richness(mask)

        # Make sure the numbers were propagated to the parent catalog
        testing.assert_equal(richness, cat.Lambda[0])
        testing.assert_equal(c0.Lambda_e, cat.Lambda_e[0])
        testing.assert_equal(c0.scaleval, cat.scaleval[0])

        # And make sure the numbers are correct
        testing.assert_allclose(richness, 24.39691734313965, rtol=1e-3)

        # Test creating a cluster catalog with default dtype
        testcat = ClusterCatalog.zeros(10)
        compcat = Catalog(np.zeros(10, dtype=cluster_dtype_base))

        self.assertEqual(testcat.dtype, compcat.dtype)

        # And test that each cluster has that dtype
        cluster = testcat[0]
        self.assertEqual(cluster.dtype, testcat.dtype)

        # Test creating a cluster catalog with a different dtype
        dtype = [('MEM_MATCH_ID', 'i4'),
                 ('RA', 'f8'),
                 ('DEC', 'f8'),
                 ('Z', 'f4')]
        testcat = ClusterCatalog.zeros(10, dtype=dtype)
        compcat = Catalog(np.zeros(10, dtype=dtype))

        self.assertEqual(testcat.dtype, compcat.dtype)

        # And test that each cluster has that dtype
        cluster = testcat[0]
        self.assertEqual(cluster.dtype, testcat.dtype)

if __name__=='__main__':
    unittest.main()

