import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
from numpy import random
import time
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration, generate_randoms
from redmapper.volumelimit import create_volume_limit_mask
from redmapper import RandomCatalog, run_randoms_zmask, weight_randoms
from redmapper import Catalog

class RedmapperRandomsTestCase(unittest.TestCase):
    """
    Tests of redmapper.GenerateRandoms and redmapper.RunRandomsZmask
    """
    def test_redmapper_randoms(self):
        """
        Run test of redmapper.GenerateRandoms and redmapper.RunRandomsZmask
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.randfile = os.path.join(self.test_dir, 'rands', '%s_random_master_table.fit' % (config.outbase))
        config.catfile = os.path.join(file_path, 'test_dr8_input_cluster_catalog.fit')

        # Use the depth map to generate a zmax map, 0.2L*
        # This outputs the file
        vlim = create_volume_limit_mask(config, 5.0)

        # Use the zmax map + test_dr8_input_cluster_catalog.fit to generate
        # a list of random points
        nrands = 100
        rng = random.RandomState(12345)

        generate_randoms(config, nrands, vlim_mask=vlim, rng=rng)

        # Read in the randoms, make sure they come back
        rands = RandomCatalog.from_randfile(config.randfile)
        self.assertEqual(rands.size, nrands)

        # The distribution was confirmed on a bigger set; this just wants a quick
        # check that the numbers match
        testing.assert_array_almost_equal(rands.ra[: 2], [140.260907, 140.920711])
        testing.assert_array_almost_equal(rands.dec[: 2], [65.888083, 66.033431])
        testing.assert_array_almost_equal(rands.z[: 2], [0.221394, 0.382162])
        testing.assert_array_almost_equal(rands.Lambda[: 2], [22.093416, 35.16693])

        from redmapper.cluster_runner import output_cluster_catalog

        # Run the random points through the zmask code
        cat, members = run_randoms_zmask(config)
        filename = output_cluster_catalog(cat, members, config, 'randoms_zmask', savemembers=False, withversion=True)

        randcat = Catalog.from_fits_file(filename)
        testing.assert_array_less(0.98, randcat.scaleval)
        testing.assert_array_less(randcat.scaleval, 3.0)
        testing.assert_array_less(-0.00001, randcat.maskfrac)
        testing.assert_array_less(randcat.maskfrac, 1.00001)

        # Figure out final selection
        wt_randfile, wt_areafile = weight_randoms(config, filename, 20.0)

        # Make sure that the weighted randoms are there...
        self.assertTrue(os.path.isfile(wt_randfile))
        self.assertTrue(os.path.isfile(wt_areafile))

        # And make sure that the numbers make sense
        wrandcat = Catalog.from_fits_file(wt_randfile)
        testing.assert_array_less(0.98, wrandcat.weight)
        testing.assert_array_almost_equal(wrandcat.weight[: 3], [1.0, 1.0, 1.0])

        astr = Catalog.from_fits_file(wt_areafile)
        testing.assert_array_less(-0.0001, astr.area)
        testing.assert_array_almost_equal(astr.area[100: 103], [0.991, 0.986, 0.980], 3)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
