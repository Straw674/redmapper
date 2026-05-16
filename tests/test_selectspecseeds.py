import matplotlib
matplotlib.use('Agg')

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

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import Catalog

class SelectSpecSeedsTestCase(unittest.TestCase):
    """
    Tests for creating spectroscopic seeds for a run in
    redmapper.calibration.selectspecseeds.select_spec_seeds_wrapper
    """

    def test_selectspecseeds(self):
        """
        Run tests on redmapper.calibration.selectspecseeds.select_spec_seeds_wrapper
        """
        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))
        config.galfile = os.path.join(file_path, 'test_dr8_trainred_gals.fit')
        config.specfile_train = os.path.join(file_path, 'test_dr8_trainred_spec.fit')

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir
        config.seedfile = os.path.join(self.test_dir, 'test_seeds.fit')

        from redmapper.calibration.selectspecseeds import select_spec_seeds_wrapper
        select_spec_seeds_wrapper(config, usetrain=True)

        self.assertTrue(os.path.isfile(config.seedfile))
        seeds = Catalog.from_fits_file(config.seedfile)
        self.assertGreater(seeds.size, 0)
        self.assertIn('zspec', seeds.colnames)
        self.assertIn('zred', seeds.colnames)


    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
