import matplotlib
matplotlib.use('Agg')

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
import time
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper.run_colormem import run_colormem
from redmapper.calibration import select_spec_red_galaxies_wrapper
from redmapper import Catalog

class RunColormemTestCase(unittest.TestCase):
    """
    Tests of redmapper.RunColormem, which computes richness by fitting the
    red-sequence for each cluster (for calibration)
    """

    def test_run_colormem(self):
        """
        Run tests of redmapper.RunColormem
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))
        config.randomseed = 12345

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        # First, we need the red galaxy model

        config.specfile_train = os.path.join(file_path, 'test_dr8_spec.fit')
        config.zrange = [0.1,0.2]

        config.redgalfile = config.redmapper_filename('test_redgals')
        config.redgalmodelfile = config.redmapper_filename('test_redgalmodel')

        select_spec_red_galaxies_wrapper(config)

        # Main test...
        config.zmemfile = config.redmapper_filename('test_zmem')

        cat, members = run_colormem(config)
        use, = np.where(members.pcol > config.calib_pcut)
        savemem = members[use]
        savemem.to_fits_file(config.zmemfile)
        cat.to_fits_file(config.redmapper_filename('colorcat'))

        # Check that the files are there...
        self.assertTrue(os.path.isfile(config.zmemfile))

        mem = fitsio.read(config.zmemfile, ext=1)
        testing.assert_equal(mem.size, 16)
        testing.assert_array_almost_equal(mem['pcol'][0:3], np.array([0.95385, 0.83143, 0.88814]), 5)
        testing.assert_array_almost_equal(mem['z'][0:3], np.array([0.191797, 0.194327, 0.186235]), 2)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
