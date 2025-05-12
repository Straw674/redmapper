import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random

from redmapper.configuration import Configuration
from redmapper.calibration import WcenCalibrator

class CenteringCalibratorTestCase(unittest.TestCase):
    """
    Test the centering model calibrator code (WcenCalibrator).
    """
    def test_centeringcal(self):
        """
        Test WcenCalibrator.
        """
        file_path = 'data_for_tests'
        conf_filename = 'testconfig_wcen.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        randcatfile = os.path.join(file_path, 'test_dr8_randcat.fit')
        randsatcatfile = os.path.join(file_path, 'test_dr8_randsatcat.fit')

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.wcenfile = os.path.join(config.outpath, '%s_testwcen.fit' % (config.d.outbase))

        # Get repeatability here
        rng = np.random.RandomState(1000)

        wc = WcenCalibrator(config, 1, randcatfile=randcatfile, randsatcatfile=randsatcatfile, rng=rng)
        wc.run(testing=True)

        # check outputs...

        # First, the schechter monte carlo.
        # These are very approximate, but checking for any unexpected changes
        testing.assert_almost_equal(wc.phi1_mmstar_m, -1.1345172724182435, 5)
        testing.assert_almost_equal(wc.phi1_mmstar_slope, -0.37794289, 5)
        testing.assert_almost_equal(wc.phi1_msig_m, 0.49644922, 5)
        testing.assert_almost_equal(wc.phi1_msig_slope, -0.13314551, 5)

        # Make sure the output file is there...
        self.assertTrue(os.path.isfile(config.wcenfile))

        # Test the reading from the config.
        vals = config._wcen_vals()
        config._set_vars_from_dict(vals)

        testing.assert_almost_equal(config.wcen_Delta0, -1.41107068614, 4)
        testing.assert_almost_equal(config.wcen_Delta1, -0.324385870342, 4)
        testing.assert_almost_equal(config.wcen_sigma_m, 0.3654790486, 4)
        testing.assert_almost_equal(config.wcen_pivot, 30.0, 4)

        testing.assert_almost_equal(config.lnw_fg_mean, -0.269133136174, 4)
        testing.assert_almost_equal(config.lnw_fg_sigma, 0.295286695144, 4)
        testing.assert_almost_equal(config.lnw_sat_mean, 0.0273435015217, 4)
        testing.assert_almost_equal(config.lnw_sat_sigma, 0.273395687497, 4)
        testing.assert_almost_equal(config.lnw_cen_mean, 0.219192726372, 4)
        testing.assert_almost_equal(config.lnw_cen_sigma, 0.136759762564, 4)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
