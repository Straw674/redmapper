import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random

from redmapper.configuration import Configuration
from redmapper.calibration import calibrate_zlambda

class ZLambdaCalTestCase(unittest.TestCase):
    """
    Tests of redmapper.calibration.calibrate_zlambda, calibration of
    z_lambda correction factors.
    """
    def test_zlambdacal(self):
        """
        Run tests on redmapper.calibration.calibrate_zlambda.
        """
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        config.zrange = [0.1, 0.2]

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.catfile = os.path.join(file_path, 'test_zlamcal_cat.fit')
        config.zlambdafile = os.path.join(self.test_dir, 'test_zlambdafile.fits')

        calibrate_zlambda(config, corrslope=False)

        # Make sure the file is there
        self.assertTrue(os.path.isfile(config.zlambdafile))

        # Read in the pars and check numbers

        pars = fitsio.read(config.zlambdafile, ext=1)
        self.assertEqual(pars['niter_true'], 3)
        testing.assert_almost_equal(pars[0]['offset_z'], np.array([0.1, 0.14, 0.20]))
        testing.assert_almost_equal(pars[0]['offset'], np.array([0.0012012032,
                                                                 -6.4453372e-05,
                                                                 -0.0023969179]))
        testing.assert_almost_equal(pars[0]['offset_true'][:, 0], np.array([0.00184188,
                                                                           0.00080734,
                                                                           -0.00375052]))
        testing.assert_almost_equal(pars[0]['offset_true'][:, 1], np.array([5.90356824e-04,
                                                                            6.35077959e-05,
                                                                            -8.10466707e-04]))
        testing.assert_almost_equal(pars[0]['offset_true'][:, 2], np.array([-1.05633946e-04,
                                                                             6.05580608e-05,
                                                                             -2.35550746e-04]))
        testing.assert_almost_equal(pars[0]['slope_z'], np.array([0.1, 0.2]))
        testing.assert_almost_equal(pars[0]['slope'], np.array([0.0, 0.0]))
        testing.assert_almost_equal(pars[0]['slope_true'][:, 0], np.array([0.0, 0.0]))
        testing.assert_almost_equal(pars[0]['scatter'], np.array([0.00274064, 0.00333985]))
        testing.assert_almost_equal(pars[0]['scatter_true'][:, 0], np.array([0.00257572,
                                                                             0.00269449]))
        testing.assert_almost_equal(pars[0]['scatter_true'][:, 1], np.array([2.79080211e-07,
                                                                             1.77075050e-03]))
        testing.assert_almost_equal(pars[0]['scatter_true'][:, 2], np.array([7.67152073e-07,
                                                                             1.73576328e-03]))
        testing.assert_almost_equal(pars[0]['zred_uncorr'], np.array([0.09935377,
                                                                      0.1407662,
                                                                      0.2042674]), 5)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
