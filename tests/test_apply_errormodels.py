import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import HPMask
from redmapper import Configuration
from redmapper.utilities import apply_errormodels

class ApplyErrormodelsTestCase(unittest.TestCase):
    """
    Test the apply_errormodels() function in mask.py.
    """
    def runTest(self):
        """
        Run the apply_errormodels() test.
        """
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        rng = np.random.RandomState(12345)

        mask = HPMask(config) #Create the mask
        maskgal_index = mask.select_maskgals_sample()

        #set all the necessary inputs from test file
        mask.maskgals.exptime = 100.
        mask.maskgals.limmag  = 20.
        mask.maskgals.zp[0]   = 22.5
        mask.maskgals.nsig[0] = 10.
        #necessary as mask.maskgals.exptime has shape (6000,)
        mag_in                = np.full(6000, 1, dtype = float)
        mag_in[:6]            = np.array([16., 17., 18., 19., 20., 21.])

        #test without noise
        mag, mag_err = apply_errormodels(mask.maskgals, mag_in, nonoise=True, rng=rng)
        idx = np.array([0, 1, 2, 3, 4, 5])
        mag_idl     = np.array([16., 17., 18., 19., 20., 21.])
        mag_err_idl = np.array([0.00602535, 0.0107989, 0.0212915, 0.0463765, 0.108574, 0.264390])
        testing.assert_almost_equal(mag[idx], mag_idl)
        testing.assert_almost_equal(mag_err[idx], mag_err_idl, decimal = 6)

        # Test with noise.
        mag, mag_err = apply_errormodels(mask.maskgals, mag_in, rng=rng)

        idx = np.array([0, 1, 2, 3, 4, 5, 1257, 2333, 3876])
        mag_test = np.array([16.00123414, 16.99484023, 18.01111633,
                             19.02608362, 19.80514741, 20.68279649,
                             0.99999688,  0.99999531,  1.00000518])
        mag_err_test = np.array([6.03219772e-03, 1.07476689e-02, 2.15105938e-02,
                                 4.75040916e-02, 9.07367637e-02, 1.97407256e-01,
                                 5.44155623e-06, 5.44154837e-06, 5.44159784e-06])
        testing.assert_almost_equal(mag[idx], mag_test)
        testing.assert_almost_equal(mag_err[idx], mag_err_test)


if __name__=='__main__':
    unittest.main()
