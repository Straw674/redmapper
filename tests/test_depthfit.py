import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
import shutil
import os
from numpy import random

from redmapper import compute_depthlim_pars, apply_depthlim
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper.mask import get_mask, select_maskgals_sample, read_maskgals
from redmapper.depth_fitting import calcErrorModel

class DepthFitTestCase(unittest.TestCase):
    """
    Test fitting the local depth with functional depth limit equivalents.
    """
    def test_depthfit(self):
        """
        Test depth fitting functions.
        """

        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        gals = GalaxyCatalog.from_galfile(config.galfile)

        np.random.seed(seed=12345)

        # First creation of depth limit pars
        initpars = compute_depthlim_pars(gals.refmag, gals.refmag_err)

        # This has been inspected that it makes sense.
        # Also, one should really be using a depth map

        testing.assert_almost_equal(initpars['EXPTIME'][0], 104.782066, 0)
        testing.assert_almost_equal(initpars['LIMMAG'][0], 20.64819717, 0)

        # And take a subpixel
        gals = GalaxyCatalog.from_galfile(config.galfile, hpix=8421, nside=128)

        config.mask_mode = 0
        mask = get_mask(config)
        mask['maskgals'], mask['maskgal_index'] = select_maskgals_sample(config, mask['maskgals_all'], mask['rng'])
        mask['maskgals_all'] = read_maskgals(config.maskgalfile)

        apply_depthlim(mask['maskgals'], gals.refmag, gals.refmag_err, initpars)

        pars, fail = calcErrorModel(gals.refmag, gals.refmag_err, calcErr=False)

        testing.assert_almost_equal(pars['EXPTIME'][0], 63.73879623, 0)
        testing.assert_almost_equal(pars['LIMMAG'][0], 20.68231583, 0)

        # And make sure the maskgals are getting the right constant value
        testing.assert_almost_equal(pars['EXPTIME'][0], mask['maskgals'].exptime.min())
        testing.assert_almost_equal(pars['EXPTIME'][0], mask['maskgals'].exptime.max())
        testing.assert_almost_equal(pars['LIMMAG'][0], mask['maskgals'].limmag.min())
        testing.assert_almost_equal(pars['LIMMAG'][0], mask['maskgals'].limmag.max())

if __name__=='__main__':
    unittest.main()