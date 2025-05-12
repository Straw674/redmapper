import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import os

from redmapper import Cluster
from redmapper import ClusterCatalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import DataObject
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import HPMask
from redmapper import DepthMap
from redmapper import RunZScan


class RunzscanTestCase(unittest.TestCase):
    """
    Tests of redmapper.RunZScan
    """
    def runTest(self):
        """Run the redmapper.RunZScan tests.
        """

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(os.path.join(file_path, conffile))
        config.catfile = os.path.join(file_path, catfile)
        config.zredfile = os.path.join(file_path, 'zreds_test', 'dr8_test_zreds_master_table.fit')
        config.randomseed = 12345

        runzscan = RunZScan(config)
        runzscan.run()

        incat = fitsio.read(config.catfile, ext=1, lower=True)

        testing.assert_equal(runzscan.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runzscan.cat.ra, incat['ra'], 9)
        testing.assert_almost_equal(runzscan.cat.dec, incat['dec'], 9)
        testing.assert_almost_equal(runzscan.cat.Lambda, [24.503702, 26.959753, 11.826818], 5)
        testing.assert_almost_equal(runzscan.cat.lambda_e, [2.5252883, 4.8358383, 2.3402085], 3)
        testing.assert_almost_equal(runzscan.cat.z_lambda, [0.22806168, 0.32260582, 0.21758862], 5)
        testing.assert_almost_equal(runzscan.cat.z_lambda_e, [0.00631659, 0.01353968, 0.00982372], 5)
        testing.assert_almost_equal(runzscan.cat.ra_opt, [142.09402206, 142.17851073, 142.12751876], 5)
        testing.assert_almost_equal(runzscan.cat.dec_opt, [65.08089019, 65.43947721, 65.10389804])
        testing.assert_almost_equal(runzscan.cat.lambda_opt, [23.435854, 26.868223, 22.200201], 5)
        testing.assert_almost_equal(runzscan.cat.lambda_opt_e, [2.534962 , 4.818528 , 2.3518147], 3)
        testing.assert_almost_equal(runzscan.cat.z_lambda_opt, [0.21612285, 0.32275185, 0.22778341], 5)
        testing.assert_almost_equal(runzscan.cat.z_lambda_opt_e, [0.0091162 , 0.01912573, 0.00890607], 5)


if __name__=='__main__':
    unittest.main()
