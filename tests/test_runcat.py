import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Cluster
from redmapper import ClusterCatalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import DataObject
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import HPMask
from redmapper import DepthMap
from redmapper import RunCatalog

class RuncatTestCase(unittest.TestCase):
    """
    Tests of redmapper.RunCatalog, which computes richness for an input catalog with
    ra/dec/z.
    """
    def runTest(self):
        """
        Run the redmapper.RunCatalog tests.
        """

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(file_path + '/' + conffile)
        config.catfile = file_path + '/' + catfile
        config.bkg_local_compute = True
        config.randomseed = 12345

        runcat = RunCatalog(config)

        runcat.run(do_percolation_masking=False)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_allclose(runcat.cat.Lambda, [24.16809, 26.92924, 13.35232], rtol=2e-3)
        testing.assert_allclose(runcat.cat.lambda_e, [2.50009, 4.85051, 2.46239], rtol=2e-3)
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278546, 0.3225739, 0.2176394], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063102, 0.0135351, 0.0098461], 5)
        testing.assert_allclose(runcat.cat.bkg_local, [1.2287012, 1.6885487, 1.7221997], rtol=2e-3)

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.16809, 26.92924, -1.], 5)
        testing.assert_allclose(runcat.cat.lambda_e, [2.50009,  4.85051, -1], rtol=2e-3)
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.2278544,  0.3225641, -1.], 5)
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.0063079,  0.0135317, -1.], 5)
        testing.assert_allclose(runcat.cat.bkg_local, [1.2287, 1.68855, 0.], rtol=2e-3)

if __name__=='__main__':
    unittest.main()
