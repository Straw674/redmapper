import unittest, os
import numpy.testing as testing
import numpy as np

import redmapper

class ReadConfigTestCase(unittest.TestCase):
    """
    Tests for reading the redmapper.Configuration class.
    """

    def runTest(self):
        """
        Run the ReadConfig tests.
        """
        file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        config = redmapper.Configuration('%s/%s' % (file_path, file_name))

        self.assertEqual(config.galfile,'./data_for_tests/pixelized_dr8_test/dr8_test_galaxies_master_table.fit')
        self.assertEqual(config.specfile,'./data_for_tests/test_dr8_spec.fit')
        self.assertEqual(config.parfile,'./data_for_tests/test_dr8_pars.fit')
        self.assertEqual(config.bkgfile,'./data_for_tests/test_bkg.fit')
        testing.assert_almost_equal(config.zrange, [0.05, 0.60])
        self.assertEqual(config.outbase,"dr8_testing")
        self.assertEqual(config.chisq_max,20.0)
        self.assertEqual(config.lval_reference,0.2)
        self.assertEqual(config.mask_mode,3)
        self.assertEqual(config.nmag,5)
        testing.assert_almost_equal(config.area, 3.3571745808326572)
        testing.assert_almost_equal(config.b[0], 1.39999998e-10)
        self.assertEqual(config.b.size, config.nmag)
        testing.assert_almost_equal(config.zeropoint,22.5)
        self.assertEqual(config.ref_ind,3)
        # all other values in the config are None

if __name__=='__main__':
    unittest.main()
