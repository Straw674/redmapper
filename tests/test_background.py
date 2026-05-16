import unittest
import numpy.testing as testing
import numpy as np
import os
import fitsio
import tempfile
import shutil

from redmapper import read_background, read_zred_background, compute_background, compute_zred_background, generate_background, generate_zred_background
from redmapper import Configuration

class BackgroundTestCase(unittest.TestCase):
    """
    Tests of the redmapper background functions.
    """

    def test_readbkg(self):
        """
        Test reading of a redmapper background file.
        """
        file_name, file_path = 'test_bkg.fit', 'data_for_tests'
        # test that we fail if we try a non-existent file
        self.assertRaises(IOError, read_background, 'nonexistent.fit')
        # test that we fail if we read a non-fits file
        self.assertRaises(IOError, read_background,'%s/testconfig.yaml' % (file_path))
        # test that we fail if we try a file without the right header info
        self.assertRaises(IOError, read_background,
                          '%s/test_dr8_pars.fit' % (file_path))
        bkg = read_background('%s/%s' % (file_path, file_name))

        # test creation of lookup table
        inputs = [(172,15,64), (323,3,103), (9,19,21), (242,4,87),
                  (70,12,58), (193,6,39), (87,14,88), (337,5,25), (333,8,9)]
        py_outputs = np.array([bkg['sigma_g'][idx] for idx in inputs])
        idl_outputs = np.array([0.32197464, 6.4165196, 0.0032830855, 
                                1.4605126, 0.0098356586, 0.79848081, 
                                0.011284498, 9.3293247, 8.7064905])
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=1)

        # test functionality of lookup table
        z = 0.23185321
        chisq = np.array([0.13315917, 3.57059131, 3.71567741, 2.46307987,
                          9.16647519, 8.24240144, -1., 1.19503491])
        refmag = np.array([1000., 15.05129281, 16.81049236, 18.07566359,
                        19.88279, 15.56617587, 18.55626717, 15.00271158])

        py_outputs = compute_background(bkg, z, chisq, refmag)
        idl_outputs = np.array([np.inf, 0.0012997627, 0.56412143, 6.4126010,
                                43.4550, 0.012194233, np.inf, np.inf])
        # idl_ouputs[4] = 42.555183
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=2)

        ###########################################
        ## And test the zred background code
        ###########################################

        zredbkg = read_zred_background('%s/%s' % (file_path, file_name))

        # test creation of lookup table
        inputs = [(60, 50), (200, 100), (300, 120)]
        py_outputs = np.array([zredbkg['sigma_g'][idx] for idx in inputs])
        idl_outputs = np.array([1.16810, 28.4379, 373.592])

        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=1)

        # test functionality of lookup table
        zred = np.array([0.2154, 0.2545, 0.2876])
        refmag = np.array([18.015,18.576,19.234])
        idl_outputs = np.array([710.17102,1000.1127,1718.0394])
        py_outputs = compute_zred_background(zredbkg, zred, refmag)
        testing.assert_almost_equal(py_outputs, idl_outputs, decimal=1)


    def test_generatebkg(self):
        """
        Test generation of a background file.
        """
        config_file = os.path.join('data_for_tests', 'testconfig.yaml')

        config = Configuration(config_file)
        config.calib_nproc = 1

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.bkgfile = os.path.join(config.outpath, '%s_testbkg.fit' % (config.outbase))
        config.zrange = [0.1, 0.2]

        generate_background(config, clobber=True)

        self.assertTrue(os.path.isfile(config.bkgfile))

        bkg = fitsio.read(config.bkgfile, ext='CHISQBKG')

        # Some spot-testing...
        testing.assert_equal(bkg[0]['sigma_g'].shape, (48, 40, 5))
        testing.assert_equal(bkg[0]['sigma_lng'].shape, (48, 40, 5))
        testing.assert_almost_equal(bkg[0]['sigma_g'][30, 20, 2], 2.8444533, 4)
        testing.assert_almost_equal(bkg[0]['sigma_g'][30, 10, 3], 7.4324584, 4)
        testing.assert_almost_equal(bkg[0]['sigma_lng'][30, 10, 3], 3.7618985, 4)
        testing.assert_almost_equal(bkg[0]['sigma_lng'][45, 10, 3], 0.0)

        #if os.path.exists(test_dir):
        #    shutil.rmtree(test_dir, True)

    def test_generatezredbkg(self):
        """
        Test generation of a zred background file.
        """

        config_file = os.path.join('data_for_tests', 'testconfig.yaml')

        config = Configuration(config_file)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.bkgfile = os.path.join(config.outpath, '%s_testbkg.fit' % (config.outbase))
        config.zrange = [0.1, 0.2]

        # First test without a zred file ...
        self.assertRaises(RuntimeError, generate_zred_background, config)

        # And now fix it ...
        config.zredfile = os.path.join('data_for_tests', 'zreds_test', 'dr8_test_zreds_master_table.fit')

        generate_zred_background(config, clobber=True)


        self.assertTrue(os.path.isfile(config.bkgfile))

        zbkg = fitsio.read(config.bkgfile, ext='ZREDBKG')

        # Some spot-testing...
        # (The numbers have been checked to be consistent with the full run tested above
        #  but can't be directly compared because this is much noisier)
        testing.assert_equal(zbkg[0]['sigma_g'].shape, (48, 10))
        testing.assert_almost_equal(zbkg[0]['sigma_g'][30, 5], 620.0223999, decimal=4)
        testing.assert_almost_equal(zbkg[0]['sigma_g'][47, 8], 30501.8398438, decimal=4)
        testing.assert_almost_equal(zbkg[0]['sigma_g'][30, 0], 384.3362732, decimal=3)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)



if __name__=='__main__':
    unittest.main()
