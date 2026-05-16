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

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import read_redsequence, redsequence_mstar
from redmapper import compute_zreds, compute_zred
from redmapper import run_zred_catalog, run_zred_pixels

class ZredTestCase(unittest.TestCase):
    """
    Tests of computing zred galaxy photo-z with various methods.
    """

    def test_zred(self):
        """
        Test redmapper.compute_zred and compute_zreds.
        """

        file_path = 'data_for_tests'

        zred_filename = 'test_dr8_pars.fit'
        zredstr = read_redsequence(file_path + '/' + zred_filename)

        galaxy_filename = 'test_dr8_gals_with_zred.fit'
        galaxies = GalaxyCatalog.from_fits_file(file_path + '/' + galaxy_filename)

        galaxies.add_fields([('zred_samp', 'f4', 4)])

        galaxies_input = copy.deepcopy(galaxies)

        # start with the first one...
        compute_zred(zredstr, galaxies[0])

        starttime = time.time()
        compute_zreds(zredstr, galaxies)

        print("Ran %d galaxies in %.3f seconds" % (galaxies.size,
                                                   time.time() - starttime))

        # Only compare galaxies that are brighter than 0.15L* in either old OR new
        # Otherwise, we're just comparing how the codes handle "out-of-range"
        # galaxies, and that does not matter
        mstar_input = redsequence_mstar(zredstr, galaxies_input.zred_uncorr)
        mstar = redsequence_mstar(zredstr, galaxies.zred_uncorr)

        # Don't compare to high redshift because the extrapolation has changed a lot
        ok, = np.where(((galaxies.refmag < (mstar_input - 2.5*np.log10(0.15))) |
                        (galaxies.refmag < (mstar - 2.5*np.log10(0.15)))) &
                       (galaxies.zred_uncorr < 0.58))

        delta_zred_uncorr = galaxies.zred_uncorr[ok] - galaxies_input.zred_uncorr[ok]

        use, = np.where(np.abs(delta_zred_uncorr) < 1e-3)

        testing.assert_array_less(0.9, float(use.size) / float(ok.size))

        delta_zred = galaxies.zred[ok[use]] - galaxies_input.zred[ok[use]]
        use2, = np.where(np.abs(delta_zred) < 1e-3)
        testing.assert_array_less(0.99, float(use2.size) / float(delta_zred.size))

        delta_zred2 = galaxies.zred2[ok[use]] - galaxies_input.zred2[ok[use]]
        use2, = np.where(np.abs(delta_zred) < 1e-3)
        testing.assert_array_less(0.99, float(use2.size) / float(delta_zred2.size))

        delta_zred_uncorr_e = galaxies.zred_uncorr_e[ok[use]] - galaxies_input.zred_uncorr_e[ok[use]]
        use2, = np.where(np.abs(delta_zred_uncorr_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred_uncorr_e.size))

        delta_zred_e = galaxies.zred_e[ok[use]] - galaxies_input.zred_e[ok[use]]
        use2, = np.where(np.abs(delta_zred_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred_e.size))

        delta_zred2_e = galaxies.zred2_e[ok[use]] - galaxies_input.zred2_e[ok[use]]
        use2, = np.where(np.abs(delta_zred2_e) < 1e-3)
        testing.assert_array_less(0.98, float(use2.size) / float(delta_zred2_e.size))

    def test_zred_runcat(self):
        """
        Test redmapper.ZredRunCatalog, computing zreds for all the galaxies in
        a single catalog file.
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir
        outfile = os.path.join(self.test_dir, 'test_zred_out.fits')

        tab = fitsio.read(config.galfile, ext=1, lower=True)
        try:
            galfile = os.path.join(os.path.dirname(config.galfile), tab[0]['filenames'][0].decode())
        except AttributeError:
            galfile = os.path.join(os.path.dirname(config.galfile), tab[0]['filenames'][0])

        run_zred_catalog(config, galfile, outfile)


        # This exercises the reading code
        gals = GalaxyCatalog.from_galfile(galfile, zredfile=outfile)

        self.assertGreater(np.min(gals.zred), 0.0)
        self.assertGreater(np.min(gals.chisq), 0.0)
        self.assertLess(np.max(gals.lkhd), 0.0)

        # And compare to the "official" run...
        config.zredfile = os.path.join(file_path, 'zreds_test', 'dr8_test_zreds_master_table.fit')
        ztab = fitsio.read(config.zredfile, ext=1, lower=True)
        try:
            zredfile = os.path.join(os.path.dirname(config.zredfile), ztab[0]['filenames'][0].decode())
        except AttributeError:
            zredfile = os.path.join(os.path.dirname(config.zredfile), ztab[0]['filenames'][0])

        gals_compare = GalaxyCatalog.from_galfile(galfile, zredfile=zredfile)

        zredstr = read_redsequence(config.parfile)
        mstar_input = redsequence_mstar(zredstr, gals_compare.zred_uncorr)
        mstar = redsequence_mstar(zredstr, gals_compare.zred_uncorr)

        # Don't compare to high redshift because the extrapolation has changed a lot
        ok, = np.where(((gals_compare.refmag < (mstar_input - 2.5*np.log10(0.15))) |
                       (gals_compare.refmag < (mstar - 2.5*np.log10(0.15)))) &
                       (gals_compare.zred_uncorr < 0.58))

        delta_zred_uncorr = gals.zred_uncorr[ok] - gals_compare.zred_uncorr[ok]

        use, = np.where(np.abs(delta_zred_uncorr) < 1e-3)
        testing.assert_array_less(0.98, float(use.size) / float(ok.size))

    def test_zred_runpixels(self):
        """
        Test redmapper.ZredRunPixels, computing zreds for all the galaxies
        in a pixelized galaxy catalog.
        """

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        config.hpix = [2163]
        config.nside = 64
        config.border = 0.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.zredfile = os.path.join(self.test_dir, 'zreds', 'testing_zreds_master_table.fit')

        # FIXME: try an illegal one...

        run_zred_pixels(config)

        # Check that the zred file has been built...

        self.assertTrue(os.path.isfile(config.zredfile))

        # Read in just the galaxies...
        gals0 = GalaxyCatalog.from_galfile(config.galfile, nside=config.nside, hpix=config.hpix, border=config.border)

        # And with the zreds...
        gals = GalaxyCatalog.from_galfile(config.galfile, zredfile=config.zredfile, nside=config.nside, hpix=config.hpix, border=config.border)

        # Confirm they're the same galaxies...
        testing.assert_array_almost_equal(gals0.ra, gals.ra)

        # Confirm the zreds are okay
        self.assertGreater(np.min(gals.zred), 0.0)
        self.assertGreater(np.min(gals.chisq), 0.0)
        self.assertLess(np.max(gals.lkhd), 0.0)

        zredfile = os.path.join(file_path, 'zreds_test', 'dr8_test_zreds_master_table.fit')
        gals_compare = GalaxyCatalog.from_galfile(config.galfile, zredfile=zredfile,
                                                  nside=config.nside, hpix=config.hpix, border=config.border)

        zredstr = read_redsequence(config.parfile)
        mstar_input = redsequence_mstar(zredstr, gals_compare.zred_uncorr)
        mstar = redsequence_mstar(zredstr, gals_compare.zred_uncorr)

        # Don't compare to high redshift because the extrapolation has changed a lot
        ok, = np.where(((gals_compare.refmag < (mstar_input - 2.5*np.log10(0.15))) |
                        (gals_compare.refmag < (mstar - 2.5*np.log10(0.15)))) &
                       (gals_compare.zred_uncorr < 0.58))

        delta_zred_uncorr = gals.zred_uncorr[ok] - gals_compare.zred_uncorr[ok]

        use, = np.where(np.abs(delta_zred_uncorr) < 1e-3)
        testing.assert_array_less(0.98, float(use.size) / float(ok.size))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
