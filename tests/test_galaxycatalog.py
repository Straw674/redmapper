import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import hpgeom as hpg
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import GalaxyCatalogMaker
from redmapper import Catalog, Entry


class GalaxyCatalogTestCase(unittest.TestCase):
    """
    Tests for redmapper.GalaxyCatalog class, including reading full catalogs,
    reading sub-regions, and matching neighbors.
    """
    def test_galaxycatalog_read(self):
        """
        Run redmapper.GalaxyCatalog tests,
        """

        file_path = 'data_for_tests'

        galfile = 'pixelized_dr8_test/dr8_test_galaxies_master_table.fit'

        gals_all = GalaxyCatalog.from_galfile(file_path + '/' + galfile)

        # check that we got the expected number...
        testing.assert_equal(gals_all.size, 14449)

        # read in a subregion, no border
        hpix = 2163
        gals_sub = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                              hpix=hpix, nside=64)

        ipring_all = hpg.angle_to_pixel(64, gals_all.ra, gals_all.dec, nest=False)
        use, = np.where(ipring_all == hpix)
        testing.assert_equal(gals_sub.size, use.size)

        # read in a subregion, no border, with tempfile
        gals_sub = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                              hpix=hpix, nside=64, use_tempfile=True)
        testing.assert_equal(gals_sub.size, use.size)

        # Read in a subregion that's made of two pixels, no border
        hpix = [2163, 2296]
        gals_sub = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                              hpix=hpix, nside=64)
        a, b = esutil.numpy_util.match(hpix, ipring_all)

        testing.assert_equal(gals_sub.size, a.size)

        # Read in the same subregion with tempfile
        gals_sub = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                              hpix=hpix, nside=64, use_tempfile=True)
        testing.assert_equal(gals_sub.size, a.size)

        # read in a subregion, with border
        gals_sub = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                              hpix=9218, nside=128, border=0.1)

        # this isn't really a big enough sample catalog to fully test...
        testing.assert_equal(gals_sub.size, 2511)

        # and test the matching...

        indices, dists = gals_all.match_one(140.5, 65.0, 0.2)
        testing.assert_equal(indices.size, 521)
        testing.assert_array_less(dists, 0.2)

        i0, i1, dists = gals_all.match_many([140.5,141.2],
                                            [65.0, 65.2], [0.2,0.1])
        testing.assert_equal(i0.size, 666)
        test, = np.where(i0 == 0)
        testing.assert_equal(test.size, 521)
        testing.assert_array_less(dists[test], 0.2)
        test, = np.where(i0 == 1)
        testing.assert_equal(test.size, 666 - 521)
        testing.assert_array_less(dists[test], 0.1)

        # read in a subregion, with border, with tempfile
        gals_sub2 = GalaxyCatalog.from_galfile(os.path.join(file_path, galfile),
                                               hpix=9218, nside=128, border=0.1, use_tempfile=True)
        aa, bb = esutil.numpy_util.match(gals_sub.id, gals_sub2.id)
        testing.assert_equal(aa.size, gals_sub.size)

    def test_galaxycatalog_create(self):
        """
        Run `redmapper.GalaxyCatalogMaker` tests.
        """

        # Make a test directory
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        # Test 1: read in catalog, write it to a single file, and then
        # try to split it up
        # make sure that it makes it properly, and that the output number
        # of files is the same as input number of files.

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside)
        maker.append_galaxies(gals._ndarray)
        maker.finalize_catalog()

        tab2 = Entry.from_fits_file(os.path.join(self.test_dir, 'test_working_master_table.fit'))
        self.assertEqual(tab.nside, tab2.nside)
        self.assertEqual(tab.filenames.size, tab2.filenames.size)
        self.assertEqual(tab2.has_truth, 0)
        self.assertEqual(tab2.has_zspec, 0)

        for filename in tab2.filenames:
            try:
                fname = os.path.join(self.test_dir, filename.decode())
            except AttributeError:
                fname = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.isfile(fname))

        # Test 2: Make a catalog that has an incomplete dtype
        dtype = [('id', 'i8'),
                 ('ra', 'f8')]
        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test'), info_dict)
        testgals = np.zeros(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 3: make a catalog that has the wrong number of magnitudes
        dtype = GalaxyCatalogMaker.get_galaxy_dtype(3)
        testgals = np.zeros(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 4: make a catalog that has some NaNs
        dtype = GalaxyCatalogMaker.get_galaxy_dtype(info_dict['NMAG'])
        testgals = np.ones(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        testgals['mag'][0, 1] = np.nan
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 5: make a catalog that has ra/dec out of range
        testgals = np.ones(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        testgals['ra'][1] = -1.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        testgals = np.ones(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        testgals['dec'][1] = -100.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 6: make a catalog that has mag > 90.0
        testgals = np.ones(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        testgals['mag'][0, 1] = 100.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 7: make a catalog that has mag_err == 0.0
        testgals = np.ones(10, dtype=dtype)
        testgals['id'] = np.arange(10)
        testgals['mag_err'][0, 1] = 0.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 8: make a catalog that has duplicate ids
        testgals = np.ones(10, dtype=dtype)
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

    def test_galaxycatalog_create_parallel(self):
        """
        Run `redmapper.GalaxyCatalogMaker` parallel tests.
        """

        # Make a test directory
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        # Not truly parallel, but possible to use two makers at once here.
        maker1 = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, parallel=True)
        maker2 = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, parallel=True)
        maker1.append_galaxies(gals._ndarray[0: gals.size//2])
        maker2.append_galaxies(gals._ndarray[gals.size//2: ])
        maker1.finalize_catalog()
        maker2.finalize_catalog()

        tab2 = Entry.from_fits_file(os.path.join(self.test_dir, 'test_working_master_table.fit'))
        self.assertEqual(tab.nside, tab2.nside)
        self.assertEqual(tab.filenames.size, tab2.filenames.size)

        for filename in tab2.filenames:
            try:
                fname = os.path.join(self.test_dir, filename.decode())
            except AttributeError:
                fname = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.isfile(fname))

    def test_galaxycatalog_create_with_truth(self):
        # Make a test directory
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        # Raise if we try to create a truth catalog here
        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, ingest_truth=True)
        self.assertRaises(RuntimeError, maker.append_galaxies, gals._ndarray)

        new_dtype = [('ztrue', 'f4'),
                     ('m200', 'f4'),
                     ('central', 'i2'),
                     ('halo_id', 'i8')]

        gals.add_fields(new_dtype)

        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, ingest_truth=True)
        maker.append_galaxies(gals._ndarray)
        maker.finalize_catalog()

        tab2 = Entry.from_fits_file(os.path.join(self.test_dir, 'test_working_master_table.fit'))
        self.assertEqual(tab.nside, tab2.nside)
        self.assertEqual(tab.filenames.size, tab2.filenames.size)

        self.assertEqual(tab2.has_truth, 1)
        self.assertEqual(tab2.has_zspec, 0)

    def test_galaxycatalog_create_with_zspec(self):
        # Make a test directory
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        # Raise if we try to create a truth catalog here
        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, ingest_zspec=True)
        self.assertRaises(RuntimeError, maker.append_galaxies, gals._ndarray)

        new_dtype = [('zspec', 'f4'),
                     ('zspec_err', 'f4')]

        gals.add_fields(new_dtype)

        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside, ingest_zspec=True)
        maker.append_galaxies(gals._ndarray)
        maker.finalize_catalog()

        tab2 = Entry.from_fits_file(os.path.join(self.test_dir, 'test_working_master_table.fit'))
        self.assertEqual(tab.nside, tab2.nside)
        self.assertEqual(tab.filenames.size, tab2.filenames.size)

        self.assertEqual(tab2.has_truth, 0)
        self.assertEqual(tab2.has_zspec, 1)

    def test_galaxycatalog_create_onepix(self):
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        # Cut down to one healpix.
        indices = hpg.angle_to_pixel(tab.nside, gals.ra, gals.dec, nest=False)
        use, = np.where(indices == indices[0])
        gals = gals[use]

        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside)
        maker.append_galaxies(gals._ndarray)
        maker.finalize_catalog()

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
