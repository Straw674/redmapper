import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import os
from numpy import random

from redmapper import Cluster
from redmapper import Configuration
from redmapper import CenteringWcenZred, CenteringBCG, CenteringRandom, CenteringRandomSatellite
from redmapper import GalaxyCatalog
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import ZredBackground
from redmapper import ZlambdaCorrectionPar


class CenteringTestCase(unittest.TestCase):
    """
    Test application of the centering models (CenteringWcenZred, CenteringBCG,
    CenteringRandom, CenteringRandomSatelliate).
    """

    def test_wcenzred(self):
        """
        Test running of CenteringWcenZred.
        """
        file_path = 'data_for_tests'

        cluster = self._setup_cluster()
        tempcat = fitsio.read(os.path.join(file_path, 'test_wcen_zred_data.fit'))

        corr_filename = 'test_dr8_zlambdacorr.fit'
        zlambda_corr = ZlambdaCorrectionPar(os.path.join(file_path, 'test_dr8_zlambdacorr.fit'), zlambda_pivot=30.0)
        zlambda_corr = ZlambdaCorrectionPar(file_path + '/' + corr_filename, zlambda_pivot=30.0)

        # And the meat of it...

        rng = np.random.RandomState(12345)

        cent = CenteringWcenZred(cluster, zlambda_corr=zlambda_corr, rng=rng)
        cent.find_center()

        testing.assert_almost_equal(cent.p_cen, tempcat[0]['PCEN'][tempcat[0]['GOOD']], 5)
        testing.assert_almost_equal(cent.q_cen, tempcat[0]['QCEN'][tempcat[0]['GOOD']], 4)
        testing.assert_almost_equal(cent.p_sat, tempcat[0]['PSAT'], 4)
        testing.assert_almost_equal(cent.p_fg, tempcat[0]['PFG'], 4)
        testing.assert_array_equal(cent.index, tempcat[0]['USE'][tempcat[0]['GOOD']])

    def test_wcenzred_zspec(self):
        """Test running CenteringWcenZred with zspecs."""
        file_path = 'data_for_tests'

        cluster = self._setup_cluster(add_zspec=True)
        cluster.config.centering_use_zspec = True
        tempcat = fitsio.read(os.path.join(file_path, 'test_wcen_zred_data.fit'))

        corr_filename = 'test_dr8_zlambdacorr.fit'
        zlambda_corr = ZlambdaCorrectionPar(os.path.join(file_path, 'test_dr8_zlambdacorr.fit'), zlambda_pivot=30.0)
        zlambda_corr = ZlambdaCorrectionPar(file_path + '/' + corr_filename, zlambda_pivot=30.0)

        # And the meat of it...

        rng = np.random.RandomState(12345)

        cent = CenteringWcenZred(cluster, zlambda_corr=zlambda_corr, rng=rng)
        cent.find_center()

        testing.assert_almost_equal(cent.p_cen, tempcat[0]['PCEN'][tempcat[0]['GOOD']], 5)
        testing.assert_almost_equal(cent.q_cen, tempcat[0]['QCEN'][tempcat[0]['GOOD']], 4)
        testing.assert_almost_equal(cent.p_sat, tempcat[0]['PSAT'], 4)
        testing.assert_almost_equal(cent.p_fg, tempcat[0]['PFG'], 4)
        testing.assert_array_equal(cent.index, tempcat[0]['USE'][tempcat[0]['GOOD']])

    def test_bcg(self):
        """
        Test running of CenteringBcg.
        """
        cluster = self._setup_cluster()

        rng = np.random.RandomState(12345)

        cent = CenteringBCG(cluster, rng=rng)
        cent.find_center()

        self.assertEqual(cent.maxind, 72)
        self.assertEqual(cent.ngood, 1)
        testing.assert_almost_equal(cent.ra, 150.55890608)
        testing.assert_almost_equal(cent.dec, 20.53794937)
        testing.assert_almost_equal(cent.p_cen[0], 1.0)
        testing.assert_almost_equal(cent.q_cen[0], 1.0)
        testing.assert_almost_equal(cent.p_sat[0], 0.0)

    def test_bcg_zspec(self):
        """
        Test running of CenteringBcg with zspec.
        """
        cluster = self._setup_cluster(add_zspec=True)
        cluster.config.centering_use_zspec = True

        rng = np.random.RandomState(12345)

        cent = CenteringBCG(cluster, rng=rng)
        cent.find_center()

        self.assertEqual(cent.maxind, 72)
        self.assertEqual(cent.ngood, 1)
        testing.assert_almost_equal(cent.ra, 150.55890608)
        testing.assert_almost_equal(cent.dec, 20.53794937)
        testing.assert_almost_equal(cent.p_cen[0], 1.0)
        testing.assert_almost_equal(cent.q_cen[0], 1.0)
        testing.assert_almost_equal(cent.p_sat[0], 0.0)

    def test_random(self):
        """
        Test running of CenteringRandom.
        """

        rng = np.random.RandomState(12345)

        cluster = self._setup_cluster()

        cent = CenteringRandom(cluster, rng=rng)
        cent.find_center()

        self.assertEqual(cent.maxind, -1)
        self.assertEqual(cent.ngood, 1)
        testing.assert_almost_equal(cent.ra[0], 150.57049502423266)
        testing.assert_almost_equal(cent.dec[0], 20.604521924053167)
        testing.assert_almost_equal(cent.p_cen[0], 1.0)
        testing.assert_almost_equal(cent.q_cen[0], 1.0)
        testing.assert_almost_equal(cent.p_sat[0], 0.0)

    def test_randsat(self):
        """
        Test running of CenteringRandomSatellite.
        """

        rng = np.random.RandomState(12345)

        cluster = self._setup_cluster()

        cent = CenteringRandomSatellite(cluster, rng=rng)
        cent.find_center()

        # Confirmed that the distribution is correct, this just checks for regression

        self.assertEqual(cent.maxind, 721)
        self.assertEqual(cent.ngood, 1)
        testing.assert_almost_equal(cent.ra[0], 150.67510227)
        testing.assert_almost_equal(cent.dec[0], 20.48011092)
        testing.assert_almost_equal(cent.p_cen[0], 1.0)
        testing.assert_almost_equal(cent.q_cen[0], 1.0)
        testing.assert_almost_equal(cent.p_sat[0], 0.0)

    def _setup_cluster(self, add_zspec=False):
        """
        Set up the cluster to run through the centering code.
        """
        file_path = 'data_for_tests'

        cluster = Cluster()

        cluster.config = Configuration(os.path.join(file_path, 'testconfig.yaml'))

        tempcat = fitsio.read(os.path.join(file_path, 'test_wcen_zred_data.fit'))

        dtype = [('RA', 'f8'),
                 ('DEC', 'f8'),
                 ('DIST', 'f4'),
                 ('R', 'f4'),
                 ('P', 'f4'),
                 ('PFREE', 'f4'),
                 ('PMEM', 'f4'),
                 ('MAG', 'f4', 5),
                 ('MAG_ERR', 'f4', 5),
                 ('REFMAG', 'f4'),
                 ('REFMAG_ERR', 'f4'),
                 ('CHISQ', 'f4'),
                 ('ZRED', 'f4'),
                 ('ZRED_E', 'f4'),
                 ('ZRED_CHISQ', 'f4')]

        if add_zspec:
            dtype.extend([('ZSPEC', 'f4'),
                          ('ZSPEC_ERR', 'f4')])

        temp_neighbors = np.zeros(tempcat[0]['RAS'].size, dtype=dtype)

        temp_neighbors['RA'] = tempcat[0]['RAS']
        temp_neighbors['DEC'] = tempcat[0]['DECS']
        temp_neighbors['R'] = tempcat[0]['R']
        temp_neighbors['P'] = tempcat[0]['PVALS']
        temp_neighbors['PFREE'] = tempcat[0]['WVALS']
        temp_neighbors['PMEM'] = tempcat[0]['WTVALS']
        temp_neighbors['REFMAG'] = tempcat[0]['REFMAG_TOTAL']
        temp_neighbors['ZRED'] = tempcat[0]['GZREDS']
        temp_neighbors['ZRED_E'] = tempcat[0]['GZREDE']
        temp_neighbors['ZRED_CHISQ'] = tempcat[0]['GCHISQ']

        if add_zspec:
            temp_neighbors['ZSPEC'] = tempcat[0]['GZREDS']
            temp_neighbors['ZSPEC_ERR'] = tempcat[0]['GZREDE']

        temp_neighbors['DIST'] = tempcat[0]['R'] / (np.radians(1.) * cluster.config.cosmo.Da(0, tempcat[0]['ZCLUSTER']))

        neighbors = GalaxyCatalog(temp_neighbors)
        cluster.set_neighbors(neighbors)

        zred_filename = 'test_dr8_pars.fit'
        cluster.zredstr = RedSequenceColorPar(os.path.join(file_path, 'test_dr8_pars.fit'), fine=True, zrange=[0.25, 0.35])

        cluster.bkg = Background(os.path.join(file_path, 'test_bkg.fit'))
        cluster.zredbkg = ZredBackground(os.path.join(file_path, 'test_bkg.fit'))

        cluster.redshift = tempcat[0]['ZCLUSTER']
        cluster.ra = tempcat[0]['RAC']
        cluster.dec = tempcat[0]['DECC']
        cluster.r_lambda = 1.0 * (tempcat[0]['LAMBDA'] / 100.0)**0.2
        cluster.Lambda = tempcat[0]['LAMBDA']
        cluster.scaleval = tempcat[0]['SCALEVAL']

        return cluster


if __name__=='__main__':
    unittest.main()
