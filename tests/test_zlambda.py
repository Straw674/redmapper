import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Cluster
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import Background
from redmapper import RedSequenceColorPar
from redmapper import HPMask
from redmapper import DepthMap
from redmapper import Zlambda
from redmapper import ZlambdaCorrectionPar

class ZlambdaTestCase(unittest.TestCase):
    """
    Tests of redmapper.Zlambda z_lambda computation code.
    """
    def runTest(self):
        """
        Run tests on redmapper.Zlambda
        """
        rng = np.random.RandomState(12345)

        file_path = 'data_for_tests'

        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'
        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        zred_filename = 'test_dr8_pars.fit'
        zredstr = RedSequenceColorPar(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        bkg = Background('%s/%s' % (file_path, bkg_filename))

        cluster = Cluster(config=config, zredstr=zredstr, bkg=bkg, neighbors=neighbors)

        hdr=fitsio.read_header(file_path+'/'+filename,ext=1)
        cluster.redshift = hdr['Z']
        richness_compare = hdr['LAMBDA']
        richness_compare_err = hdr['LAMBDA_E']
        cluster.ra = hdr['RA']
        cluster.dec = hdr['DEC']

        #Set up the mask
        mask = HPMask(cluster.config, rng=rng)
        maskgal_index = mask.select_maskgals_sample(maskgal_index=0)
        mask.set_radmask(cluster)

        #depthstr
        depthstr = DepthMap(cluster.config)
        depthstr.calc_maskdepth(mask.maskgals, cluster.ra, cluster.dec, cluster.mpc_scale)

        cluster.neighbors.dist = np.degrees(cluster.neighbors.r/cluster.cosmo.Dl(0,cluster.redshift))

        # make a zlambda object
        zlam = Zlambda(cluster)

        z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, mask, calc_err=True, calcpz=True)

        # I am not sure why this isn't repeatable better than this
        testing.assert_almost_equal(cluster.z_lambda, 0.22666427, 6)
        testing.assert_almost_equal(cluster.z_lambda_err, 0.00443601, 4)

        # zlambda_err test
        z_lambda_err = zlam._zlambda_calc_gaussian_err(cluster.z_lambda)

        testing.assert_almost_equal(z_lambda_err, 0.0063738347, 5)

        # and test the correction on its own
        corr_filename = 'test_dr8_zlambdacorr.fit'

        zlambda_corr = ZlambdaCorrectionPar(file_path + '/' + corr_filename, zlambda_pivot=30.0)

        zlam_in = 0.227865
        zlam_e_in = 0.00629995
        zlam_out = 0.228654
        zlam_e_out = 0.00840213

        zlam_new, zlam_e_new = zlambda_corr.apply_correction(24.5, zlam_in, zlam_e_in)

        testing.assert_almost_equal(zlam_new, zlam_out, 5)
        testing.assert_almost_equal(zlam_e_new, zlam_e_out, 5)


if __name__=='__main__':
    unittest.main()
