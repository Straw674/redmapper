import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Cluster
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import read_background
from redmapper import read_redsequence, redsequence_mstar
from redmapper.mask import get_mask, select_maskgals_sample, compute_maskgals_mark
from redmapper import depthmap
from redmapper import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from redmapper.zlambda import _zlambda_calc_gaussian_err

class ZlambdaTestCase(unittest.TestCase):
    """
    Tests of redmapper.zlambda z_lambda computation code.
    """
    def runTest(self):
        """
        Run tests on redmapper.zlambda
        """
        rng = np.random.RandomState(12345)

        file_path = 'data_for_tests'

        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'
        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        zred_filename = 'test_dr8_pars.fit'
        zredstr = read_redsequence(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        bkg = read_background('%s/%s' % (file_path, bkg_filename))

        cluster = Cluster(config=config, zredstr=zredstr, bkg=bkg, neighbors=neighbors)

        hdr=fitsio.read_header(file_path+'/'+filename,ext=1)
        cluster.redshift = hdr['Z']
        richness_compare = hdr['LAMBDA']
        richness_compare_err = hdr['LAMBDA_E']
        cluster.ra = hdr['RA']
        cluster.dec = hdr['DEC']

        #Set up the mask
        mask = get_mask(cluster.config, rng=rng)
        mask['maskgals'], mask['maskgal_index'] = select_maskgals_sample(cluster.config, mask['maskgals_all'], mask['rng'], maskgal_index=0)
        mask['maskgals'].mark = compute_maskgals_mark(mask['mask_data'], cluster, mask['maskgals'], rng=mask['rng'], config=cluster.config)

        #depthstr
        depth_data = depthmap.read_depth_map(cluster.config)
        depthmap.compute_maskdepth(depth_data, mask['maskgals'], cluster.ra, cluster.dec, cluster.mpc_scale)

        cluster.neighbors.dist = np.degrees(cluster.neighbors.r/cluster.cosmo.Dl(0,cluster.redshift))

        # make a zlambda computation
        z_lambda, z_lambda_e, pzbins, pz, niter = compute_zlambda(cluster, mask, cluster.redshift, calc_err=True, calcpz=True)

        # I am not sure why this isn't repeatable better than this
        testing.assert_almost_equal(z_lambda, 0.22666427, 6)
        testing.assert_almost_equal(z_lambda_e, 0.00443601, 4)

        # zlambda_err test
        # We need to set up the neighbors and state for the helper function
        cluster.redshift = z_lambda
        maxrad = 1.2 * cluster.r0 * 3.**cluster.beta
        in_r, = np.where(cluster.neighbors.r < maxrad)
        cluster.calc_richness(mask, calc_err=False, index=in_r)

        wtvals_mod = cluster.neighbors.pcol
        state = {'zlambda_fail': False, 'targval': 0.0}
        maxmag = redsequence_mstar(zredstr, z_lambda) - 2.5*np.log10(config.lval_reference)
        from redmapper.zlambda import _zlambda_select_neighbors
        _zlambda_select_neighbors(state, cluster, wtvals_mod, maxrad, maxmag)
        z_lambda_err = _zlambda_calc_gaussian_err(state, cluster, z_lambda)

        testing.assert_almost_equal(z_lambda_err, 0.006397615717245883, 5)

        # and test the correction on its own
        corr_filename = 'test_dr8_zlambdacorr.fit'

        zlambda_corr_data = read_zlambda_correction(file_path + '/' + corr_filename, zlambda_pivot=30.0)

        zlam_in = 0.227865
        zlam_e_in = 0.00629995
        zlam_out = 0.228654
        zlam_e_out = 0.00840213

        zlam_new, zlam_e_new = apply_zlambda_correction(zlambda_corr_data, 24.5, zlam_in, zlam_e_in)

        testing.assert_almost_equal(zlam_new, zlam_out, 5)
        testing.assert_almost_equal(zlam_e_new, zlam_e_out, 5)


if __name__=='__main__':
    unittest.main()
