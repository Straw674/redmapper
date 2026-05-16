import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Entry
from redmapper import Cluster
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import read_background
from redmapper import read_redsequence
from redmapper.mask import get_mask, select_maskgals_sample, compute_maskgals_mark
from redmapper import depthmap
from redmapper.utilities import calc_theta_i

class ClusterTestCase(unittest.TestCase):
    """
    This file tests multiple features of the redmapper.Cluster class, including
    background and richness computation.
    """
    def runTest(self):
        """
        Run the ClusterTest
        """

        rng = np.random.RandomState(12345)

        file_path = 'data_for_tests'

        cluster = Cluster()

        conf_filename = 'testconfig.yaml'
        cluster.config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'

        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        cluster.set_neighbors(neighbors)

        zred_filename = 'test_dr8_pars.fit'
        cluster.zredstr = read_redsequence(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        cluster.bkg = read_background('%s/%s' % (file_path, bkg_filename))

        hdr=fitsio.read_header(file_path+'/'+filename,ext=1)
        cluster.redshift = hdr['Z']
        richness_compare = hdr['LAMBDA']
        richness_compare_err = hdr['LAMBDA_E']
        scaleval_compare = hdr['SCALEVAL']
        cpars_compare = np.array([hdr['CPARS0'], hdr['CPARS1'], hdr['CPARS2'], hdr['CPARS3']])
        cval_compare = hdr['CVAL']
        mstar_compare = hdr['MSTAR']
        cluster.ra = hdr['RA']
        cluster.dec = hdr['DEC']

        mask = get_mask(cluster.config, rng=rng)
        mask['maskgals'], mask['maskgal_index'] = select_maskgals_sample(cluster.config, mask['maskgals_all'], mask['rng'], maskgal_index=0)
        mask['maskgals'].mark = compute_maskgals_mark(mask['mask_data'], cluster, mask['maskgals'], rng=mask['rng'], config=cluster.config)

        depth_data = depthmap.read_depth_map(cluster.config)
        depthmap.compute_maskdepth(depth_data, mask['maskgals'], cluster.ra, cluster.dec, cluster.mpc_scale)

        # Test the NFW profile on its own
        #  (this works to 5 decimal places because of the 2*pi*r scaling)
        nfw_python = cluster._calc_radial_profile()
        testing.assert_almost_equal(nfw_python, neighbors.nfw/(2.*np.pi*neighbors.r),5)

        # Test the background
        #  Note that this uses the input chisq values
        bkg_python = cluster.calc_bkg_density(cluster.neighbors.r,
                                              cluster.neighbors.chisq,
                                              cluster.neighbors.refmag)
        # this is cheating here...
        to_test, = np.where((cluster.neighbors.refmag < cluster.bkg['refmagbins'][-1]))

        richness = cluster.calc_richness(mask)

        # these are regression tests.  Various mask issues make the matching
        #  to idl for the time being
        testing.assert_allclose(cluster.Lambda, 24.396917, rtol=1e-3)
        testing.assert_allclose(cluster.lambda_e, 2.5160403, rtol=1e-3)


if __name__=='__main__':
    unittest.main()

