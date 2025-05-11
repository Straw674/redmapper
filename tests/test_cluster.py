import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper import Entry
from redmapper import Cluster
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import Background
from redmapper import RedSequenceColorPar
from redmapper import HPMask
from redmapper import DepthMap
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

        # all new...

        random.seed(seed=12345)

        file_path = 'data_for_tests'

        cluster = Cluster()

        conf_filename = 'testconfig.yaml'
        cluster.config = Configuration(file_path + '/' + conf_filename)

        filename = 'test_cluster_members.fit'

        neighbors = GalaxyCatalog.from_fits_file(file_path + '/' + filename)

        cluster.set_neighbors(neighbors)

        zred_filename = 'test_dr8_pars.fit'
        cluster.zredstr = RedSequenceColorPar(file_path + '/' + zred_filename, fine=True)

        bkg_filename = 'test_bkg.fit'
        cluster.bkg = Background('%s/%s' % (file_path, bkg_filename))

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

        mask = HPMask(cluster.config)
        maskgal_index = mask.select_maskgals_sample(maskgal_index=0)
        mask.set_radmask(cluster)

        depthstr = DepthMap(cluster.config)
        depthstr.calc_maskdepth(mask.maskgals, cluster.ra, cluster.dec, cluster.mpc_scale)

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
        to_test, = np.where((cluster.neighbors.refmag < cluster.bkg.refmagbins[-1]))

        seed = 0
        random.seed(seed = 0)

        richness = cluster.calc_richness(mask)

        # these are regression tests.  Various mask issues make the matching
        #  to idl for the time being
        testing.assert_almost_equal(cluster.Lambda, 24.366407, 4)
        testing.assert_almost_equal(cluster.lambda_e, 2.5137918, 4)

        return


if __name__=='__main__':
    unittest.main()

