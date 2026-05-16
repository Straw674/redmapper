import unittest
import numpy.testing as testing
import numpy as np
import fitsio

import redmapper

class ChisqColorTestCase(unittest.TestCase):
    """
    Test computation of chisq using color-based red-sequence model.
    """
    def runTest(self):
        """
        Run the ChisqColor Test.  Chisq computation is computed using
        all 3 different modes in the input code.
        """
        file_path = 'data_for_tests'

        file_mode0 = 'testgals_chisq_mode0.fit'
        file_mode1 = 'testgals_chisq_mode1.fit'
        file_mode2 = 'testgals_chisq_mode2.fit'

        parfile = 'test_dr8_pars.fit'

        # read in the parameters (note this is tested in test_redsequence.py)
        # we're only reading in a small redshift range for testing speed.
        zredstr = redmapper.read_redsequence('%s/%s' % (file_path, parfile), fine=True, zrange=[0.15, 0.25])

        # test mode 0: many galaxies, one redshift

        mode0data = fitsio.read('%s/%s' % (file_path, file_mode0),ext=1)

        # find the index in the zredstr
        zind = redmapper.redsequence_zindex(zredstr, mode0data['Z'][0])
        magind = redmapper.redsequence_refmagindex(zredstr, mode0data['REFMAG_INDEXED'])

        galcolor = mode0data['MODEL_MAG'][:,0:4] - mode0data['MODEL_MAG'][:,1:5]

        chisq, lkhd = redmapper.compute_chisq(zredstr['covmat'][:,:,zind], zredstr['c'][zind,:], zredstr['slope'][zind,:], zredstr['pivotmag'][zind], mode0data['REFMAG'], mode0data['MODEL_MAGERR'], galcolor, refmagerr=mode0data['REFMAG_ERR'], lupcorr=zredstr['lupcorr'][magind,zind,:], calc_chisq=True, calc_lkhd=True)
        testing.assert_almost_equal(chisq, mode0data['CHISQ'],decimal=3)
        testing.assert_almost_equal(lkhd, mode0data['LKHD'],decimal=3)

        # test mode 1: one galaxy, many redshifts

        mode1data = fitsio.read('%s/%s' % (file_path, file_mode1),ext=1)

        # find the index in the zredstr
        zind = redmapper.redsequence_zindex(zredstr, mode1data[0]['Z_INDEXED'])
        magind = redmapper.redsequence_refmagindex(zredstr, mode1data[0]['REFMAG_INDEXED'])

        galcolor = mode1data[0]['MODEL_MAG'][0:4] - mode1data[0]['MODEL_MAG'][1:5]

        chisq, lkhd = redmapper.compute_chisq(zredstr['covmat'][:,:,zind], zredstr['c'][zind,:], zredstr['slope'][zind,:], zredstr['pivotmag'][zind], mode1data['REFMAG'], mode1data[0]['MODEL_MAGERR'], galcolor, refmagerr=mode1data[0]['REFMAG_ERR'], lupcorr=zredstr['lupcorr'][magind,zind,:], calc_chisq=True, calc_lkhd=True)

        testing.assert_almost_equal(chisq, mode1data[0]['CHISQ'],decimal=3)
        testing.assert_almost_equal(lkhd, mode1data[0]['LKHD'],decimal=3)

        # test mode 2: many galaxies, many redshifts

        mode2data = fitsio.read('%s/%s' % (file_path, file_mode2),ext=1)

        # find the index in the zredstr
        zind = redmapper.redsequence_zindex(zredstr, mode2data['Z_INDEXED'])
        magind = redmapper.redsequence_refmagindex(zredstr, mode2data['REFMAG_INDEXED'])

        galcolor = mode2data['MODEL_MAG'][:,0:4] - mode2data['MODEL_MAG'][:,1:5]

        # there are just too many floating point rounding errors compared to the IDL
        # version.  So this is a really loose check.

        # will want to write python double-precision regression check as well.
        chisq, lkhd = redmapper.compute_chisq(zredstr['covmat'][:,:,zind], zredstr['c'][zind,:], zredstr['slope'][zind,:], zredstr['pivotmag'][zind], mode2data['REFMAG'], mode2data['MODEL_MAGERR'], galcolor, refmagerr=mode2data['REFMAG_ERR'], lupcorr=zredstr['lupcorr'][magind,zind,:], calc_chisq=True, calc_lkhd=True)

        testing.assert_almost_equal(chisq, mode2data['CHISQ'],decimal=1)
        testing.assert_almost_equal(lkhd, mode2data['LKHD'],decimal=2)


if __name__=='__main__':
    unittest.main()
