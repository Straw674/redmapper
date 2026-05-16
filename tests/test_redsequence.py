import unittest
import numpy.testing as testing
import numpy as np
import fitsio

import redmapper

class RedSequenceColorTestCase(unittest.TestCase):
    """
    Tests of redmapper.RedSequenceColorPar, including reading and interpolation.
    """

    def runTest(self):
        """
        Run tests of redmapper.RedSequenceColorPar.
        """
        file_name = 'test_dr8_pars.fit'
        file_path = 'data_for_tests'

        # test that we fail if we try a non-existent file
        self.assertRaises(IOError, redmapper.read_redsequence, 'nonexistent.fit')

        # test that we fail if we read a non-fits file
        self.assertRaises(IOError, redmapper.read_redsequence, '%s/testconfig.yaml' % (file_path))

        # test that we fail if we try a file without the right header info
        self.assertRaises(ValueError, redmapper.read_redsequence, '%s/test_bkg.fit' % (file_path))

        # read in the parameters
        zredstr = redmapper.read_redsequence('%s/%s' % (file_path, file_name))

        # make sure that nmag matches
        testing.assert_equal(zredstr['nmag'], 5)

        # check the z range...
        testing.assert_almost_equal([zredstr['z'][0], zredstr['z'][zredstr['z'].size-2]], np.array([0.01, 0.665]))

        # and the number of zs (+1 for overflow bin)
        testing.assert_equal(zredstr['z'].size, 132+1)

        # check lookup tables...
        testing.assert_equal(redmapper.redsequence_zindex(zredstr, [0.0, 0.2, 0.502, 0.7]), [0, 38, 98, 132])
        testing.assert_equal(redmapper.redsequence_refmagindex(zredstr, [11.0, 15.19, 19.195, 50.0]), [0, 319, 720, 930])
        testing.assert_equal(redmapper.redsequence_lumrefmagindex(zredstr, [11.0, 15.19, 19.195, 50.0]), [0, 319, 720, 1153])

        indices = np.array([20, 50, 100])

        # spot check of pivotmags ... IDL & python
        testing.assert_almost_equal(zredstr['pivotmag'][indices], np.array([17.199219, 19.013103, 19.877018]), decimal=5)

        # spot check of c
        testing.assert_almost_equal(zredstr['c'][indices, 0], np.array([1.877657,  1.762201,  2.060556]), decimal=5)
        testing.assert_almost_equal(zredstr['c'][indices, 1], np.array([0.963666,  1.419433,  1.620476]), decimal=5)
        testing.assert_almost_equal(zredstr['c'][indices, 2], np.array([0.411257,  0.524789,  0.873508]), decimal=5)
        testing.assert_almost_equal(zredstr['c'][indices, 3], np.array([0.327266,  0.333995,  0.448025]), decimal=5)

        # and slope ...
        testing.assert_almost_equal(zredstr['slope'][indices, 0], np.array([-0.029465, -0.105079,  0.646350]), decimal=5)

        # sigma...diagonals
        testing.assert_almost_equal(zredstr['sigma'][0, 0, indices], np.array([0.083769,  0.209662,  2.744580]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][1, 1, indices], np.array([0.042948,  0.078304,  0.070614]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][2, 2, indices], np.array([0.020754,  0.021819,  0.087495]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][3, 3, indices], np.array([0.021977,  0.021785,  0.018490]), decimal=5)

        # a couple of off-diagonal checks
        testing.assert_almost_equal(zredstr['sigma'][1, 2, indices], np.array([0.721869,  0.790119,  0.717613]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][2, 1, indices], np.array([0.721869,  0.790119,  0.717613]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][0, 3, indices], np.array([0.136167,  0.154741,  0.000111]), decimal=5)
        testing.assert_almost_equal(zredstr['sigma'][3, 0, indices], np.array([0.136167,  0.154741,  0.000111]), decimal=5)

        # covmat...diagonals
        testing.assert_almost_equal(zredstr['covmat'][0, 0, indices], np.array([0.007017,  0.043958,  7.532721]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][1, 1, indices], np.array([0.001845,  0.006132,  0.004986]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][2, 2, indices], np.array([0.000431,  0.000476,  0.007655]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][3, 3, indices], np.array([0.000483,  0.000475,  0.000342]), decimal=5)

        # and off-diagonal checks
        testing.assert_almost_equal(zredstr['covmat'][1, 2, indices], np.array([0.000643,  0.001350,  0.004434]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][2, 1, indices], np.array([0.000643,  0.001350,  0.004434]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][0, 3, indices], np.array([0.000251,  0.000707,  0.000006]), decimal=5)
        testing.assert_almost_equal(zredstr['covmat'][3, 0, indices], np.array([0.000251,  0.000707,  0.000006]), decimal=5)

        # lupcorr...here we want to test all colors...
        testing.assert_almost_equal(zredstr['lupcorr'][800, indices, 0], np.array([-0.060606, -0.127144, -0.510393]), decimal=3)
        testing.assert_almost_equal(zredstr['lupcorr'][800, indices, 1], np.array([-0.000689, -0.002615, -0.007729]), decimal=3)
        testing.assert_almost_equal(zredstr['lupcorr'][800, indices, 2], np.array([0.000025, -0.000057, -0.000425]), decimal=3)
        testing.assert_almost_equal(zredstr['lupcorr'][800, indices, 3], np.array([0.002966,  0.002853,  0.002245]), decimal=3)

        # corr stuff
        testing.assert_almost_equal(zredstr['corr'][indices], np.array([0.004373,  0.006569,  0.008507]), decimal=5)
        testing.assert_almost_equal(zredstr['corr_slope'][indices], np.array([0.000000,  0.000000,  0.000000]), decimal=5)
        testing.assert_almost_equal(zredstr['corr_r'][indices], np.array([0.710422,  0.500000,  0.500000]), decimal=5)

        # corr2 stuff
        testing.assert_almost_equal(zredstr['corr2'][indices], np.array([0.004148,  0.004810, -0.002500]), decimal=5)
        testing.assert_almost_equal(zredstr['corr2_slope'][indices], np.array([0.000000,  0.000000,  0.000000]), decimal=5)
        testing.assert_almost_equal(zredstr['corr2_r'][indices], np.array([0.664222,  0.527134,  0.500000]), decimal=5)

        # volume factor
        testing.assert_almost_equal(zredstr['volume_factor'][indices], np.array([0.758344,  0.820818,  0.947833]), decimal=5)

        # mstar
        testing.assert_almost_equal(zredstr['_mstar'][indices], np.array([16.461048, 18.476456, 20.232077]), decimal=5)

        # lumnorm
        testing.assert_almost_equal(zredstr['lumnorm'][400, indices], np.array([0.105102,  0.000006,  0.000000]), decimal=5)
        testing.assert_almost_equal(zredstr['lumnorm'][800, indices], np.array([2.958363,  1.152083,  0.163357]), decimal=5)

        # And the extrapolated parts

        extrap_indices = np.array([8, 119])
        notextrap_indices = np.array([9, 118])

        # pivotmag
        testing.assert_almost_equal(zredstr['pivotmag'][extrap_indices], zredstr['pivotmag'][notextrap_indices])

        # slope
        for j in range(zredstr['ncol']):
            testing.assert_almost_equal(zredstr['slope'][extrap_indices, j], zredstr['slope'][notextrap_indices, j])

        # Colors
        testing.assert_almost_equal(zredstr['c'][extrap_indices, 0], np.array([1.82362366, 1.72837418]))
        testing.assert_almost_equal(zredstr['c'][extrap_indices, 1], np.array([0.82371621, 1.58101897]))
        testing.assert_almost_equal(zredstr['c'][extrap_indices, 2], np.array([0.38392715, 1.0715787]))
        testing.assert_almost_equal(zredstr['c'][extrap_indices, 3], np.array([0.30677113, 0.49210047]))


if __name__=='__main__':
    unittest.main()

