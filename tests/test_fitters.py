import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random

from redmapper.fitters import fit_ecgmm
from redmapper.fitters import fit_med_z
from redmapper.fitters import fit_red_sequence
from redmapper.fitters import fit_correction
from redmapper.fitters import fit_error_bin
from redmapper.utilities import make_nodes, cubic_spline_compute_y2, cubic_spline_interpolate

class FitterTestCase(unittest.TestCase):
    """
    Tests for various fitters in redmapper.fitters, including fit_ecgmm,
    fit_med_z, fit_red_sequence, and
    fit_correction.
    """
    def test_ecgmm(self):
        """
        Run tests of redmapper.fitters.fit_ecgmm
        """
        random.seed(seed=1000)

        # Test Ecgmm Fitter

        file_path = 'data_for_tests'
        ecgmm_testdata_filename = 'test_ecgmm.fit'

        ecgmmdata, hdr = fitsio.read(file_path + '/' + ecgmm_testdata_filename, ext=1, header=True)

        wt, mu, sigma = fit_ecgmm(ecgmmdata['DELTA'], ecgmmdata['GALCOLOR_ERR'], [0.2], [-0.5, 0.0], [0.2, 0.05], offset=0.5)

        testing.assert_almost_equal(wt, [0.56733591, 0.43266409], 5)
        testing.assert_almost_equal(mu, [-0.31850688, -0.11686182], 5)
        testing.assert_almost_equal(sigma, [0.15283559, 0.04079095], 5)

    def test_make_nodes(self):
        """
        Run tests of redmapper.utilities.make_nodes()
        """
        # Test make_nodes
        nodes = make_nodes([0.1,0.65], 0.05)
        testing.assert_almost_equal(nodes, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.5, 0.65])
        nodes = make_nodes([0.1, 0.65], 0.1, maxnode=0.4)
        testing.assert_almost_equal(nodes, [0.1, 0.2, 0.3, 0.4, 0.65])

    def test_red_sequence_fitting(self):
        """
        Run tests of fit_med_z and fit_red_sequence.
        """
        # Test Median z Fitter
        from redmapper.fitters import fit_med_z, fit_red_sequence
        file_path = 'data_for_tests'

        nodes = make_nodes([0.1, 0.2], 0.05)
        fit_testdata_filename = 'test_rsfit.fit'

        fitdata = fitsio.read(file_path + '/' + fit_testdata_filename, ext=1)

        p0 = np.array([1.0, 1.1, 1.2])
        medpars = fit_med_z(nodes, fitdata['Z'], fitdata['GALCOLOR'], p0)

        testing.assert_almost_equal(medpars, [0.94603754, 1.09285003, 1.27060679], 4)

        # Test median scatter fit
        y2 = cubic_spline_compute_y2(nodes, medpars)
        m = cubic_spline_interpolate(fitdata['Z'], nodes, medpars, y2)

        p0 = np.array([0.05, 0.05, 0.05])
        medscatpars = fit_med_z(nodes, fitdata['Z'], np.abs(fitdata['GALCOLOR'] - m), p0)

        testing.assert_almost_equal(medscatpars, [0.02407928, 0.02745547, 0.03131481], 4)

        # And the red sequence fitter part
        mag_err = np.zeros((fitdata.size, 2))
        mag_err[:, 0] = fitdata['GALCOLOR_ERR']
        p0_mean = medpars
        p0_slope = np.zeros(nodes.size)
        p0_scatter = medscatpars * 1.4826
        meanpars_list = fit_red_sequence(nodes, fitdata['Z'], fitdata['GALCOLOR'], mag_err,
                                         p0_mean, p0_slope, p0_scatter,
                                         fit_mean=True, use_scatter_prior=False)
        meanpars = meanpars_list[0]

        testing.assert_almost_equal(meanpars, [0.94867445, 1.09166076, 1.26180069], 4)

        # Second, just the scatter
        p0_mean = meanpars
        scatpars_list = fit_red_sequence(nodes, fitdata['Z'], fitdata['GALCOLOR'], mag_err,
                                         p0_mean, p0_slope, p0_scatter,
                                         fit_scatter=True, use_scatter_prior=False)
        scatpars = scatpars_list[0]

        testing.assert_almost_equal(scatpars, [0.03197, 0.035, 0.0406], 4)

        # Third, just the slope ... need to write this!
        p0_scatter = scatpars
        # If we don't have the dmags then this will raise a ValueError
        self.assertRaises(ValueError, fit_red_sequence, nodes, fitdata['Z'], fitdata['GALCOLOR'], mag_err,
                          p0_mean, p0_slope, p0_scatter, fit_slope=True)

        dmags = fitdata['REFMAG'] - np.median(fitdata['REFMAG'])
        slopepars_list = fit_red_sequence(nodes, fitdata['Z'], fitdata['GALCOLOR'], mag_err,
                                          p0_mean, p0_slope, p0_scatter,
                                          fit_slope=True, dmags=dmags)
        slopepars = slopepars_list[0]

        # These values seem reasonable, but this is just a toy test
        testing.assert_almost_equal(slopepars, [-0.00830984,-0.01538641, -0.02372566], 4)

        # Fourth, combined all 3
        # this test is a lot slower than the others, but still reasonable.
        p0_slope = slopepars
        meanpars2, slopepars2, scatpars2 = fit_red_sequence(nodes, fitdata['Z'], fitdata['GALCOLOR'], mag_err,
                                                            p0_mean, p0_slope, p0_scatter,
                                                            fit_mean=True, fit_slope=True, fit_scatter=True,
                                                            dmags=dmags)

        testing.assert_almost_equal(meanpars2, [0.94397328, 1.09101342, 1.26557143], 4)
        testing.assert_almost_equal(slopepars2, [-0.01229, -0.0158, -0.02707], 4)
        testing.assert_almost_equal(scatpars2, [0.03115847, 0.03430587, 0.03987611], 4)

        # Fifth, fit_mean with truncation
        p0_mean = meanpars2
        p0_slope = slopepars2
        p0_scatter = scatpars2

        y2 = cubic_spline_compute_y2(nodes, medpars)
        m = cubic_spline_interpolate(fitdata['Z'], nodes, medpars, y2)
        y2_scat = cubic_spline_compute_y2(nodes, medscatpars)
        mscat = cubic_spline_interpolate(fitdata['Z'], nodes, medscatpars, y2_scat) * 1.4826

        nsig = 1.5
        ok, = np.where(np.abs(fitdata['GALCOLOR'] - m) < (nsig * mscat))
        trunc = nsig * mscat[ok]
        meanpars3_list = fit_red_sequence(nodes, fitdata['Z'][ok],
                                          fitdata['GALCOLOR'][ok], mag_err[ok, :],
                                          p0_mean, p0_slope, p0_scatter,
                                          fit_mean=True,
                                          dmags=fitdata['REFMAG'][ok] - np.median(fitdata['REFMAG'][ok]),
                                          trunc=trunc)
        meanpars3 = meanpars3_list[0]

        testing.assert_almost_equal(meanpars3, [0.93789514, 1.09257947, 1.27767928], 4)

        # Finally, refit with scatter prior...
        meanpars3, slopepars3, scatpars3 = fit_red_sequence(nodes, fitdata['Z'][ok],
                                                            fitdata['GALCOLOR'][ok], mag_err[ok, :],
                                                            p0_mean, p0_slope, p0_scatter,
                                                            fit_mean=True, fit_slope=True, fit_scatter=True,
                                                            dmags=fitdata['REFMAG'][ok] - np.median(fitdata['REFMAG'][ok]),
                                                            trunc=trunc, use_scatter_prior=True)

        testing.assert_almost_equal(meanpars3, [0.93834492, 1.09276382, 1.27604319], 4)
        testing.assert_almost_equal(slopepars3, [-0.01054824, -0.01296713, -0.01605626], 4)
        testing.assert_almost_equal(scatpars3, [0.03453208, 0.04054536, 0.03832689], 4)

    def test_zred_correction_fitter(self):
        """
        Run tests of redmapper.fitters.fit_correction
        """

        # Test the zred correction fitter.
        # No tests right now for the slope fitting, because we don't
        # use it in the IDL code.
        file_path = 'data_for_tests'

        corr_testdata_filename = 'test_zredcorr_values.fit'
        fitdata = fitsio.read(file_path + '/' + corr_testdata_filename, ext=1)

        p0_mean = np.zeros(fitdata['NODES'][0].size)
        p0_slope = np.zeros(fitdata['SNODES'][0].size)
        p0_r = np.ones(fitdata['SNODES'][0].size)
        p0_bkg = np.zeros(fitdata['SNODES'][0].size) + 0.01

        # Common kwargs
        kwargs = {
            'slope_nodes': fitdata['SNODES'][0],
            'probs': fitdata['PI'][0],
            'dmags': fitdata['DMAG'][0],
            'ws': fitdata['W'][0]
        }

        pars_list = fit_correction(fitdata['NODES'][0], fitdata['Z'][0], fitdata['DZ'][0], fitdata['DZ_ERR'][0],
                                   p0_mean, p0_slope, p0_r, p0_bkg, fit_mean=True, **kwargs)
        pars_mean = pars_list[0]

        testing.assert_almost_equal(pars_mean, np.array([0.00483505, 0.00159392, -0.00018447]), 5)

        p0_mean = pars_mean
        pars_list = fit_correction(fitdata['NODES'][0], fitdata['Z'][0], fitdata['DZ'][0], fitdata['DZ_ERR'][0],
                                   p0_mean, p0_slope, p0_r, p0_bkg, fit_r=True, **kwargs)
        pars_r = pars_list[0]

        testing.assert_almost_equal(pars_r, np.array([0.79166451, 0.35962414]), 5)

        p0_r = pars_r
        pars_list = fit_correction(fitdata['NODES'][0], fitdata['Z'][0], fitdata['DZ'][0], fitdata['DZ_ERR'][0],
                                   p0_mean, p0_slope, p0_r, p0_bkg, fit_bkg=True, **kwargs)
        pars_bkg = pars_list[0]

        testing.assert_almost_equal(pars_bkg, np.array([0., 0.00043553]), 5)
        p0_bkg = pars_bkg
        pars_mean, pars_r, pars_bkg = fit_correction(fitdata['NODES'][0], fitdata['Z'][0], fitdata['DZ'][0], fitdata['DZ_ERR'][0],
                                                     p0_mean, p0_slope, p0_r, p0_bkg, fit_mean=True, fit_r=True, fit_bkg=True, **kwargs)

        testing.assert_almost_equal(pars_mean, np.array([0.00518834, 0.00178504, 0.00117311]), 5)
        testing.assert_almost_equal(pars_r, np.array([0.81519327, 0.33098604]), 5)
        testing.assert_almost_equal(pars_bkg, np.array([0., 0.00043576]), 5)

    def test_error_bin_fitter(self):
        """Test the error ratio fitter."""
        # Create a fake set of data with a ~2x error ratio, and 0.05 mag sig_int.
        np.random.seed(12345)

        ngal = 10000

        dmag = np.random.uniform(low=-6.0, high=2.0, size=ngal)
        sigint = np.zeros(ngal) + 0.05
        delta_col_true = np.random.normal(0.0, scale=sigint, size=ngal)
        err_0_true = np.zeros(ngal) + 0.02
        err_1_true = np.zeros(ngal) + 0.03

        delta_col = delta_col_true + np.random.normal(0.0, scale=err_0_true, size=ngal) + np.random.normal(0.0, scale=err_1_true, size=ngal)

        err_0_scale = 2.0 - 0.5*dmag
        err_1_scale = 2.0 - 0.5*dmag
        err_0_obs = err_0_true/err_0_scale
        err_1_obs = err_1_true/err_1_scale

        pars = fit_error_bin(delta_col, dmag, err_0_obs, err_1_obs, sigint**2., [1.0, 0.0], scale_indices=[0, 1])

        # These are approximately the input 2.0/-0.5 values.
        self.assertTrue(np.abs(pars[0] - 2.0) < 0.05)
        self.assertTrue(np.abs(pars[1] - (-0.5)) < 0.05)


if __name__=='__main__':
    unittest.main()
