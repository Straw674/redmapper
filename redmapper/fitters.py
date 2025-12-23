"""Classes for fitting red sequence and related parameters.

This file contains the classes used to fit the red sequence model, including
the median relations, mean relations, scatter, covariance, etc.
"""
import numpy as np
from scipy import special
import scipy.optimize
import esutil
import warnings

from .utilities import CubicSpline, interpol

class MedZFitter(object):
    """
    Class to fit a spline to the median value as a function of redshift.

    This can be the color or magnitude or any quantity.
    """
    def __init__(self, z_nodes, redshifts, values):
        """
        Instantiate a MedZFitter object

        Parameters
        ----------
        z_nodes: `np.array`
           Float array for redshift nodes
        redshifts: `np.array`
           Float array of input redshifts to fit
        values: `np.array`
           Float array of color values to fit
        """
        self._z_nodes = z_nodes.astype(np.float64)
        self._redshifts = redshifts.astype(np.float64)
        self._values = values.astype(np.float64)

    def fit(self, p0, min_val=-np.inf, max_val=np.inf):
        """
        Perform a spline fit to the median value as a function of redshift.

        Parameters
        ----------
        p0: `np.array`
           Initial fit parameters, with same number of elements as nodes

        Returns
        -------
        pars: `np.array`
           Spline node parameters
        """
        # add bounds if we need it...
        bounds = []
        for i in range(len(p0)):
            bounds.append([min_val, max_val])

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      jac=False,
                                      options={'maxfun': 2000,
                                               'maxiter': 2000,
                                               'maxcor': 20,
                                               'eps': 1e-5,
                                               'gtol': 1e-8},
                                      callback=None)
        pars = res.x

        return pars

    def __call__(self, pars):
        """
        Compute the median cost function for f(pars)

        Parameters
        ----------
        pars: `np.array`
           Fit parameters, with same number of elements as nodes

        Returns
        -------
        t: `float`
           Median cost
        """
        spl = CubicSpline(self._z_nodes, pars)
        m = spl(self._redshifts)

        absdev = np.abs(self._values - m)
        t = np.sum(absdev.astype(np.float64))

        return t

class RedSequenceFitter(object):
    """
    Class to fit a spline to the red sequence as a function of redshift.

    This class computes the mean value of the color, the red sequence slope,
    and the intrinsic scatter as a function of redshift.

    The node positions can be different for the mean color, the slope, and the
    intrinsic scatter.
    """
    def __init__(self, mean_nodes,
                 redshifts, colors, mag_errs, dmags=None, trunc=None,
                 slope_nodes=None, scatter_nodes=None, lupcorrs=None,
                 probs=None, bkgs=None, scatter_max=None, use_scatter_prior=False,
                 dmags_err_ratio=None):
        """
        Instantiate a RedSequenceFitter

        Parameters
        ----------
        mean_nodes: `np.array`
           Float array for mean color redshift nodes
        redshifts: `np.array`
           Float array of input redshifts for fit
        colors: `np.array`
           Float array of input colors to fit
        mag_errs: `np.array`
           Float array of input mag errors to fit [size, 2]
        dmags: `np.array`, optional
           Float array of delta-mag (refmag - pivot) (location on x axis for slope).
           Must have same length as colors if used.  Default is None (don't fit slope).
        trunc: `np.array`, optional
           Float array of delta-color used to truncate input colors.
           This outlier rejection is corrected for in the Gaussian likelihood.
           Must have same length as colors if used.  Default is None (no truncation).
        slope_nodes: `np.array`, optional
           Float array of red sequence slope node locations.
           Default is None (use mean_nodes for slope).
        scatter_nodes: `np.array`, optional
           Float array of red sequence scatter node locations.
           Default is None (use mean_nodes for scatter).
        lupcorrs: `np.array`, optional
           Float array of model corrections based on use of luptitudes.
           Must have same length as colors if used.
           Default is None (no luptitude corrections required).
        probs: `np.array`, optional
           Float array of membership probabilities.
           Must have same length as colors if used; must also supply "bkgs" if used.
           Default is None (assume all galaxies have p=1)
        bkgs: `np.array`, optional
           Float array of background likelihoods
           Must have same length as colors if used; must also supply "probs" if used.
           Default is None (assume all galaxies have bkg=0)
        scatter_max: `float`, optional
           Maximum intrinsic scatter allowed for any node.  Default is None (no max).
        use_scatter_prior: `bool`, optional
           Use Jeffry's prior on intrinsic scatter.  Default is False.
        dmags_err_ratio : `np.ndarray`, optional
            Delta-mag for error ratio computation.
        """

        self._use_scatter_prior = use_scatter_prior

        self._mean_nodes = np.atleast_1d(mean_nodes).astype(np.float64)
        if slope_nodes is None:
            self._slope_nodes = self._mean_nodes
        else:
            self._slope_nodes = np.atleast_1d(slope_nodes).astype(np.float64)
        if scatter_nodes is None:
            self._scatter_nodes = self._mean_nodes
        else:
            self._scatter_nodes = np.atleast_1d(scatter_nodes).astype(np.float64)

        self._redshifts = np.atleast_1d(redshifts).astype(np.float64)
        self._colors = np.atleast_1d(colors).astype(np.float64)
        self._mag_err2s = np.atleast_2d(mag_errs).astype(np.float64)**2.

        self._n_mean_nodes = self._mean_nodes.size
        self._n_slope_nodes = self._slope_nodes.size
        self._n_scatter_nodes = self._scatter_nodes.size

        if self._redshifts.size != self._colors.size:
            raise ValueError("Number of redshifts must be equal to colors")
        if self._redshifts.size != self._mag_err2s.shape[0]:
            raise ValueError("Number of redshifts must be equal to mag_errs.shape[1]")
        if self._mag_err2s.shape[1] != 2:
            raise ValueError("Mag_errs must by 2xNgals")

        if trunc is not None:
            self._trunc = np.atleast_1d(trunc).astype(np.float64)
            if self._redshifts.size != self._trunc.size:
                raise ValueError("Number of redshifts must be equal to truncs")
        else:
            self._trunc = None

        if dmags is not None:
            self._dmags = np.atleast_1d(dmags).astype(np.float64)
            if self._redshifts.size != self._dmags.size:
                raise ValueError("Number of redshifts must be equal to dmags")
            self._has_dmags = True
        else:
            self._dmags = np.zeros(self._redshifts.size).astype(np.float64)
            self._has_dmags = False

        if dmags_err_ratio is not None:
            self._dmags_err_ratio = np.atleast_1d(dmags_err_ratio).astype(np.float64)
            if self._redshifts.size != self._dmags_err_ratio.size:
                raise ValueError("Number of redshifts must be equal to dmags_err_ratio")
        else:
            self._dmags_err_ratio = np.zeros(self._redshifts.size)

        if lupcorrs is not None:
            self._lupcorrs = np.atleast_1d(lupcorrs).astype(np.float64)
            if self._redshifts.size != self._lupcorrs.size:
                raise ValueError("Number of redshifts must be equal to lupcorrs")
            self._has_lupcorrs = True
        else:
            self._lupcorrs = np.zeros(self._redshifts.size)
            self._has_lupcorrs = False

        if probs is not None:
            self._probs = np.atleast_1d(probs).astype(np.float64)
            if self._redshifts.size != self._probs.size:
                raise ValueError("Number of redshifts must be equal to probs")
            self._has_probs = True
        else:
            self._has_probs = False

        if bkgs is not None:
            self._bkgs = np.atleast_1d(bkgs).astype(np.float64)
            if self._redshifts.size != self._bkgs.size:
                raise ValueError("Number of redshifts must be equal to bkgs")
            self._has_bkgs = True
        else:
            self._has_bkgs = False

        if scatter_max is not None:
            self._scatter_max = np.atleast_1d(scatter_max).astype(np.float64)
            if self._scatter_max.size != self._n_scatter_nodes:
                raise ValueError("Number of scatter_max must be equal to scatter_nodes")
            self._has_scatter_max = True
        else:
            self._has_scatter_max = False

        if self._has_probs and not self._has_bkgs:
            raise ValueError("If you supply probs you must also supply bkgs")

    def fit(self, p0_mean, p0_slope, p0_scatter,
            fit_mean=False, fit_slope=False, fit_scatter=False,
            min_scatter=0.001, err_ratio_pars=None, fit_err_ratio_ind=[0, 1]):
        """
        Fit the red sequence mean, slope, and intrinsic scatter.

        Parameters
        ----------
        p0_mean: `np.array`
           Float array of initial fit parameters for mean relation.
           Must have same number of elements as mean_nodes
        p0_slope: `np.array`
           Float array of initial fit parameters for slope relation.
           Must have same number of elements as slope_nodes
        p0_scatter: `np.array`
           Float array of initial fit parameters for scatter relation.
           Must have same number of elements as scatter_nodes
        fit_mean: `bool`, optional
           Fit the mean relation? (else fix to p0_mean).  Default is False.
        fit_slope: `bool`, optional
           Fit the slope? (else fix to p0_slope).  Default is False.
        fit_scatter: `bool`, optional
           Fit the scatter? (else fix to p0_scatter).  Default is False.
        min_scatter: `float`, optional
           Minimum intrinsic scatter.  Default is 0.001.
        err_ratio_pars : `float`, optional
            Error ratio parameters.  Will be fit if fit_scatter=True and is
            not None.
        fit_err_ratio_ind : array-like, optional
            Magnitude indices to apply error ratio to fit.

        Returns
        -------
        pars_mean: `np.array`
           Mean parameters (if fit_mean is True)
        pars_slope: `np.array`
           Slope parameters (if fit_slope is True)
        pars_scatter: `np.array`
           Scatter parameters (if fit_scatter is True)
        """
        self._fit_mean = fit_mean
        self._fit_slope = fit_slope
        self._fit_scatter = fit_scatter
        self._min_scatter = min_scatter
        if err_ratio_pars is not None:
            self._has_err_ratios = True
            self._err_ratio_pars = err_ratio_pars
            self._fit_err_ratio_ind = fit_err_ratio_ind
        else:
            # Do not fit, use 1.0
            self._has_err_ratios = False
            self._err_ratio_pars = [1.0, 0.0]
            self._fit_err_ratio_ind = []

        if not self._has_dmags and self._fit_slope:
            raise ValueError("Can only do fit_slope if dmags were supplied")

        ctr = 0
        p0 = np.array([])
        bounds = []
        if self._fit_mean:
            self._mean_index = 0
            ctr += self._n_mean_nodes
            p0 = np.append(p0, p0_mean)
            for i in range(self._n_mean_nodes):
                bounds.append([-np.inf, np.inf])
        if self._fit_slope:
            self._slope_index = ctr
            ctr += self._n_slope_nodes
            p0 = np.append(p0, p0_slope)
            for i in range(self._n_slope_nodes):
                bounds.append([-np.inf, np.inf])
        if self._fit_scatter:
            self._scatter_index = ctr
            ctr += self._n_scatter_nodes
            p0 = np.append(p0, p0_scatter)
            for i in range(self._n_scatter_nodes):
                if self._has_scatter_max:
                    bounds.append([self._min_scatter, self._scatter_max[i]])
                else:
                    bounds.append([self._min_scatter, np.inf])
            if self._has_err_ratios:
                p0 = np.append(p0, self._err_ratio_pars[0])
                bounds.append([0.5, 50.0])
                p0 = np.append(p0, self._err_ratio_pars[1])
                bounds.append([-5.0, 5.0])

        if ctr == 0:
            raise ValueError("Must select at least one of fit_mean, fit_slope, fit_scatter")

        # Precompute...
        if not self._fit_mean:
            spl = CubicSpline(self._mean_nodes, p0_mean)
            self._gmean = spl(self._redshifts)
        if not self._fit_slope:
            spl = CubicSpline(self._slope_nodes, p0_slope)
            self._gslope = spl(self._redshifts)
        if not self._fit_scatter:
            spl = CubicSpline(self._scatter_nodes, p0_scatter)
            err_ratios = self._err_ratio_pars[0] + self._err_ratio_pars[1]*self._dmags_err_ratio
            if 0 in self._fit_err_ratio_ind:
                e2 = (err_ratios**2.)*self._mag_err2s[:, 0]
            else:
                e2 = self._mag_err2s[:, 0]
            if 1 in self._fit_err_ratio_ind:
                e2 += (err_ratios**2.)*self._mag_err2s[:, 1]
            else:
                e2 += self._mag_err2s[:, 1]
            self._gsig = np.sqrt(np.clip(spl(self._redshifts), self._min_scatter, None)**2. + e2)

        if not self._fit_scatter and self._trunc is not None:
            self._phi_bma = special.erf((self._trunc / self._gsig) / np.sqrt(2.))

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      jac=False,
                                      options={'maxfun': 2000,
                                               'maxiter': 2000,
                                               'maxcor': 20,
                                               'eps': 1e-5,
                                               'gtol': 1e-8},
                                      callback=None)
        pars = res.x

        retval = []
        if self._fit_mean:
            retval.append(pars[self._mean_index: self._mean_index + self._n_mean_nodes])
        if self._fit_slope:
            retval.append(pars[self._slope_index: self._slope_index + self._n_slope_nodes])
        if self._fit_scatter:
            if self._has_err_ratios:
                retval.append(pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes + 2])
            else:
                retval.append(pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes])

        return retval

    def __call__(self, pars):
        """
        Compute the red sequence log-likelihood (negative for minimization).

        Parameters
        ----------
        pars: `np.array`
           Concatenated array of all the fit parameters

        Returns
        -------
        t: `float`
           Total negative log-likelihood.
        """
        if self._fit_mean:
            # We are fitting the mean
            spl = CubicSpline(self._mean_nodes, pars[self._mean_index: self._mean_index + self._n_mean_nodes])
            gmean = spl(self._redshifts)
        else:
            gmean = self._gmean

        if self._fit_slope:
            # We are fitting the slope
            spl = CubicSpline(self._slope_nodes, pars[self._slope_index: self._slope_index + self._n_slope_nodes])
            gslope = spl(self._redshifts)
        else:
            gslope = self._gslope

        if self._fit_scatter:
            # We are fitting the scatter
            spl = CubicSpline(self._scatter_nodes, pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes])
            if self._has_err_ratios:
                # Always the last one
                err_ratio_pars = pars[-2: ]
            else:
                err_ratio_pars = [1.0, 0.0]
            err_ratios = err_ratio_pars[0] + err_ratio_pars[1]*self._dmags_err_ratio
            if 0 in self._fit_err_ratio_ind:
                e2 = (err_ratios**2.)*self._mag_err2s[:, 0]
            else:
                e2 = self._mag_err2s[:, 0]
            if 1 in self._fit_err_ratio_ind:
                e2 += (err_ratios**2.)*self._mag_err2s[:, 1]
            else:
                e2 += self._mag_err2s[:, 1]
            self._gsig = np.sqrt(np.clip(spl(self._redshifts), self._min_scatter, None)**2. + e2)

        if self._fit_scatter and self._trunc is not None:
            phi_bma = special.erf((self._trunc / self._gsig) / np.sqrt(2.))
        elif self._trunc is not None:
            phi_bma = self._phi_bma
        else:
            phi_bma = 1.0

        if self._has_dmags:
            model_color = gmean + gslope * self._dmags + self._lupcorrs
        else:
            model_color = gmean

        xi = (self._colors - model_color) / self._gsig

        phi = (1. / self._gsig) * (1. / np.sqrt(2. * np.pi)) * np.exp(-0.5 * xi**2.)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This might have a divide-by-zero, that's okay, we check later.
            gci = phi / phi_bma

        if self._has_probs:
            # Use probabilities and bkgs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                vals = np.log(self._probs * gci + (1.0 - self._probs) * self._bkgs)
        else:
            # No probabilities or bkgs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                vals = np.log(gci)

        bad, = np.where(~np.isfinite(vals))
        vals[bad] = -100.0

        if self._fit_scatter and self._use_scatter_prior:
            t = -(np.sum(vals) - np.sum(np.log(np.clip(pars[self._scatter_index: self._scatter_index + self._n_scatter_nodes], self._min_scatter, None))))
        else:
            t = -np.sum(vals)

        return t

class RedSequenceOffDiagonalFitter(object):
    """
    Class to fit a spline to a pair of off-diagonal elements of the
    red-sequence covariance matrix.

    This can only be run after the diagonal elements of the red-sequence model
    have been constrained.
    """
    def __init__(self, nodes, redshifts, d1, d2, s1, s2, mag_errs, j, k, probs, bkgs, covmat_prior, min_eigenvalue=0.0):
        """
        Instantiate a RedSequenceOffDiagonalFitter

        Parameters
        ----------
        nodes: `np.array`
           Float array of covariance matrix redshift nodes
        redshifts: `np.array`
           Float array of input redshifts for fit
        d1: `np.array`
           Float array of color residual (color - model_color) for color 1 (j)
           Must have same number of elements as redshifts.
        d2: `np.array`
           Float array of color residual (color - model_color) for color 2 (k)
           Must have same number of elements as redshifts.
        s1: `np.array`
           Float array of intrinsic scatters for color 1 (j)
           Must have same number of elements as redshifts.
        s2: `np.array`
           Float array of intrinsic scatters for color 2 (k)
           Must have same number of elements as redshifts.
        mag_errs: `np.array`
           2d float array of magnitude errors [n_redshifts, nmag]
        j: `int`
           Index for color 1
        k: `int`
           Index for color 2
        probs: `np.array`
           Float array of membership probabilities
           Must have same number of elements as redshifts.
        bkgs: `np.array`
           Float array of background likelihood for color 1, color 2.
           Must have same number of elements as redshifts.
        covmat_prior: `float`
           Prior on covariance matrix off-diagonal elements.
        min_eigenvalue: `float`, optional
           Minimum eigenvalue of covariance matrix.  Default is 0.0.
        """
        self._nodes = np.atleast_1d(nodes).astype(np.float64)
        self._redshifts = np.atleast_1d(redshifts).astype(np.float64)
        self._d1 = np.atleast_1d(d1).astype(np.float64)
        self._d2 = np.atleast_1d(d2).astype(np.float64)
        self._s1 = np.atleast_1d(s1).astype(np.float64)
        self._s2 = np.atleast_1d(s2).astype(np.float64)
        self._probs = np.atleast_1d(probs).astype(np.float64)
        self._bkgs = np.atleast_1d(bkgs).astype(np.float64)

        if self._redshifts.size != self._d1.size:
            raise ValueError("Number of redshifts must be equal to d1")
        if self._redshifts.size != self._d2.size:
            raise ValueError("Number of redshifts must be equal to d2")
        if self._redshifts.size != self._s1.size:
            raise ValueError("Number of redshifts must be equal to s1")
        if self._redshifts.size != self._s2.size:
            raise ValueError("Number of redshifts must be equal to s2")
        if self._redshifts.size != self._probs.size:
            raise ValueError("Number of redshifts must be equal to probs")
        if self._redshifts.size != self._bkgs.size:
            raise ValueError("Number of redshifts must be equal to bkgs")

        self._j = j
        self._k = k

        if len(mag_errs.shape) != 2:
            raise ValueError("mag_errs must be 2d")
        if mag_errs.shape[0] != self._redshifts.size:
            raise ValueError("Number of redshifts must be number of mag_errs")

        self._covmat_prior = covmat_prior
        self._min_eigenvalue = min_eigenvalue

        self._c_int = np.zeros((2, 2, self._redshifts.size))
        self._c_int[0, 0, :] = self._s1**2.
        self._c_int[1, 1, :] = self._s2**2.

        self._c_noise = np.zeros_like(self._c_int)
        self._c_noise[0, 0, :] = mag_errs[:, j]**2. + mag_errs[:, j + 1]**2.
        self._c_noise[1, 1, :] = mag_errs[:, k]**2. + mag_errs[:, k + 1]**2.
        if k == (j + 1):
            self._c_noise[0, 1, :] = -mag_errs[:, k]**2.
            self._c_noise[1, 0, :] = self._c_noise[0, 1, :]

    def fit(self, p0, full_covmats=None):
        """
        Perform a spline fit to a pair of elements of the covariance matrix.

        Parameters
        ----------
        p0: `np.array`
           Initial fit parameters, with same number of elements as nodes
        full_covmats: `np.array`, optional
           Full covariance matrix node values [nmag, nmag, nnode]
           Default is None.  If not None, then minimum eigenvalue is checked during fit.

        Returns
        -------
        pars: `np.array`
           Correlation values (r) for each node.
        """

        self._full_covmats = full_covmats

        bounds = []
        for i in range(self._nodes.size):
            bounds.append((-0.9, 0.9))

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      jac=False,
                                      options={'maxfun': 2000,
                                               'maxiter': 2000,
                                               'maxcor': 20,
                                               'eps': 1e-3,
                                               'gtol': 1e-8},
                                      callback=None)
        pars = res.x

        return pars

    def __call__(self, pars):
        """
        Compute the off-diagonal log-likelihood (negative for minimization)

        Parameters
        ----------
        pars: `np.array`
           Float array of the correlation (r) values

        Returns
        -------
        t: `float`
           Total negative log-likelihood
        """

        spl = CubicSpline(self._nodes, pars)
        r = np.clip(spl(self._redshifts), -0.9, 0.9)

        metrics = np.zeros((2, 2, self._redshifts.size))
        self._c_int[0, 1, :] = r * self._s1 * self._s2
        self._c_int[1, 0, :] = self._c_int[0, 1, :]

        if self._full_covmats is not None:
            self._full_covmats[self._j, self._k, :] = pars * np.sqrt(self._full_covmats[self._j, self._j, :]) * np.sqrt(self._full_covmats[self._k, self._k, :])
            self._full_covmats[self._k, self._j, :] = self._full_covmats[self._j, self._k, :]

        covmats = self._c_int + self._c_noise

        dets = covmats[0, 0, :] * covmats[1, 1, :] - covmats[0, 1, :] * covmats[1, 0, :]

        metrics[0, 0, :] = covmats[1, 1, :] / dets
        metrics[1, 1, :] = covmats[0, 0, :] / dets
        metrics[1, 0, :] = -covmats[0, 1, :] / dets
        metrics[0, 1, :] = -covmats[1, 0, :] / dets

        exponents = -0.5 * (metrics[0, 0, :] * self._d1 * self._d1 +
                            (metrics[0, 1, :] + metrics[1, 0, :]) * self._d1 * self._d2 +
                            metrics[1, 1, :] * self._d2 * self._d2)

        gci = (dets**(-0.5) / (2. * np.pi)) * np.exp(exponents)

        vals = np.log(self._probs * gci + (1. - self._probs ) * self._bkgs)

        bad, = np.where(~np.isfinite(vals))
        vals[bad] = -100

        t=-(np.sum(vals) - np.sum(0.5 * (pars / self._covmat_prior)**2.))

        if ~np.isfinite(t):
            t = 1e11
        else:
            wall = False

            # Check for negative eigenvalues
            if self._full_covmats is not None:
                for i in range(self._nodes.size):
                    a = self._full_covmats[:, :, i]
                    d = np.linalg.eigvalsh(a)
                    if (np.min(d) < self._min_eigenvalue):
                        wall = True
            if wall:
                t += 100000

        return t

class CorrectionFitter(object):
    """
    Class for fit a spline to the zred corrections as a function of redshift.

    This class computes the zred corrections, including slope (if desired) and
    error adjustment factor.
    """
    def __init__(self, mean_nodes, redshifts, dzs, dz_errs,
                 slope_nodes=None, r_nodes=None, bkg_nodes=None,
                 probs=None, dmags=None, ws=None):
        """
        Instantiate a CorrectionFitter

        Parameters
        ----------
        mean_nodes: `np.array`
           Float array for mean correction redshift nodes
        redshifts: `np.array`
           Float array of input redshifts for fit
        dzs: `np.array`
           Float array of input delta-z (zred - ztrue) for fit
        dz_errs: `np.array`
           Float array of error on delta-z
        slope_nodes: `np.array`, optional
           Float array for slope redshift nodes.
           Default is None (use mean_nodes for slope).
        r_nodes: `np.array`, optional
           Float array for error factor redshift nodes.
           Default is None (use slope_nodes for correction factor).
        bkg_nodes: `np.array`, optional
           Float array for background/outlier redshift nodes.
           Default is None (use slope_nodes for bkg.)
        probs: `np.array`, optional
           Float array for membership probabilities.
           Default is None (no probabilities used).
        dmags: `np.array`, optional
           Float array of delta-mag (refmag - pivot) (location on x axis for slope).
           Must have same length as redshifts if used.
           Default is None (don't fit slope).
        ws: `np.array`, optional
           Float array of likelihood weighting factors.
           Default is None (no weighting factors).
        """
        self._mean_nodes = np.atleast_1d(mean_nodes).astype(np.float64)
        # Note that the slope_nodes are the default for the r, bkg as well
        if slope_nodes is None:
            self._slope_nodes = self._mean_nodes
        else:
            self._slope_nodes = np.atleast_1d(slope_nodes).astype(np.float64)
        if r_nodes is None:
            self._r_nodes = self._slope_nodes
        else:
            self._r_nodes = np.atleast_1d(r_nodes).astype(np.float64)
        if bkg_nodes is None:
            self._bkg_nodes = self._slope_nodes
        else:
            self._bkg_nodes = np.atleast_1d(bkg_nodes).astype(np.float64)

        self._n_mean_nodes = self._mean_nodes.size
        self._n_slope_nodes = self._slope_nodes.size
        self._n_r_nodes = self._r_nodes.size
        self._n_bkg_nodes = self._bkg_nodes.size

        self._redshifts = np.atleast_1d(redshifts).astype(np.float64)
        self._dzs = np.atleast_1d(dzs).astype(np.float64)
        self._dz_errs = np.atleast_1d(dz_errs).astype(np.float64)

        if self._redshifts.size != self._dzs.size:
            raise ValueError("Number of redshifts must be equal to dzs")
        if self._redshifts.size != self._dz_errs.size:
            raise ValueError("Number of redshifts must be equal to dz_errs")

        if probs is not None:
            self._probs = np.atleast_1d(probs).astype(np.float64)
            if self._redshifts.size != self._probs.size:
                raise ValueError("Number of redshifts must be equal to probs")
        else:
            self._probs = np.ones_like(self._redshifts)

        if dmags is not None:
            self._dmags = np.atleast_1d(dmags).astype(np.float64)
            if self._redshifts.size != self._dmags.size:
                raise ValueError("Number of redshifts must be equal to dmags")
        else:
            self._dmags = np.zeros_like(self._redshifts)

        if ws is not None:
            self._ws = np.atleast_1d(ws).astype(np.float64)
            if self._redshifts.size != self._ws.size:
                raise ValueError("Number of redshifts must be equal to ws")
        else:
            self._ws = np.ones_like(self._redshifts)

    def fit(self, p0_mean, p0_slope, p0_r, p0_bkg, fit_mean=False, fit_slope=False, fit_r=False, fit_bkg=False):
        """
        Fit the zred correction factor.

        Parameters
        ----------
        p0_mean: `np.array`
           Float array of initial fit parameters for mean offset relation
           Must have same number of elements as mean_nodes
        p0_slope: `np.array`
           Float array of initial fit parameters for slope relation
           Must have same number of elements as slope_nodes
        p0_r: `np.array`
           Float array of initial fit parameters for error scaling relation
           Must have same number of elements as r_nodes
        p0_bkg: `np.array`
           Float array of initial fit parameters for bkg outlier relation
           Must have same number of elements as bkg_nodes
        fit_mean: `bool`, optional
           Fit the mean relation? (else fix to p0_mean).  Default is False.
        fit_slope: `bool`, optional
           Fit the slope relation? (else fix to p0_slope).  Default is False.
        fit_r: `bool`, optional
           Fit the r relation? (else fix to p0_r).  Default is False.
        fit_bkg: `bool`, optional
           Fit the bkg relation? (else fix to p0_bkg).  Default is False.

        Returns
        -------
        pars_mean: `np.array`
           Mean parameters (if fit_mean is True)
        pars_slope: `np.array`
           Slope parameters (if fit_slope is True)
        pars_r: `np.array`
           Error scaling parameters (if fit_r is True)
        pars_bkg: `np.array`
           Background parameters (if fit_bkg is True)
        """
        self._fit_mean = fit_mean
        self._fit_slope = fit_slope
        self._fit_r = fit_r
        self._fit_bkg = fit_bkg

        ctr = 0
        p0 = np.array([])
        bounds = []
        if self._fit_mean:
            self._mean_index = 0
            ctr += self._n_mean_nodes
            p0 = np.append(p0, p0_mean)
            for i in range(self._n_mean_nodes):
                bounds.append([-np.inf, np.inf])
        if self._fit_slope:
            self._slope_index = ctr
            ctr += self._n_slope_nodes
            p0 = np.append(p0, p0_slope)
            for i in range(self._n_slope_nodes):
                bounds.append([-np.inf, np.inf])
        if self._fit_r:
            self._r_index = ctr
            ctr += self._n_r_nodes
            p0 = np.append(p0, p0_r)
            for i in range(self._n_r_nodes):
                bounds.append([0.1, 2.0])
        if self._fit_bkg:
            self._bkg_index = ctr
            ctr += self._n_bkg_nodes
            p0 = np.append(p0, p0_bkg)
            for i in range(self._n_bkg_nodes):
                bounds.append([0.0, np.inf])

        if ctr == 0:
            raise ValueError("Must select at least one of fit_mean, fit_slope")

        # Precompute
        if not self._fit_mean:
            spl = CubicSpline(self._mean_nodes, p0_mean)
            self._gmean = spl(self._redshifts)
        if not self._fit_slope:
            spl = CubicSpline(self._slope_nodes, p0_slope)
            self._gslope = spl(self._redshifts)
        if not self._fit_r:
            spl = CubicSpline(self._r_nodes, p0_r)
            self._gr = spl(self._redshifts)
        if not self._fit_bkg:
            spl = CubicSpline(self._bkg_nodes, p0_bkg)
            self._gbkg = np.clip(spl(self._redshifts), 1e-10, None)
            self._gci1 = (1. / np.sqrt(2. * np.pi * self._gbkg)) * np.exp(-self._dzs**2. / (2. * self._gbkg))

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      jac=False,
                                      options={'maxfun': 5000,
                                               'maxiter': 5000,
                                               'maxcor': 20,
                                               'eps': 1e-5,
                                               'gtol': 1e-10},
                                      callback=None)
        pars = res.x

        retval = []
        if self._fit_mean:
            retval.append(pars[self._mean_index: self._mean_index + self._n_mean_nodes])
        if self._fit_slope:
            retval.append(pars[self._slope_index: self._slope_index + self._n_slope_nodes])
        if self._fit_r:
            retval.append(pars[self._r_index: self._r_index + self._n_r_nodes])
        if self._fit_bkg:
            retval.append(pars[self._bkg_index: self._bkg_index + self._n_bkg_nodes])

        return retval

    def __call__(self, pars):
        """
        Compute the correction log-likelihood (negative for minimization)

        Parameters
        ----------
        pars: `np.array`
           Float array of the consolidate parameters

        Returns
        -------
        t: `float`
           Total negative log-likelihood
        """

        if self._fit_mean:
            spl = CubicSpline(self._mean_nodes, pars[self._mean_index: self._mean_index + self._n_mean_nodes])
            gmean = spl(self._redshifts)
        else:
            gmean = self._gmean

        if self._fit_slope:
            spl = CubicSpline(self._slope_nodes, pars[self._slope_index: self._slope_index + self._n_slope_nodes])
            gslope = spl(self._redshifts)
        else:
            gslope = self._gslope

        if self._fit_r:
            spl = CubicSpline(self._r_nodes, pars[self._r_index: self._r_index + self._n_r_nodes])
            gr = spl(self._redshifts)
        else:
            gr = self._gr

        if self._fit_bkg:
            #if pars[self._bkg_index: self._bkg_index + self._n_bkg_nodes].min() < 0.0:
            #    return 1e11
            spl = CubicSpline(self._bkg_nodes, pars[self._bkg_index: self._bkg_index + self._n_bkg_nodes])
            gbkg = np.clip(spl(self._redshifts), 1e-10, None)
            gci1 = (1. / np.sqrt(2. * np.pi * gbkg)) * np.exp(-self._dzs**2. / (2. * gbkg))
        else:
            gbkg = self._gbkg
            gci1 = self._gci1

        var0 = (gr * self._dz_errs)**2.
        gci0 = (1. / np.sqrt(2. * np.pi * var0)) * np.exp(-(self._dzs - (gmean + gslope * self._dmags))**2. / (2. * var0))

        vals = self._ws * (self._probs * gci0 + (1. - self._probs) * gci1)

        bad, = np.where((~np.isfinite(vals)) | (vals <= 0.0))
        vals[bad] = 4e-44

        vals = np.log(vals)

        t = -np.sum(vals)

        return t


class EcgmmFitter(object):
    """
    Class to fit the error-corrected Gaussian Mixture Model (ECGMM), see Hao et
    al. (2009) (ApJ).

    This is a simplified version that will do two components.
    """
    def __init__(self, y, y_err):
        """
        Instantiate a EcgmmFitter.

        Parameters
        ----------
        y: `np.array`
           Float array of y values to decompose.
        y_err: `np.array`
           Float array of y error values to decompose.
        """
        self._y = y.astype(np.float64)
        self._y_err2 = y_err.astype(np.float64)**2.

    def fit(self, wt0, mu, sigma, bounds=None, offset=0.0):
        """
        Perform an ECGMM fit.

        Parameters
        ----------
        wt0: `float` or `np.array` (1 elements)
           Initial guess for 0th component weight
           1st component weight is 1.0 - wt0
        mu: `np.array` (2 elements)
           Initial guess for component means
        sigma: `np.array` (2 elements)
           Initial guess for component widths
        bounds: `list`, optional
           bounds[0][:] is a two element range of wt0
           bounds[1][:] is a two element range of mu[0]
           bounds[2][:] is a two element range of mu[1]
           bounds[3][:] is a two element range of sigma[0]
           bounds[4][:] is a two element range of sigma[1]
           Default is bounds is None (no bounds).
        offset: `float`, optional
           Arbitrary offset to ensure that neither mean is ~0 which
           for some reason confuses the fitter.  Default is 0 (no offset).

        Returns
        -------
        wt: `np.array`
           Two element array with [wt0, wt1]
        mu: `np.array`
           Two element array with mean values
        sigma: `np.array`
           Two element array with sigmas
        """

        p0 = np.concatenate([np.atleast_1d(wt0),
                             np.atleast_1d(mu) + offset,
                             np.atleast_1d(sigma)])

        self._y += offset

        if bounds is None:
            _bounds = [(1e-5, 1.0), # wt0
                       (-1.0 + offset, 1.0 + offset), # mu0
                       (-1.0 + offset, 1.0 + offset), # mu1
                       (1e-2, 0.5), # sigma0
                       (1e-2, 0.5)] # sigma1
        else:
            _bounds = bounds

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=_bounds,
                                      jac=False,
                                      options={'maxfun': 2000,
                                               'maxiter': 2000,
                                               'maxcor': 20,
                                               'eps': 1e-5,
                                               'gtol': 1e-8},
                                      callback=None)
        pars = res.x

        wt = np.array([pars[0], 1.0 - pars[0]])
        mu = pars[1:3] - offset
        sigma = pars[3:5]

        # sort so that the red is the *second* one
        st = np.argsort(mu)

        self._y -= offset

        return wt[st], mu[st], sigma[st]

    def __call__(self, pars):
        """
        Compute the ECGMM log-likelihood (negative for minimization).

        Parameters
        ----------
        pars: `np.array`
           Float array of the combined parameters
           pars: [wt0, mu0, mu1, sigma0, sigma1]

        Returns
        -------
        t: `float`
          Total negative log-likelihood
        """
        wt0 = pars[0]
        mu0 = pars[1]
        mu1 = pars[2]
        sigma0 = pars[3]
        sigma1 = pars[4]

        wt1 = 1.0 - wt0
        """
        if (wt0 < self._bounds[0][0] or wt0 > self._bounds[0][1] or
            mu0 < self._bounds[1][0] or mu0 > self._bounds[1][1] or
            mu1 < self._bounds[2][0] or mu1 > self._bounds[2][1] or
            sigma0 < self._bounds[3][0] or sigma0 > self._bounds[3][1] or
            sigma1 < self._bounds[4][0] or sigma1 > self._bounds[4][1]):
            return np.inf
            """
        g = ((wt0 / np.sqrt(2. * np.pi * (sigma0**2. + self._y_err2)) * np.exp(-(self._y - mu0)**2. / (2. * (sigma0**2. + self._y_err2)))) +
             (wt1 / np.sqrt(2. * np.pi * (sigma1**2. + self._y_err2)) * np.exp(-(self._y - mu1)**2. / (2. * (sigma1**2. + self._y_err2)))))

        t = np.sum(np.log(g))

        return -t


class ErrorBinFitter(object):
    """Class for fitting error ratios in bins, using median statistics.

    Parameters
    ----------
    delta_col : `np.ndarray`
        Array of delta colors
    delta_mag : `np.ndarray`
        Array of delta mag (mag - pivot)
    err_0 : `np.ndarray`
        Raw or scaled error for first magnitude in color
    err_1 : np.ndarray`
        Raw or scaled error for second magnitude in color
    sigint2 : `np.ndarray`
        Intrinsic scatter squared.
    binsize : `float`
        Bin size.
    ntrial : `int`
        Number of trials for bootstrap errors.
    """
    def __init__(self, delta_col, delta_mag, err_0, err_1, sigint2, binsize=0.5, ntrial=100):
        self.delta_col = delta_col
        self.delta_mag = delta_mag
        self.err_0 = err_0
        self.err_1 = err_1
        self.sigint2 = sigint2

        h_full = esutil.stat.histogram(delta_mag, binsize=binsize, more=True)

        self.nbin = h_full['nbin']
        self.binmag = h_full['center']
        self.rev = h_full['rev']

        delta_err = np.sqrt(self.sigint2 +
                            self.err_0**2. +
                            self.err_1**2.)
        pulls = self.delta_col/delta_err

        self.mad_err = np.zeros(self.nbin)
        for i in range(self.nbin):
            bin_mads = np.zeros(ntrial)
            i1a = self.rev[self.rev[i]: self.rev[i + 1]]
            for j in range(ntrial):
                r = np.random.choice(i1a, size=i1a.size, replace=True)

                med = np.median(pulls[r])
                bin_mads[j] = 1.4826*np.median(np.abs(pulls[r] - med))

            mad = np.median(bin_mads)
            self.mad_err[i] = 1.4826*np.median(np.abs(bin_mads - mad))

        # We require a positive error estimate (if too few in a bin)
        self.use_bins = np.where(self.mad_err > 0.0)

    def fit(self, p0, scale_indices=[0]):
        """Perform a fit to the error scaling as a function of magnitude.

        Parameters
        ----------
        p0 : array-like
            Initial fit parameters, intercept and slope.
        scale_indices : `list`
            List of mag indices to scale.  Can be [0], [1], or [0, 1]

        Returns
        -------
        pars : `np.ndarray`
            Fit parameters, intercept and slope.
        """
        self._scale_indices = scale_indices

        bounds = [[0.5, 50.0],
                  [-5.0, 5.0]]

        res = scipy.optimize.minimize(self,
                                      p0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      jac=False,
                                      options={'maxfun': 2000,
                                               'maxiter': 2000,
                                               'maxcor': 20,
                                               'eps': 1e-5,
                                               'gtol': 1e-8},
                                      callback=None)
        return res.x

    def __call__(self, pars):
        """Compute the chi2 for the pars.

        Parameters
        ----------
        pars : `list`
            Parameters, intercept and slope.

        Returns
        -------
        chi2 : `float`
        """
        mad = np.zeros(self.nbin)

        err_ratio = pars[0] + pars[1]*self.delta_mag

        if 0 in self._scale_indices:
            scaled_err_0 = err_ratio*self.err_0
        else:
            scaled_err_0 = self.err_0
        if 1 in self._scale_indices:
            scaled_err_1 = err_ratio*self.err_1
        else:
            scaled_err_1 = self.err_1

        delta_err = np.sqrt(self.sigint2 +
                            scaled_err_0**2. +
                            scaled_err_1**2.)
        pulls = self.delta_col/delta_err

        for i in range(self.nbin):
            i1a = self.rev[self.rev[i]: self.rev[i + 1]]
            med = np.median(pulls[i1a])
            mad[i] = 1.4826*np.median(np.abs(pulls[i1a] - med))

        chi2 = np.sum((mad[self.use_bins] - 1.0)**2./self.mad_err[self.use_bins]**2., dtype=np.float64)
        return chi2
