"""Classes for fitting red sequence and related parameters.

This file contains the classes used to fit the red sequence model, including
the median relations, mean relations, scatter, covariance, etc.
"""
import numpy as np
from scipy import special
import scipy.optimize
import esutil
import warnings

from .utilities import cubic_spline_compute_y2, cubic_spline_interpolate, interpol

def med_z_cost(pars, z_nodes, redshifts, values):
    """
    Compute the median cost function for f(pars)

    Parameters
    ----------
    pars: `np.array`
       Fit parameters, with same number of elements as nodes
    z_nodes: `np.array`
       Float array for redshift nodes
    redshifts: `np.array`
       Float array of input redshifts to fit
    values: `np.array`
       Float array of color values to fit

    Returns
    -------
    t: `float`
       Median cost
    """
    y2 = cubic_spline_compute_y2(z_nodes, pars)
    m = cubic_spline_interpolate(redshifts, z_nodes, pars, y2)

    absdev = np.abs(values - m)
    t = np.sum(absdev.astype(np.float64))

    return t

def fit_med_z(z_nodes, redshifts, values, p0, min_val=-np.inf, max_val=np.inf):
    """
    Perform a spline fit to the median value as a function of redshift.

    Parameters
    ----------
    z_nodes: `np.array`
       Float array for redshift nodes
    redshifts: `np.array`
       Float array of input redshifts to fit
    values: `np.array`
       Float array of color values to fit
    p0: `np.array`
       Initial fit parameters, with same number of elements as nodes
    min_val: `float`, optional
       Minimum value for fit parameters. Default is -inf.
    max_val: `float`, optional
       Maximum value for fit parameters. Default is inf.

    Returns
    -------
    pars: `np.array`
       Spline node parameters
    """
    z_nodes = np.atleast_1d(z_nodes).astype(np.float64)
    redshifts = np.atleast_1d(redshifts).astype(np.float64)
    values = np.atleast_1d(values).astype(np.float64)

    bounds = [(min_val, max_val) for _ in range(len(p0))]

    res = scipy.optimize.minimize(med_z_cost,
                                  p0,
                                  args=(z_nodes, redshifts, values),
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

def red_sequence_cost(pars, mean_nodes, slope_nodes, scatter_nodes,
                       redshifts, colors, mag_err2s, dmags, trunc, dmags_err_ratio,
                       lupcorrs, probs, bkgs,
                       fit_mean, fit_slope, fit_scatter,
                       n_mean_nodes, n_slope_nodes, n_scatter_nodes,
                       mean_index, slope_index, scatter_index,
                       gmean_fixed, gslope_fixed, gsig_fixed, phi_bma_fixed,
                       has_dmags, has_lupcorrs, has_probs, has_bkgs, has_err_ratios,
                       fit_err_ratio_ind, min_scatter, use_scatter_prior):
    """
    Compute the red sequence log-likelihood (negative for minimization).

    Parameters
    ----------
    pars: `np.array`
       Concatenated array of all the fit parameters
    ... (omitting documentation for brevity as it's internal)
    """
    if fit_mean:
        pars_mean = pars[mean_index: mean_index + n_mean_nodes]
        y2 = cubic_spline_compute_y2(mean_nodes, pars_mean)
        gmean = cubic_spline_interpolate(redshifts, mean_nodes, pars_mean, y2)
    else:
        gmean = gmean_fixed

    if fit_slope:
        pars_slope = pars[slope_index: slope_index + n_slope_nodes]
        y2 = cubic_spline_compute_y2(slope_nodes, pars_slope)
        gslope = cubic_spline_interpolate(redshifts, slope_nodes, pars_slope, y2)
    else:
        gslope = gslope_fixed

    if fit_scatter:
        pars_scatter = pars[scatter_index: scatter_index + n_scatter_nodes]
        y2 = cubic_spline_compute_y2(scatter_nodes, pars_scatter)
        if has_err_ratios:
            # Always the last one
            err_ratio_pars = pars[-2: ]
        else:
            err_ratio_pars = [1.0, 0.0]
        err_ratios = err_ratio_pars[0] + err_ratio_pars[1] * dmags_err_ratio
        if 0 in fit_err_ratio_ind:
            e2 = (err_ratios**2.) * mag_err2s[:, 0]
        else:
            e2 = mag_err2s[:, 0]
        if 1 in fit_err_ratio_ind:
            e2 += (err_ratios**2.) * mag_err2s[:, 1]
        else:
            e2 += mag_err2s[:, 1]
        gsig = np.sqrt(np.clip(cubic_spline_interpolate(redshifts, scatter_nodes, pars_scatter, y2), min_scatter, None)**2. + e2)
    else:
        gsig = gsig_fixed

    if fit_scatter and trunc is not None:
        phi_bma = special.erf((trunc / gsig) / np.sqrt(2.))
    elif trunc is not None:
        phi_bma = phi_bma_fixed
    else:
        phi_bma = 1.0

    if has_dmags:
        model_color = gmean + gslope * dmags + lupcorrs
    else:
        model_color = gmean

    xi = (colors - model_color) / gsig
    phi = (1. / gsig) * (1. / np.sqrt(2. * np.pi)) * np.exp(-0.5 * xi**2.)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This might have a divide-by-zero, that's okay, we check later.
        gci = phi / phi_bma

    if has_probs:
        # Use probabilities and bkgs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            vals = np.log(probs * gci + (1.0 - probs) * bkgs)
    else:
        # No probabilities or bkgs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            vals = np.log(gci)

    bad, = np.where(~np.isfinite(vals))
    vals[bad] = -100.0

    if fit_scatter and use_scatter_prior:
        t = -(np.sum(vals) - np.sum(np.log(np.clip(pars[scatter_index: scatter_index + n_scatter_nodes], min_scatter, None))))
    else:
        t = -np.sum(vals)

    return t

def fit_red_sequence(mean_nodes, redshifts, colors, mag_errs, p0_mean, p0_slope, p0_scatter,
                     fit_mean=False, fit_slope=False, fit_scatter=False,
                     dmags=None, trunc=None, slope_nodes=None, scatter_nodes=None,
                     lupcorrs=None, probs=None, bkgs=None, scatter_max=None,
                     use_scatter_prior=False, min_scatter=0.001,
                     err_ratio_pars=None, fit_err_ratio_ind=[0, 1],
                     dmags_err_ratio=None):
    """
    Fit the red sequence mean, slope, and intrinsic scatter.

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
    min_scatter: `float`, optional
       Minimum intrinsic scatter.  Default is 0.001.
    err_ratio_pars : `float`, optional
        Error ratio parameters.  Will be fit if fit_scatter=True and is
        not None.
    fit_err_ratio_ind : array-like, optional
        Magnitude indices to apply error ratio to fit.
    dmags_err_ratio : `np.ndarray`, optional
        Delta-mag for error ratio computation.

    Returns
    -------
    pars_list: `list`
       List of fit parameters (mean, slope, scatter) as requested.
    """
    mean_nodes = np.atleast_1d(mean_nodes).astype(np.float64)
    if slope_nodes is None:
        slope_nodes = mean_nodes
    else:
        slope_nodes = np.atleast_1d(slope_nodes).astype(np.float64)
    if scatter_nodes is None:
        scatter_nodes = mean_nodes
    else:
        scatter_nodes = np.atleast_1d(scatter_nodes).astype(np.float64)

    redshifts = np.atleast_1d(redshifts).astype(np.float64)
    colors = np.atleast_1d(colors).astype(np.float64)
    mag_err2s = np.atleast_2d(mag_errs).astype(np.float64)**2.

    n_mean_nodes = mean_nodes.size
    n_slope_nodes = slope_nodes.size
    n_scatter_nodes = scatter_nodes.size

    if redshifts.size != colors.size:
        raise ValueError("Number of redshifts must be equal to colors")
    if redshifts.size != mag_err2s.shape[0]:
        raise ValueError("Number of redshifts must be equal to mag_errs.shape[1]")
    if mag_err2s.shape[1] != 2:
        raise ValueError("Mag_errs must by 2xNgals")

    if trunc is not None:
        trunc = np.atleast_1d(trunc).astype(np.float64)
        if redshifts.size != trunc.size:
            raise ValueError("Number of redshifts must be equal to truncs")

    if dmags is not None:
        dmags = np.atleast_1d(dmags).astype(np.float64)
        if redshifts.size != dmags.size:
            raise ValueError("Number of redshifts must be equal to dmags")
        has_dmags = True
    else:
        dmags = np.zeros(redshifts.size).astype(np.float64)
        has_dmags = False

    if dmags_err_ratio is not None:
        dmags_err_ratio = np.atleast_1d(dmags_err_ratio).astype(np.float64)
        if redshifts.size != dmags_err_ratio.size:
            raise ValueError("Number of redshifts must be equal to dmags_err_ratio")
    else:
        dmags_err_ratio = np.zeros(redshifts.size)

    if lupcorrs is not None:
        lupcorrs = np.atleast_1d(lupcorrs).astype(np.float64)
        if redshifts.size != lupcorrs.size:
            raise ValueError("Number of redshifts must be equal to lupcorrs")
        has_lupcorrs = True
    else:
        lupcorrs = np.zeros(redshifts.size)
        has_lupcorrs = False

    if probs is not None:
        probs = np.atleast_1d(probs).astype(np.float64)
        if redshifts.size != probs.size:
            raise ValueError("Number of redshifts must be equal to probs")
        has_probs = True
    else:
        has_probs = False

    if bkgs is not None:
        bkgs = np.atleast_1d(bkgs).astype(np.float64)
        if redshifts.size != bkgs.size:
            raise ValueError("Number of redshifts must be equal to bkgs")
        has_bkgs = True
    else:
        has_bkgs = False

    if has_probs and not has_bkgs:
        raise ValueError("If you supply probs you must also supply bkgs")

    if not has_dmags and fit_slope:
        raise ValueError("Can only do fit_slope if dmags were supplied")

    if err_ratio_pars is not None:
        has_err_ratios = True
        err_ratio_pars = err_ratio_pars
        fit_err_ratio_ind = fit_err_ratio_ind
    else:
        # Do not fit, use 1.0
        has_err_ratios = False
        err_ratio_pars = [1.0, 0.0]
        fit_err_ratio_ind = []

    ctr = 0
    p0 = np.array([])
    bounds = []
    mean_index = -1
    slope_index = -1
    scatter_index = -1

    if fit_mean:
        mean_index = 0
        ctr += n_mean_nodes
        p0 = np.append(p0, p0_mean)
        for i in range(n_mean_nodes):
            bounds.append([-np.inf, np.inf])
    if fit_slope:
        slope_index = ctr
        ctr += n_slope_nodes
        p0 = np.append(p0, p0_slope)
        for i in range(n_slope_nodes):
            bounds.append([-np.inf, np.inf])
    if fit_scatter:
        scatter_index = ctr
        ctr += n_scatter_nodes
        p0 = np.append(p0, p0_scatter)
        if scatter_max is not None:
            scatter_max = np.atleast_1d(scatter_max).astype(np.float64)
            if scatter_max.size != n_scatter_nodes:
                raise ValueError("Number of scatter_max must be equal to scatter_nodes")
            has_scatter_max = True
        else:
            has_scatter_max = False

        for i in range(n_scatter_nodes):
            if has_scatter_max:
                bounds.append([min_scatter, scatter_max[i]])
            else:
                bounds.append([min_scatter, np.inf])
        if has_err_ratios:
            p0 = np.append(p0, err_ratio_pars[0])
            bounds.append([0.5, 50.0])
            p0 = np.append(p0, err_ratio_pars[1])
            bounds.append([-5.0, 5.0])

    if ctr == 0:
        raise ValueError("Must select at least one of fit_mean, fit_slope, fit_scatter")

    # Precompute...
    gmean_fixed = None
    gslope_fixed = None
    gsig_fixed = None
    phi_bma_fixed = None

    if not fit_mean:
        y2 = cubic_spline_compute_y2(mean_nodes, p0_mean)
        gmean_fixed = cubic_spline_interpolate(redshifts, mean_nodes, p0_mean, y2)
    if not fit_slope:
        y2 = cubic_spline_compute_y2(slope_nodes, p0_slope)
        gslope_fixed = cubic_spline_interpolate(redshifts, slope_nodes, p0_slope, y2)
    if not fit_scatter:
        y2 = cubic_spline_compute_y2(scatter_nodes, p0_scatter)
        err_ratios = err_ratio_pars[0] + err_ratio_pars[1]*dmags_err_ratio
        if 0 in fit_err_ratio_ind:
            e2 = (err_ratios**2.)*mag_err2s[:, 0]
        else:
            e2 = mag_err2s[:, 0]
        if 1 in fit_err_ratio_ind:
            e2 += (err_ratios**2.)*mag_err2s[:, 1]
        else:
            e2 += mag_err2s[:, 1]
        gsig_fixed = np.sqrt(np.clip(cubic_spline_interpolate(redshifts, scatter_nodes, p0_scatter, y2), min_scatter, None)**2. + e2)

    if not fit_scatter and trunc is not None:
        phi_bma_fixed = special.erf((trunc / gsig_fixed) / np.sqrt(2.))

    res = scipy.optimize.minimize(red_sequence_cost,
                                  p0,
                                  args=(mean_nodes, slope_nodes, scatter_nodes,
                                        redshifts, colors, mag_err2s, dmags, trunc, dmags_err_ratio,
                                        lupcorrs, probs, bkgs,
                                        fit_mean, fit_slope, fit_scatter,
                                        n_mean_nodes, n_slope_nodes, n_scatter_nodes,
                                        mean_index, slope_index, scatter_index,
                                        gmean_fixed, gslope_fixed, gsig_fixed, phi_bma_fixed,
                                        has_dmags, has_lupcorrs, has_probs, has_bkgs, has_err_ratios,
                                        fit_err_ratio_ind, min_scatter, use_scatter_prior),
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
    if fit_mean:
        retval.append(pars[mean_index: mean_index + n_mean_nodes])
    if fit_slope:
        retval.append(pars[slope_index: slope_index + n_slope_nodes])
    if fit_scatter:
        if has_err_ratios:
            retval.append(pars[scatter_index: scatter_index + n_scatter_nodes + 2])
        else:
            retval.append(pars[scatter_index: scatter_index + n_scatter_nodes])

    return retval

def red_sequence_off_diagonal_cost(pars, nodes, redshifts, d1, d2, s1, s2,
                                   c_int_diag, c_noise, probs, bkgs,
                                   covmat_prior, full_covmats, j, k, min_eigenvalue):
    """
    Compute the off-diagonal log-likelihood (negative for minimization)

    Parameters
    ----------
    pars: `np.array`
       Float array of the correlation (r) values
    ...
    """
    y2 = cubic_spline_compute_y2(nodes, pars)
    r = np.clip(cubic_spline_interpolate(redshifts, nodes, pars, y2), -0.9, 0.9)

    c_int = np.zeros((2, 2, redshifts.size))
    c_int[0, 0, :] = c_int_diag[0]
    c_int[1, 1, :] = c_int_diag[1]
    c_int[0, 1, :] = r * s1 * s2
    c_int[1, 0, :] = c_int[0, 1, :]

    if full_covmats is not None:
        full_covmats[j, k, :] = pars * np.sqrt(full_covmats[j, j, :]) * np.sqrt(full_covmats[k, k, :])
        full_covmats[k, j, :] = full_covmats[j, k, :]

    covmats = c_int + c_noise

    dets = covmats[0, 0, :] * covmats[1, 1, :] - covmats[0, 1, :] * covmats[1, 0, :]

    # We need metrics to compute exponents
    m00 = covmats[1, 1, :] / dets
    m11 = covmats[0, 0, :] / dets
    m10 = -covmats[0, 1, :] / dets
    m01 = -covmats[1, 0, :] / dets

    exponents = -0.5 * (m00 * d1 * d1 + (m01 + m10) * d1 * d2 + m11 * d2 * d2)

    gci = (dets**(-0.5) / (2. * np.pi)) * np.exp(exponents)

    vals = np.log(probs * gci + (1. - probs) * bkgs)

    bad, = np.where(~np.isfinite(vals))
    vals[bad] = -100

    t = -(np.sum(vals) - np.sum(0.5 * (pars / covmat_prior)**2.))

    if ~np.isfinite(t):
        t = 1e11
    else:
        wall = False

        # Check for negative eigenvalues
        if full_covmats is not None:
            for i in range(nodes.size):
                a = full_covmats[:, :, i]
                d = np.linalg.eigvalsh(a)
                if (np.min(d) < min_eigenvalue):
                    wall = True
        if wall:
            t += 100000

    return t

def fit_red_sequence_off_diagonal(nodes, redshifts, d1, d2, s1, s2, mag_errs, j, k, probs, bkgs, covmat_prior, p0, min_eigenvalue=0.0, full_covmats=None):
    """
    Perform a spline fit to a pair of elements of the covariance matrix.

    Parameters
    ----------
    nodes: `np.array`
       Float array of covariance matrix redshift nodes
    redshifts: `np.array`
       Float array of input redshifts for fit
    d1: `np.array`
       Float array of color residual (color - model_color) for color 1 (j)
    d2: `np.array`
       Float array of color residual (color - model_color) for color 2 (k)
    s1: `np.array`
       Float array of intrinsic scatters for color 1 (j)
    s2: `np.array`
       Float array of intrinsic scatters for color 2 (k)
    mag_errs: `np.array`
       2d float array of magnitude errors [n_redshifts, nmag]
    j: `int`
       Index for color 1
    k: `int`
       Index for color 2
    probs: `np.array`
       Float array of membership probabilities
    bkgs: `np.array`
       Float array of background likelihood for color 1, color 2.
    covmat_prior: `float`
       Prior on covariance matrix off-diagonal elements.
    p0: `np.array`
       Initial fit parameters, with same number of elements as nodes
    min_eigenvalue: `float`, optional
       Minimum eigenvalue of covariance matrix.  Default is 0.0.
    full_covmats: `np.array`, optional
       Full covariance matrix node values [nmag, nmag, nnode]

    Returns
    -------
    pars: `np.array`
       Correlation values (r) for each node.
    """
    nodes = np.atleast_1d(nodes).astype(np.float64)
    redshifts = np.atleast_1d(redshifts).astype(np.float64)
    d1 = np.atleast_1d(d1).astype(np.float64)
    d2 = np.atleast_1d(d2).astype(np.float64)
    s1 = np.atleast_1d(s1).astype(np.float64)
    s2 = np.atleast_1d(s2).astype(np.float64)
    probs = np.atleast_1d(probs).astype(np.float64)
    bkgs = np.atleast_1d(bkgs).astype(np.float64)

    if redshifts.size != d1.size:
        raise ValueError("Number of redshifts must be equal to d1")
    if redshifts.size != d2.size:
        raise ValueError("Number of redshifts must be equal to d2")
    if redshifts.size != s1.size:
        raise ValueError("Number of redshifts must be equal to s1")
    if redshifts.size != s2.size:
        raise ValueError("Number of redshifts must be equal to s2")
    if redshifts.size != probs.size:
        raise ValueError("Number of redshifts must be equal to probs")
    if redshifts.size != bkgs.size:
        raise ValueError("Number of redshifts must be equal to bkgs")

    if len(mag_errs.shape) != 2:
        raise ValueError("mag_errs must be 2d")
    if mag_errs.shape[0] != redshifts.size:
        raise ValueError("Number of redshifts must be number of mag_errs")

    c_int_diag = (s1**2., s2**2.)

    c_noise = np.zeros((2, 2, redshifts.size))
    c_noise[0, 0, :] = mag_errs[:, j]**2. + mag_errs[:, j + 1]**2.
    c_noise[1, 1, :] = mag_errs[:, k]**2. + mag_errs[:, k + 1]**2.
    if k == (j + 1):
        c_noise[0, 1, :] = -mag_errs[:, k]**2.
        c_noise[1, 0, :] = c_noise[0, 1, :]

    bounds = [(-0.9, 0.9) for _ in range(nodes.size)]

    res = scipy.optimize.minimize(red_sequence_off_diagonal_cost,
                                  p0,
                                  args=(nodes, redshifts, d1, d2, s1, s2,
                                        c_int_diag, c_noise, probs, bkgs,
                                        covmat_prior, full_covmats, j, k, min_eigenvalue),
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

def correction_cost(pars, mean_nodes, slope_nodes, r_nodes, bkg_nodes,
                    redshifts, dzs, dz_errs, probs, dmags, ws,
                    fit_mean, fit_slope, fit_r, fit_bkg,
                    n_mean_nodes, n_slope_nodes, n_r_nodes, n_bkg_nodes,
                    mean_index, slope_index, r_index, bkg_index,
                    gmean_fixed, gslope_fixed, gr_fixed, gbkg_fixed, gci1_fixed):
    """
    Compute the correction log-likelihood (negative for minimization)

    Parameters
    ----------
    pars: `np.array`
       Float array of the consolidate parameters
    ... (omitting documentation for brevity)
    """
    if fit_mean:
        pars_mean = pars[mean_index: mean_index + n_mean_nodes]
        y2 = cubic_spline_compute_y2(mean_nodes, pars_mean)
        gmean = cubic_spline_interpolate(redshifts, mean_nodes, pars_mean, y2)
    else:
        gmean = gmean_fixed

    if fit_slope:
        pars_slope = pars[slope_index: slope_index + n_slope_nodes]
        y2 = cubic_spline_compute_y2(slope_nodes, pars_slope)
        gslope = cubic_spline_interpolate(redshifts, slope_nodes, pars_slope, y2)
    else:
        gslope = gslope_fixed

    if fit_r:
        pars_r = pars[r_index: r_index + n_r_nodes]
        y2 = cubic_spline_compute_y2(r_nodes, pars_r)
        gr = cubic_spline_interpolate(redshifts, r_nodes, pars_r, y2)
    else:
        gr = gr_fixed

    if fit_bkg:
        pars_bkg = pars[bkg_index: bkg_index + n_bkg_nodes]
        y2 = cubic_spline_compute_y2(bkg_nodes, pars_bkg)
        gbkg = np.clip(cubic_spline_interpolate(redshifts, bkg_nodes, pars_bkg, y2), 1e-10, None)
        gci1 = (1. / np.sqrt(2. * np.pi * gbkg)) * np.exp(-dzs**2. / (2. * gbkg))
    else:
        gci1 = gci1_fixed

    var0 = (gr * dz_errs)**2.
    gci0 = (1. / np.sqrt(2. * np.pi * var0)) * np.exp(-(dzs - (gmean + gslope * dmags))**2. / (2. * var0))

    vals = ws * (probs * gci0 + (1. - probs) * gci1)

    bad, = np.where((~np.isfinite(vals)) | (vals <= 0.0))
    vals[bad] = 4e-44

    vals = np.log(vals)

    t = -np.sum(vals)

    return t

def fit_correction(mean_nodes, redshifts, dzs, dz_errs,
                   p0_mean, p0_slope, p0_r, p0_bkg,
                   slope_nodes=None, r_nodes=None, bkg_nodes=None,
                   probs=None, dmags=None, ws=None,
                   fit_mean=False, fit_slope=False, fit_r=False, fit_bkg=False):
    """
    Fit a spline to the zred corrections as a function of redshift.

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
    p0_mean: `np.array`
       Initial fit parameters for mean offset relation
    p0_slope: `np.array`
       Initial fit parameters for slope relation
    p0_r: `np.array`
       Initial fit parameters for error scaling relation
    p0_bkg: `np.array`
       Initial fit parameters for bkg outlier relation
    slope_nodes: `np.array`, optional
       Float array for slope redshift nodes.
       Default is None (use mean_nodes).
    r_nodes: `np.array`, optional
       Float array for error factor redshift nodes.
       Default is None (use slope_nodes).
    bkg_nodes: `np.array`, optional
       Float array for background/outlier redshift nodes.
       Default is None (use slope_nodes).
    probs: `np.array`, optional
       Float array for membership probabilities.
       Default is None (all 1.0).
    dmags: `np.array`, optional
       Float array of delta-mag.
       Default is None (all 0.0).
    ws: `np.array`, optional
       Float array of likelihood weighting factors.
       Default is None (all 1.0).
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
    pars_list: `list`
       List of fit parameters as requested.
    """
    mean_nodes = np.atleast_1d(mean_nodes).astype(np.float64)
    if slope_nodes is None:
        slope_nodes = mean_nodes
    else:
        slope_nodes = np.atleast_1d(slope_nodes).astype(np.float64)
    if r_nodes is None:
        r_nodes = slope_nodes
    else:
        r_nodes = np.atleast_1d(r_nodes).astype(np.float64)
    if bkg_nodes is None:
        bkg_nodes = slope_nodes
    else:
        bkg_nodes = np.atleast_1d(bkg_nodes).astype(np.float64)

    n_mean_nodes = mean_nodes.size
    n_slope_nodes = slope_nodes.size
    n_r_nodes = r_nodes.size
    n_bkg_nodes = bkg_nodes.size

    redshifts = np.atleast_1d(redshifts).astype(np.float64)
    dzs = np.atleast_1d(dzs).astype(np.float64)
    dz_errs = np.atleast_1d(dz_errs).astype(np.float64)

    if redshifts.size != dzs.size:
        raise ValueError("Number of redshifts must be equal to dzs")
    if redshifts.size != dz_errs.size:
        raise ValueError("Number of redshifts must be equal to dz_errs")

    if probs is not None:
        probs = np.atleast_1d(probs).astype(np.float64)
        if redshifts.size != probs.size:
            raise ValueError("Number of redshifts must be equal to probs")
    else:
        probs = np.ones_like(redshifts)

    if dmags is not None:
        dmags = np.atleast_1d(dmags).astype(np.float64)
        if redshifts.size != dmags.size:
            raise ValueError("Number of redshifts must be equal to dmags")
    else:
        dmags = np.zeros_like(redshifts)

    if ws is not None:
        ws = np.atleast_1d(ws).astype(np.float64)
        if redshifts.size != ws.size:
            raise ValueError("Number of redshifts must be equal to ws")
    else:
        ws = np.ones_like(redshifts)

    ctr = 0
    p0 = np.array([])
    bounds = []
    mean_index = -1
    slope_index = -1
    r_index = -1
    bkg_index = -1

    if fit_mean:
        mean_index = 0
        ctr += n_mean_nodes
        p0 = np.append(p0, p0_mean)
        for i in range(n_mean_nodes):
            bounds.append([-np.inf, np.inf])
    if fit_slope:
        slope_index = ctr
        ctr += n_slope_nodes
        p0 = np.append(p0, p0_slope)
        for i in range(n_slope_nodes):
            bounds.append([-np.inf, np.inf])
    if fit_r:
        r_index = ctr
        ctr += n_r_nodes
        p0 = np.append(p0, p0_r)
        for i in range(n_r_nodes):
            bounds.append([0.1, 2.0])
    if fit_bkg:
        bkg_index = ctr
        ctr += n_bkg_nodes
        p0 = np.append(p0, p0_bkg)
        for i in range(n_bkg_nodes):
            bounds.append([0.0, np.inf])

    if ctr == 0:
        raise ValueError("Must select at least one of fit_mean, fit_slope")

    # Precompute
    gmean_fixed = None
    gslope_fixed = None
    gr_fixed = None
    gbkg_fixed = None
    gci1_fixed = None

    if not fit_mean:
        y2 = cubic_spline_compute_y2(mean_nodes, p0_mean)
        gmean_fixed = cubic_spline_interpolate(redshifts, mean_nodes, p0_mean, y2)
    if not fit_slope:
        y2 = cubic_spline_compute_y2(slope_nodes, p0_slope)
        gslope_fixed = cubic_spline_interpolate(redshifts, slope_nodes, p0_slope, y2)
    if not fit_r:
        y2 = cubic_spline_compute_y2(r_nodes, p0_r)
        gr_fixed = cubic_spline_interpolate(redshifts, r_nodes, p0_r, y2)
    if not fit_bkg:
        y2 = cubic_spline_compute_y2(bkg_nodes, p0_bkg)
        gbkg_fixed = np.clip(cubic_spline_interpolate(redshifts, bkg_nodes, p0_bkg, y2), 1e-10, None)
        gci1_fixed = (1. / np.sqrt(2. * np.pi * gbkg_fixed)) * np.exp(-dzs**2. / (2. * gbkg_fixed))

    res = scipy.optimize.minimize(correction_cost,
                                  p0,
                                  args=(mean_nodes, slope_nodes, r_nodes, bkg_nodes,
                                        redshifts, dzs, dz_errs, probs, dmags, ws,
                                        fit_mean, fit_slope, fit_r, fit_bkg,
                                        n_mean_nodes, n_slope_nodes, n_r_nodes, n_bkg_nodes,
                                        mean_index, slope_index, r_index, bkg_index,
                                        gmean_fixed, gslope_fixed, gr_fixed, gbkg_fixed, gci1_fixed),
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
    if fit_mean:
        retval.append(pars[mean_index: mean_index + n_mean_nodes])
    if fit_slope:
        retval.append(pars[slope_index: slope_index + n_slope_nodes])
    if fit_r:
        retval.append(pars[r_index: r_index + n_r_nodes])
    if fit_bkg:
        retval.append(pars[bkg_index: bkg_index + n_bkg_nodes])

    return retval


def ecgmm_cost(pars, y, y_err2):
    """
    Compute the ECGMM log-likelihood (negative for minimization).

    Parameters
    ----------
    pars: `np.array`
       Float array of the combined parameters
       pars: [wt0, mu0, mu1, sigma0, sigma1]
    y: `np.array`
       Float array of y values
    y_err2: `np.array`
       Float array of y error values squared

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

    g = ((wt0 / np.sqrt(2. * np.pi * (sigma0**2. + y_err2)) * np.exp(-(y - mu0)**2. / (2. * (sigma0**2. + y_err2)))) +
         (wt1 / np.sqrt(2. * np.pi * (sigma1**2. + y_err2)) * np.exp(-(y - mu1)**2. / (2. * (sigma1**2. + y_err2)))))

    t = np.sum(np.log(g))

    return -t

def fit_ecgmm(y, y_err, wt0, mu, sigma, bounds=None, offset=0.0):
    """
    Perform an ECGMM fit.

    Parameters
    ----------
    y: `np.array`
       Float array of y values to decompose.
    y_err: `np.array`
       Float array of y error values to decompose.
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
    y = y.astype(np.float64) + offset
    y_err2 = y_err.astype(np.float64)**2.

    p0 = np.concatenate([np.atleast_1d(wt0),
                         np.atleast_1d(mu) + offset,
                         np.atleast_1d(sigma)])

    if bounds is None:
        _bounds = [(1e-5, 1.0), # wt0
                   (-1.0 + offset, 1.0 + offset), # mu0
                   (-1.0 + offset, 1.0 + offset), # mu1
                   (1e-2, 0.5), # sigma0
                   (1e-2, 0.5)] # sigma1
    else:
        _bounds = bounds

    res = scipy.optimize.minimize(ecgmm_cost,
                                  p0,
                                  args=(y, y_err2),
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

    return wt[st], mu[st], sigma[st]



def error_bin_cost(pars, delta_mag, delta_col, err_0, err_1, sigint2,
                   nbin, rev, use_bins, mad_err, scale_indices):
    """
    Compute the chi2 for the error scaling parameters.

    Parameters
    ----------
    pars : `list`
        Parameters, intercept and slope.
    delta_mag : `np.ndarray`
        Array of delta mag
    delta_col : `np.ndarray`
        Array of delta color
    err_0 : `np.ndarray`
        First magnitude error
    err_1 : `np.ndarray`
        Second magnitude error
    sigint2 : `np.ndarray`
        Intrinsic scatter squared
    nbin : `int`
        Number of bins
    rev : `np.ndarray`
        Reverse indices from histogram
    use_bins : `np.ndarray`
        Indices of bins to use
    mad_err : `np.ndarray`
        Error on the MAD in each bin
    scale_indices : `list`
        List of magnitude indices to scale.

    Returns
    -------
    chi2 : `float`
    """
    mad = np.zeros(nbin)

    err_ratio = pars[0] + pars[1]*delta_mag

    if 0 in scale_indices:
        scaled_err_0 = err_ratio*err_0
    else:
        scaled_err_0 = err_0
    if 1 in scale_indices:
        scaled_err_1 = err_ratio*err_1
    else:
        scaled_err_1 = err_1

    delta_err = np.sqrt(sigint2 +
                        scaled_err_0**2. +
                        scaled_err_1**2.)
    pulls = delta_col/delta_err

    for i in range(nbin):
        i1a = rev[rev[i]: rev[i + 1]]
        med = np.median(pulls[i1a])
        mad[i] = 1.4826*np.median(np.abs(pulls[i1a] - med))

    chi2 = np.sum((mad[use_bins] - 1.0)**2./mad_err[use_bins]**2., dtype=np.float64)
    return chi2

def fit_error_bin(delta_col, delta_mag, err_0, err_1, sigint2, p0,
                  binsize=0.5, ntrial=100, scale_indices=[0]):
    """
    Perform a fit to the error scaling as a function of magnitude.

    Parameters
    ----------
    delta_col : `np.ndarray`
        Array of delta colors
    delta_mag : `np.ndarray`
        Array of delta mag (mag - pivot)
    err_0 : `np.ndarray`
        Raw or scaled error for first magnitude in color
    err_1 : `np.ndarray`
        Raw or scaled error for second magnitude in color
    sigint2 : `np.ndarray`
        Intrinsic scatter squared.
    p0 : array-like
        Initial fit parameters, intercept and slope.
    binsize : `float`, optional
        Bin size. Default is 0.5.
    ntrial : `int`, optional
        Number of trials for bootstrap errors. Default is 100.
    scale_indices : `list`, optional
        List of mag indices to scale.  Can be [0], [1], or [0, 1].
        Default is [0].

    Returns
    -------
    pars : `np.ndarray`
        Fit parameters, intercept and slope.
    """

    h_full = esutil.stat.histogram(delta_mag, binsize=binsize, more=True)

    nbin = h_full['nbin']
    binmag = h_full['center']
    rev = h_full['rev']

    delta_err = np.sqrt(sigint2 +
                        err_0**2. +
                        err_1**2.)
    pulls = delta_col/delta_err

    mad_err = np.zeros(nbin)
    for i in range(nbin):
        bin_mads = np.zeros(ntrial)
        i1a = rev[rev[i]: rev[i + 1]]
        for j in range(ntrial):
            r = np.random.choice(i1a, size=i1a.size, replace=True)

            med = np.median(pulls[r])
            bin_mads[j] = 1.4826*np.median(np.abs(pulls[r] - med))

        mad = np.median(bin_mads)
        mad_err[i] = 1.4826*np.median(np.abs(bin_mads - mad))

    # We require a positive error estimate (if too few in a bin)
    use_bins, = np.where(mad_err > 0.0)

    bounds = [[0.5, 50.0],
              [-5.0, 5.0]]

    res = scipy.optimize.minimize(error_bin_cost,
                                  p0,
                                  args=(delta_mag, delta_col, err_0, err_1, sigint2,
                                        nbin, rev, use_bins, mad_err, scale_indices),
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

