"""Functions to calibrate the z_lambda afterburner
"""
import os
import numpy as np
import fitsio
import time
import scipy.optimize
import copy

from ..configuration import Configuration
from ..galaxy import GalaxyCatalog
from ..cluster import ClusterCatalog
from ..zlambda import read_zlambda_correction, apply_zlambda_correction
from ..utilities import make_nodes, cubic_spline_compute_y2, cubic_spline_interpolate, interpol
from ..fitters import fit_med_z
from ..catalog import Entry
from ..logger import logger

def zlambda_cost(pars, fit_delta, fit_slope, fit_scatter, 
                 delta_index, n_nodes, slope_index, n_slope_nodes, scatter_index,
                 nodes, slope_nodes, redshifts, dzs, redshift_err2s, loglambdas, min_scatter,
                 gdelta_pre, gslope_pre, gscatter_pre):
    """
    Calculate the negative log-likelihood cost function for a set of parameters.

    Parameters
    ----------
    pars: `list`
       Parameters for fit, including delta, slope, scatter concatenated.
    fit_delta: `bool`
    fit_slope: `bool`
    fit_scatter: `bool`
    delta_index: `int`
    n_nodes: `int`
    slope_index: `int`
    n_slope_nodes: `int`
    scatter_index: `int`
    nodes: `np.array`
    slope_nodes: `np.array`
    redshifts: `np.array`
    dzs: `np.array`
    redshift_err2s: `np.array`
    loglambdas: `np.array`
    min_scatter: `float`
    gdelta_pre: `np.array` or None
    gslope_pre: `np.array` or None
    gscatter_pre: `np.array` or None

    Returns
    -------
    t: `float`
       Total cost function of negative log-likelihood to minimize.
    """
    if fit_delta:
        y_delta = pars[delta_index: delta_index + n_nodes]
        y2 = cubic_spline_compute_y2(nodes, y_delta)
        gdelta = cubic_spline_interpolate(redshifts, nodes, y_delta, y2)
    else:
        gdelta = gdelta_pre

    if fit_slope:
        y_slope = pars[slope_index: slope_index + n_slope_nodes]
        y2 = cubic_spline_compute_y2(slope_nodes, y_slope)
        gslope = cubic_spline_interpolate(redshifts, slope_nodes, y_slope, y2)
    else:
        gslope = gslope_pre

    if fit_scatter:
        y_scatter = pars[scatter_index: scatter_index + n_slope_nodes]
        y2 = cubic_spline_compute_y2(slope_nodes, y_scatter)
        gscatter = np.clip(cubic_spline_interpolate(redshifts, slope_nodes, y_scatter, y2), min_scatter, None)
    else:
        gscatter = gscatter_pre

    vartot = gscatter**2. + redshift_err2s
    gdi = (1. / np.sqrt(2.*np.pi*vartot)) * np.exp(-(dzs -
                                                     (gdelta + gslope*loglambdas))**2. / (2.*vartot))

    vals = np.log(gdi)
    bad, = np.where(~np.isfinite(vals))
    vals[bad] = -100.0

    t = -np.sum(vals)

    if fit_scatter:
        if pars[scatter_index: scatter_index + n_slope_nodes].min() < min_scatter:
            t += 10000

    return t

def fit_zlambda(nodes, slope_nodes, redshifts, dzs, redshift_errs, loglambdas,
                p0_delta, p0_slope, p0_scatter,
                fit_delta=False, fit_slope=False, fit_scatter=False,
                min_scatter=0.0):
    """
    Fit the afterburner correction parameters.

    Parameters
    ----------
    nodes: `np.array`
       Float array of spline nodes for mean correction
    slope_nodes: `np.array`
       Float array of spline nodes for slope (as a function of log-richness)
       correction.
    redshifts: `np.array`
       Float array of redshifts on x axis (either zspec or z_lambda).
    dzs: `np.array`
       Float array of delta_z (z_spec - z_lambda)
    redshift_errs: `np.array`
       Float array of errors on redshifts
    loglambdas: `np.array`
       Float array of log((lambda / scaleval) / pivot)
    p0_delta: `list`
       Initial guess at values of mean correction at nodes
    p0_slope: `list`
       Initial guess at values of slope correction at slope_nodes
    p0_scatter: `list`
       Initial guess at values of scatter corrections at slope_nodes
    fit_delta: `bool`, optional
       Fit the delta parameters?  Default is False.
    fit_slope: `bool`, optional
       Fit the slope parameters?  Default is False.
    fit_scatter: `bool`, optional
       Fit the scatter parameters?  Default is False.
    min_scatter: `float`, optional
       Minimum scatter.  Default is 0.0.

    Returns
    -------
    pars_delta: `np.array` (optional)
       Delta parameters.  Present if fit_delta=True.
    pars_slope: `np.array` (optional)
       Slope parameters.  Present if fit_slope=True.
    pars_scatter: `np.array` (optional)
       Scatter parameters.  Present if fit_scatter=True.
    """
    nodes = np.atleast_1d(nodes)
    slope_nodes = np.atleast_1d(slope_nodes)
    redshifts = np.atleast_1d(redshifts)
    dzs = np.atleast_1d(dzs)
    redshift_err2s = np.atleast_1d(redshift_errs)**2.
    loglambdas = np.atleast_1d(loglambdas)

    n_nodes = nodes.size
    n_slope_nodes = slope_nodes.size

    if redshifts.size != dzs.size:
        raise ValueError("Number of redshifts must be equal to dzs")
    if redshifts.size != redshift_err2s.size:
        raise ValueError("Number of redshifts must be equal to redshift_errs")
    if redshifts.size != loglambdas.size:
        raise ValueError("Number of redshifts must be equal to loglambdas")

    ctr = 0
    p0 = np.array([])
    delta_index = -1
    slope_index = -1
    scatter_index = -1
    
    if fit_delta:
        delta_index = 0
        ctr += n_nodes
        p0 = np.append(p0, p0_delta)
    if fit_slope:
        slope_index = ctr
        ctr += n_slope_nodes
        p0 = np.append(p0, p0_slope)
    if fit_scatter:
        scatter_index = ctr
        ctr += n_slope_nodes
        p0 = np.append(p0, p0_scatter)

    if ctr == 0:
        raise ValueError("Must select at least one of fit_delta, fit_slope, fit_scatter")

    # Precompute
    gdelta_pre, gslope_pre, gscatter_pre = None, None, None
    if not fit_delta:
        y2 = cubic_spline_compute_y2(nodes, p0_delta)
        gdelta_pre = cubic_spline_interpolate(redshifts, nodes, p0_delta, y2)
    if not fit_slope:
        y2 = cubic_spline_compute_y2(slope_nodes, p0_slope)
        gslope_pre = cubic_spline_interpolate(redshifts, slope_nodes, p0_slope, y2)
    if not fit_scatter:
        y2 = cubic_spline_compute_y2(slope_nodes, p0_scatter)
        gscatter_pre = np.clip(cubic_spline_interpolate(redshifts, slope_nodes, p0_scatter, y2), min_scatter, None)

    pars = scipy.optimize.fmin(zlambda_cost, p0, 
                               args=(fit_delta, fit_slope, fit_scatter, 
                                     delta_index, n_nodes, slope_index, n_slope_nodes, scatter_index,
                                     nodes, slope_nodes, redshifts, dzs, redshift_err2s, loglambdas, min_scatter,
                                     gdelta_pre, gslope_pre, gscatter_pre),
                               disp=False)

    retval = []
    if fit_delta:
        retval.append(pars[delta_index: delta_index + n_nodes])
    if fit_slope:
        retval.append(pars[slope_index: slope_index + n_slope_nodes])
    if fit_scatter:
        retval.append(pars[scatter_index: scatter_index + n_slope_nodes])

    return tuple(retval)

def calibrate_zlambda(config, corrslope=False):
    """
    Run the z_lambda afterburner calibration routine.

    Output goes to config.zlambdafile.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    corrslope: `bool`, optional
       Compute correction for richness slope.  Default is False.
    """
    # Import plotting libraries if needed
    if config.more_qa_plots:
        import matplotlib.pyplot as plt
        os.makedirs(config.plotpath, exist_ok=True)

    cat = ClusterCatalog.from_catfile(config.catfile, cosmo=config.cosmo)

    # We set the redshift according to the initial spec redshift for training
    cat.z = cat.z_spec_init

    use, = np.where((cat.Lambda/cat.scaleval > config.calib_zlambda_minlambda) &
                    (cat.scaleval > 0.0) &
                    (cat.maskfrac < config.max_maskfrac))

    nodes = make_nodes(config.zrange, config.calib_zlambda_nodesize)
    slope_nodes = make_nodes(config.zrange, config.calib_zlambda_slope_nodesize)

    # Confirm that we have enough clusters to do the fit
    hist, _ = np.histogram(cat.z[use], bins=nodes)
    if hist.min() == 0:
        raise RuntimeError("Calibration of zlambda correction cannot continue, as "
                           "there are redshift bins with no cluster spectra. "
                           "You must either reduce config.calib_zlambda_minlambda or "
                           "increase config.calib_zlambda_nodesize.")

    cat = cat[use]

    # QA plot: Initial z_spec vs z_lambda
    if config.more_qa_plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(cat.z, cat.z_lambda, alpha=0.5, s=10)
        ax.plot([cat.z.min(), cat.z.max()], [cat.z.min(), cat.z.max()], 'r--', lw=2, label='1:1')
        ax.set_xlabel('$z_{spec}$', fontsize=14)
        ax.set_ylabel('$z_\lambda$ (uncorrected)', fontsize=14)
        ax.set_title('Initial Spec-z vs Photo-z', fontsize=16)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.plotpath, 'zlambda_initial_comparison.png'), dpi=300)
        plt.close()

    # we have two runs, first "<zlambda|ztrue>" the second "<ztrue|zlambda>".

    out_struct = Entry(np.zeros(1, dtype=[('niter_true', 'i4'),
                                          ('offset_z', 'f4', nodes.size),
                                          ('offset', 'f4', nodes.size),
                                          ('offset_true', 'f4', (nodes.size, config.calib_zlambda_correct_niter)),
                                          ('slope_z', 'f4', slope_nodes.size),
                                          ('slope', 'f4', slope_nodes.size),
                                          ('slope_true', 'f4', (slope_nodes.size, config.calib_zlambda_correct_niter)),
                                          ('scatter', 'f4', slope_nodes.size),
                                          ('scatter_true', 'f4', (slope_nodes.size, config.calib_zlambda_correct_niter)),
                                          ('zred_uncorr', 'f4', nodes.size)]))

    out_struct.niter_true = config.calib_zlambda_correct_niter
    out_struct.offset_z = nodes
    out_struct.slope_z = slope_nodes

    for fitType in range(2):
        if fitType == 0:
            logger.info("Fitting zlambda corrections...")
            nziter = 1
        else:
            logger.info("Fitting ztrue corrections...")
            nziter = config.calib_zlambda_correct_niter

        # Make a backup copy of the catalog
        cat_orig = copy.deepcopy(cat)

        # ziter is the iterations stored for the correction
        # (these are needed for the ztrue corrections because
        #  we don't know ztrue at first, we need to zero-in on it)
        for ziter in range(nziter):
            # get the starting points

            delta_vals = np.zeros(nodes.size)
            slope_vals = np.zeros(slope_nodes.size)
            scatter_vals = np.zeros(slope_nodes.size) + 0.001

            # We have another iteration to remove outliers
            for outlier_iter in range(2):
                if outlier_iter == 0:
                    # Straightforward outlier removal
                    use, = np.where(np.abs(cat.z - cat.z_lambda) < 3.0*cat.z_lambda_e)
                else:
                    # Add on the estimate of the scatter
                    y2 = cubic_spline_compute_y2(slope_nodes, scatter_vals)
                    scatter = cubic_spline_interpolate(cat.z, slope_nodes, scatter_vals, y2)
                    use, = np.where(np.abs(cat.z - cat.z_lambda) < 3.0*np.sqrt(cat.z_lambda_e**2. + scatter**2.))

                if fitType == 0:
                    z_fit = cat.z[use]
                else:
                    z_fit = cat.z_lambda[use]

                dzs = cat.z[use] - cat.z_lambda[use]
                zerrs = cat.z_lambda_e[use]
                llam = np.log((cat.Lambda[use] / cat.scaleval[use]) / config.zlambda_pivot)

                delta_vals, = fit_zlambda(nodes, slope_nodes, z_fit, dzs, zerrs, llam, delta_vals, slope_vals, scatter_vals, fit_delta=True)
                if corrslope:
                    slope_vals, = fit_zlambda(nodes, slope_nodes, z_fit, dzs, zerrs, llam, delta_vals, slope_vals, scatter_vals, fit_slope=True)

                scatter_vals, = fit_zlambda(nodes, slope_nodes, z_fit, dzs, zerrs, llam, delta_vals, slope_vals, scatter_vals, fit_scatter=True)

                if corrslope:
                    delta_vals, slope_vals, scatter_vals = fit_zlambda(nodes, slope_nodes, z_fit, dzs, zerrs, llam, delta_vals, slope_vals, scatter_vals, fit_delta=True, fit_slope=True, fit_scatter=True)
                else:
                    delta_vals, scatter_vals = fit_zlambda(nodes, slope_nodes, z_fit, dzs, zerrs, llam, delta_vals, slope_vals, scatter_vals, fit_delta=True, fit_slope=False, fit_scatter=True)

                # Record the fit values in the output structure

                if (fitType == 0):
                    # Record offset, slope, scatter
                    out_struct.offset = delta_vals
                    out_struct.slope = slope_vals
                    out_struct.scatter = scatter_vals
                else:
                    # Record offset_true, slope_true, scatter_true
                    out_struct.offset_true[:, ziter] = delta_vals
                    out_struct.slope_true[:, ziter] = slope_vals
                    out_struct.scatter_true[:, ziter] = scatter_vals

                # QA plots after final outlier iteration
                if config.more_qa_plots and outlier_iter == 1:
                    fittype_str = "zlambda" if fitType == 0 else f"ztrue_iter{ziter}"
                    
                    # Plot residuals vs redshift with fitted correction
                    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
                    
                    # Top panel: residuals
                    axes[0].scatter(z_fit, dzs, alpha=0.5, s=10, label='All clusters')
                    y2_delta = cubic_spline_compute_y2(nodes, delta_vals)
                    z_grid = np.linspace(z_fit.min(), z_fit.max(), 200)
                    axes[0].plot(z_grid, cubic_spline_interpolate(z_grid, nodes, delta_vals, y2_delta), 'r-', lw=2, label='Offset correction')
                    if corrslope:
                        y2_slope = cubic_spline_compute_y2(slope_nodes, slope_vals)
                        # Show correction at pivot richness
                        axes[0].plot(z_grid, cubic_spline_interpolate(z_grid, nodes, delta_vals, y2_delta), 'g--', lw=2, label='Offset + slope (pivot)')
                    axes[0].axhline(0, color='k', ls='--', alpha=0.5)
                    axes[0].set_xlabel('Redshift', fontsize=12)
                    axes[0].set_ylabel('$\Delta z = z_{spec} - z_\lambda$', fontsize=12)
                    axes[0].set_title(f'Residuals ({fittype_str})', fontsize=14)
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)
                    
                    # Bottom panel: scatter
                    y2_scatter = cubic_spline_compute_y2(slope_nodes, scatter_vals)
                    axes[1].plot(z_grid, cubic_spline_interpolate(z_grid, slope_nodes, scatter_vals, y2_scatter), 'b-', lw=2, label='Fitted scatter')
                    axes[1].fill_between(z_grid, 0, cubic_spline_interpolate(z_grid, slope_nodes, scatter_vals, y2_scatter), alpha=0.3)
                    axes[1].set_xlabel('Redshift', fontsize=12)
                    axes[1].set_ylabel('$\sigma_z$', fontsize=12)
                    axes[1].set_title(f'Scatter ({fittype_str})', fontsize=14)
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(config.plotpath, f'zlambda_residuals_{fittype_str}.png'), dpi=300)
                    plt.close()
                    
                    # Histogram of residuals
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(dzs, bins=50, alpha=0.7, edgecolor='black')
                    ax.axvline(0, color='r', ls='--', lw=2, label='Zero')
                    ax.axvline(np.median(dzs), color='g', ls='--', lw=2, label=f'Median={np.median(dzs):.4f}')
                    ax.set_xlabel('$\Delta z = z_{spec} - z_\lambda$', fontsize=12)
                    ax.set_ylabel('Number of clusters', fontsize=12)
                    ax.set_title(f'Residual Distribution ({fittype_str})', fontsize=14)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(config.plotpath, f'zlambda_residual_hist_{fittype_str}.png'), dpi=300)
                    plt.close()

            # Run the corrections if we're doing ztrue fitType == 1

            if fitType == 1:
                zlambda_corr_data = read_zlambda_correction(pars=out_struct,
                                                            zrange=np.array([config.zrange[0] - 0.02,
                                                                             config.zrange[1] + 0.07]),
                                                            zbinsize=config.zlambda_binsize,
                                                            zlambda_pivot=config.zlambda_pivot)

                # reset the catalog before applying correction
                cat = copy.deepcopy(cat_orig)

                for cluster in cat:
                    # Need to apply correction here
                    zlam, zlam_e = apply_zlambda_correction(zlambda_corr_data, cluster.Lambda, cluster.z_lambda, cluster.z_lambda_e)
                    cluster.z_lambda = zlam
                    cluster.z_lambda_e = zlam_e

    # QA plot: Final corrected z_spec vs z_lambda
    if config.more_qa_plots:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Before correction (cat_orig)
        axes[0].scatter(cat_orig.z, cat_orig.z_lambda, alpha=0.5, s=10, c=cat_orig.Lambda/cat_orig.scaleval, 
                       cmap='Reds', vmin=20, vmax=100)
        axes[0].plot([cat_orig.z.min(), cat_orig.z.max()], [cat_orig.z.min(), cat_orig.z.max()], 
                    'r--', lw=2, label='1:1')
        axes[0].set_xlabel('$z_{spec}$', fontsize=14)
        axes[0].set_ylabel('$z_\lambda$ (uncorrected)', fontsize=14)
        axes[0].set_title('Before Correction', fontsize=16)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # After correction (cat)
        sc = axes[1].scatter(cat.z, cat.z_lambda, alpha=0.5, s=10, c=cat.Lambda/cat.scaleval, 
                            cmap='Reds', vmin=20, vmax=100)
        axes[1].plot([cat.z.min(), cat.z.max()], [cat.z.min(), cat.z.max()], 
                    'r--', lw=2, label='1:1')
        axes[1].set_xlabel('$z_{spec}$', fontsize=14)
        axes[1].set_ylabel('$z_\lambda$ (corrected)', fontsize=14)
        axes[1].set_title('After Correction', fontsize=16)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.colorbar(sc, ax=axes[1], label='$\lambda$')
        plt.tight_layout()
        plt.savefig(os.path.join(config.plotpath, 'zlambda_final_comparison.png'), dpi=300)
        plt.close()
        
        # Plot correction parameters
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Offset
        axes[0].plot(nodes, out_struct.offset, 'o-', lw=2, label='<z_lambda|z_true>')
        for i in range(config.calib_zlambda_correct_niter):
            axes[0].plot(nodes, out_struct.offset_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
        axes[0].axhline(0, color='k', ls='--', alpha=0.3)
        axes[0].set_xlabel('Redshift', fontsize=12)
        axes[0].set_ylabel('Offset correction', fontsize=12)
        axes[0].set_title('Offset Correction Parameters', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Slope
        axes[1].plot(slope_nodes, out_struct.slope, 'o-', lw=2, label='<z_lambda|z_true>')
        for i in range(config.calib_zlambda_correct_niter):
            axes[1].plot(slope_nodes, out_struct.slope_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
        axes[1].axhline(0, color='k', ls='--', alpha=0.3)
        axes[1].set_xlabel('Redshift', fontsize=12)
        axes[1].set_ylabel('Slope correction', fontsize=12)
        axes[1].set_title('Slope Correction Parameters', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Scatter
        axes[2].plot(slope_nodes, out_struct.scatter, 'o-', lw=2, label='<z_lambda|z_true>')
        for i in range(config.calib_zlambda_correct_niter):
            axes[2].plot(slope_nodes, out_struct.scatter_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
        axes[2].set_xlabel('Redshift', fontsize=12)
        axes[2].set_ylabel('$\sigma_z$', fontsize=12)
        axes[2].set_title('Scatter Parameters', fontsize=14)
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.plotpath, 'zlambda_correction_parameters.png'), dpi=300)
        plt.close()

    # Need to do the zred uncorr calibration, blah.
    zred_uncorr = fit_med_z(nodes, cat_orig.z_lambda, cat_orig.zred, nodes)

    out_struct.zred_uncorr = zred_uncorr

    # And now save the file

    hdr = fitsio.FITSHDR()
    hdr['ZRANGE0'] = config.zrange[0] - 0.02
    hdr['ZRANGE1'] = config.zrange[1] + 0.07
    hdr['ZBINSIZE'] = config.zlambda_binsize
    hdr['ZLAMPIV'] = config.zlambda_pivot

    out_struct.to_fits_file(config.zlambdafile, header=hdr)