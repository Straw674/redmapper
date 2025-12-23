"""Classes to calibrate the z_lambda afterburner
"""
import os
import numpy as np
import fitsio
import time
import scipy.optimize
import copy

from ..configuration import Configuration
from ..redsequence import RedSequenceColorPar
from ..galaxy import GalaxyCatalog
from ..cluster import ClusterCatalog
from ..zlambda import ZlambdaCorrectionPar
from ..utilities import make_nodes, CubicSpline, interpol
from ..fitters import MedZFitter
from ..catalog import Entry

class ZLambdaFitter(object):
    """
    Class to fit the z_lambda afterburner spline function.
    """

    def __init__(self, nodes, slope_nodes, redshifts, dzs, redshift_errs, loglambdas):
        """
        Instantiate a ZLambdaFitter object.

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
        """
        self._nodes = np.atleast_1d(nodes)
        self._slope_nodes = np.atleast_1d(slope_nodes)
        self._redshifts = np.atleast_1d(redshifts)
        self._dzs = np.atleast_1d(dzs)
        self._redshift_err2s = np.atleast_1d(redshift_errs)**2.
        self._loglambdas = np.atleast_1d(loglambdas)

        self._n_nodes = self._nodes.size
        self._n_slope_nodes = self._slope_nodes.size

        if self._redshifts.size != self._dzs.size:
            raise ValueError("Number of redshifts must be equal to dzs")
        if self._redshifts.size != self._redshift_err2s.size:
            raise ValueError("Number of redshifts must be equal to redshift_errs")
        if self._redshifts.size != self._loglambdas.size:
            raise ValueError("Number of redshifts must be equal to loglambdas")

    def fit(self, p0_delta, p0_slope, p0_scatter,
            fit_delta=False, fit_slope=False, fit_scatter=False,
            min_scatter=0.0):
        """
        Fit the afterburner correction parameters.

        Parameters
        ----------
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
        pars_delta: `list`
           Delta parameters.  Present if fit_delta=True.
        pars_slope: `list`
           Slope parameters.  Present if fit_slope=True.
        pars_scatter: `list`
           Scatter parameters.  Present if fit_scatter=True.
        """
        self._fit_delta = fit_delta
        self._fit_slope = fit_slope
        self._fit_scatter = fit_scatter
        self._min_scatter = min_scatter

        ctr = 0
        p0 = np.array([])
        if self._fit_delta:
            self._delta_index = 0
            ctr += self._n_nodes
            p0 = np.append(p0, p0_delta)
        if self._fit_slope:
            self._slope_index = ctr
            ctr += self._n_slope_nodes
            p0 = np.append(p0, p0_slope)
        if self._fit_scatter:
            self._scatter_index = ctr
            ctr += self._n_slope_nodes
            p0 = np.append(p0, p0_scatter)

        if ctr == 0:
            raise ValueError("Must select at least one of fit_delta, fit_slope, fit_scatter")

        # Precompute
        if not self._fit_delta:
            spl = CubicSpline(self._nodes, p0_delta)
            self._gdelta = spl(self._redshifts)
        if not self._fit_slope:
            spl = CubicSpline(self._slope_nodes, p0_slope)
            self._gslope = spl(self._redshifts)
        if not self._fit_scatter:
            spl = CubicSpline(self._slope_nodes, p0_scatter)
            self._gscatter = np.clip(spl(self._redshifts), self._min_scatter, None)

        # FIXME
        pars = scipy.optimize.fmin(self, p0, disp=False)

        retval = []
        if self._fit_delta:
            retval.append(pars[self._delta_index: self._delta_index + self._n_nodes])
        if self._fit_slope:
            retval.append(pars[self._slope_index: self._slope_index + self._n_slope_nodes])
        if self._fit_scatter:
            retval.append(pars[self._scatter_index: self._scatter_index + self._n_slope_nodes])

        return retval

    def __call__(self, pars):
        """
        Calculate the negative log-likelihood cost function for a set of parameters.

        Parameters
        ----------
        pars: `list`
           Parameters for fit, including delta, slope, scatter concatenated.

        Returns
        -------
        t: `float`
           Total cost function of negative log-likelihood to minimize.
        """
        if self._fit_delta:
            spl = CubicSpline(self._nodes, pars[self._delta_index: self._delta_index + self._n_nodes])
            gdelta = spl(self._redshifts)
        else:
            gdelta = self._gdelta

        if self._fit_slope:
            spl = CubicSpline(self._slope_nodes, pars[self._slope_index: self._slope_index + self._n_slope_nodes])
            gslope = spl(self._redshifts)
        else:
            gslope = self._gslope

        if self._fit_scatter:
            spl = CubicSpline(self._slope_nodes, pars[self._scatter_index: self._scatter_index + self._n_slope_nodes])
            gscatter = np.clip(spl(self._redshifts), self._min_scatter, None)
        else:
            gscatter = self._gscatter

        vartot = gscatter**2. + self._redshift_err2s
        gdi = (1. / np.sqrt(2.*np.pi*vartot)) * np.exp(-(self._dzs -
                                                         (gdelta + gslope*self._loglambdas))**2. / (2.*vartot))

        vals = np.log(gdi)
        bad, = np.where(~np.isfinite(vals))
        vals[bad] = -100.0

        t = -np.sum(vals)

        if self._fit_scatter:
            if pars[self._scatter_index: self._scatter_index + self._n_slope_nodes].min() < self._min_scatter:
                t += 10000

        return t

class ZLambdaCalibrator(object):
    """
    Class to calibrate the z_lambda correction afterburner.
    """

    def __init__(self, config, corrslope=False):
        """
        Instantiate a ZLambdaCalibrator object.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        corrslope: `bool`, optional
           Compute correction for richness slope.  Default is False.
        """
        self.config = config
        self.corrslope = corrslope

    def run(self):
        """
        Run the z_lambda afterburner calibration routine.

        Output goes to self.config.zlambdafile.
        """
        # Import plotting libraries if needed
        if self.config.more_qa_plots:
            import matplotlib.pyplot as plt
            os.makedirs(self.config.plotpath, exist_ok=True)

        cat = ClusterCatalog.from_catfile(self.config.catfile, cosmo=self.config.cosmo)

        # We set the redshift according to the initial spec redshift for training
        cat.z = cat.z_spec_init

        use, = np.where((cat.Lambda/cat.scaleval > self.config.calib_zlambda_minlambda) &
                        (cat.scaleval > 0.0) &
                        (cat.maskfrac < self.config.max_maskfrac))

        nodes = make_nodes(self.config.zrange, self.config.calib_zlambda_nodesize)
        slope_nodes = make_nodes(self.config.zrange, self.config.calib_zlambda_slope_nodesize)

        # Confirm that we have enough clusters to do the fit
        hist, _ = np.histogram(cat.z[use], bins=nodes)
        if hist.min() == 0:
            raise RuntimeError("Calibration of zlambda correction cannot continue, as "
                               "there are redshift bins with no cluster spectra. "
                               "You must either reduce config.calib_zlambda_minlambda or "
                               "increase config.calib_zlambda_nodesize.")

        cat = cat[use]

        # QA plot: Initial z_spec vs z_lambda
        if self.config.more_qa_plots:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(cat.z, cat.z_lambda, alpha=0.5, s=10)
            ax.plot([cat.z.min(), cat.z.max()], [cat.z.min(), cat.z.max()], 'r--', lw=2, label='1:1')
            ax.set_xlabel('$z_{spec}$', fontsize=14)
            ax.set_ylabel('$z_\lambda$ (uncorrected)', fontsize=14)
            ax.set_title('Initial Spec-z vs Photo-z', fontsize=16)
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plotpath, 'zlambda_initial_comparison.png'), dpi=300)
            plt.close()

        # we have two runs, first "<zlambda|ztrue>" the second "<ztrue|zlambda>".

        out_struct = Entry(np.zeros(1, dtype=[('niter_true', 'i4'),
                                              ('offset_z', 'f4', nodes.size),
                                              ('offset', 'f4', nodes.size),
                                              ('offset_true', 'f4', (nodes.size, self.config.calib_zlambda_correct_niter)),
                                              ('slope_z', 'f4', slope_nodes.size),
                                              ('slope', 'f4', slope_nodes.size),
                                              ('slope_true', 'f4', (slope_nodes.size, self.config.calib_zlambda_correct_niter)),
                                              ('scatter', 'f4', slope_nodes.size),
                                              ('scatter_true', 'f4', (slope_nodes.size, self.config.calib_zlambda_correct_niter)),
                                              ('zred_uncorr', 'f4', nodes.size)]))

        out_struct.niter_true = self.config.calib_zlambda_correct_niter
        out_struct.offset_z = nodes
        out_struct.slope_z = slope_nodes

        for fitType in range(2):
            if fitType == 0:
                self.config.logger.info("Fitting zlambda corrections...")
                nziter = 1
            else:
                self.config.logger.info("Fitting ztrue corrections...")
                nziter = self.config.calib_zlambda_correct_niter

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
                        spl = CubicSpline(slope_nodes, scatter_vals)
                        scatter = spl(cat.z)
                        use, = np.where(np.abs(cat.z - cat.z_lambda) < 3.0*np.sqrt(cat.z_lambda_e**2. + scatter**2.))

                    if fitType == 0:
                        z_fit = cat.z[use]
                    else:
                        z_fit = cat.z_lambda[use]

                    dzs = cat.z[use] - cat.z_lambda[use]
                    zerrs = cat.z_lambda_e[use]
                    llam = np.log((cat.Lambda[use] / cat.scaleval[use]) / self.config.zlambda_pivot)

                    fitter = ZLambdaFitter(nodes, slope_nodes, z_fit, dzs, zerrs, llam)
                    delta_vals, = fitter.fit(delta_vals, slope_vals, scatter_vals, fit_delta=True)
                    if self.corrslope:
                        slope_vals, = fitter.fit(delta_vals, slope_vals, scatter_vals, fit_slope=True)

                    scatter_vals, = fitter.fit(delta_vals, slope_vals, scatter_vals, fit_scatter=True)

                    if self.corrslope:
                        delta_vals, slope_vals, scatter_vals = fitter.fit(delta_vals, slope_vals, scatter_vals, fit_delta=True, fit_slope=True, fit_scatter=True)
                    else:
                        delta_vals, scatter_vals = fitter.fit(delta_vals, slope_vals, scatter_vals, fit_delta=True, fit_slope=False, fit_scatter=True)

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
                    if self.config.more_qa_plots and outlier_iter == 1:
                        fittype_str = "zlambda" if fitType == 0 else f"ztrue_iter{ziter}"
                        
                        # Plot residuals vs redshift with fitted correction
                        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
                        
                        # Top panel: residuals
                        axes[0].scatter(z_fit, dzs, alpha=0.5, s=10, label='All clusters')
                        spl_delta = CubicSpline(nodes, delta_vals)
                        z_grid = np.linspace(z_fit.min(), z_fit.max(), 200)
                        axes[0].plot(z_grid, spl_delta(z_grid), 'r-', lw=2, label='Offset correction')
                        if self.corrslope:
                            spl_slope = CubicSpline(slope_nodes, slope_vals)
                            # Show correction at pivot richness
                            axes[0].plot(z_grid, spl_delta(z_grid), 'g--', lw=2, label='Offset + slope (pivot)')
                        axes[0].axhline(0, color='k', ls='--', alpha=0.5)
                        axes[0].set_xlabel('Redshift', fontsize=12)
                        axes[0].set_ylabel('$\Delta z = z_{spec} - z_\lambda$', fontsize=12)
                        axes[0].set_title(f'Residuals ({fittype_str})', fontsize=14)
                        axes[0].legend()
                        axes[0].grid(alpha=0.3)
                        
                        # Bottom panel: scatter
                        spl_scatter = CubicSpline(slope_nodes, scatter_vals)
                        axes[1].plot(z_grid, spl_scatter(z_grid), 'b-', lw=2, label='Fitted scatter')
                        axes[1].fill_between(z_grid, 0, spl_scatter(z_grid), alpha=0.3)
                        axes[1].set_xlabel('Redshift', fontsize=12)
                        axes[1].set_ylabel('$\sigma_z$', fontsize=12)
                        axes[1].set_title(f'Scatter ({fittype_str})', fontsize=14)
                        axes[1].legend()
                        axes[1].grid(alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.config.plotpath, f'zlambda_residuals_{fittype_str}.png'), dpi=300)
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
                        plt.savefig(os.path.join(self.config.plotpath, f'zlambda_residual_hist_{fittype_str}.png'), dpi=300)
                        plt.close()

                # Run the corrections if we're doing ztrue fitType == 1

                if fitType == 1:
                    zlambda_corr = ZlambdaCorrectionPar(pars=out_struct,
                                                        zrange=np.array([self.config.zrange[0] - 0.02,
                                                                         self.config.zrange[1] + 0.07]),
                                                        zbinsize=self.config.zlambda_binsize,
                                                        zlambda_pivot=self.config.zlambda_pivot)

                    # reset the catalog before applying correction
                    cat = copy.deepcopy(cat_orig)

                    for cluster in cat:
                        # Need to apply correction here
                        zlam, zlam_e = zlambda_corr.apply_correction(cluster.Lambda, cluster.z_lambda, cluster.z_lambda_e)
                        cluster.z_lambda = zlam
                        cluster.z_lambda_e = zlam_e

        # QA plot: Final corrected z_spec vs z_lambda
        if self.config.more_qa_plots:
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
            plt.savefig(os.path.join(self.config.plotpath, 'zlambda_final_comparison.png'), dpi=300)
            plt.close()
            
            # Plot correction parameters
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Offset
            axes[0].plot(nodes, out_struct.offset, 'o-', lw=2, label='<z_lambda|z_true>')
            for i in range(self.config.calib_zlambda_correct_niter):
                axes[0].plot(nodes, out_struct.offset_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
            axes[0].axhline(0, color='k', ls='--', alpha=0.3)
            axes[0].set_xlabel('Redshift', fontsize=12)
            axes[0].set_ylabel('Offset correction', fontsize=12)
            axes[0].set_title('Offset Correction Parameters', fontsize=14)
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Slope
            axes[1].plot(slope_nodes, out_struct.slope, 'o-', lw=2, label='<z_lambda|z_true>')
            for i in range(self.config.calib_zlambda_correct_niter):
                axes[1].plot(slope_nodes, out_struct.slope_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
            axes[1].axhline(0, color='k', ls='--', alpha=0.3)
            axes[1].set_xlabel('Redshift', fontsize=12)
            axes[1].set_ylabel('Slope correction', fontsize=12)
            axes[1].set_title('Slope Correction Parameters', fontsize=14)
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            # Scatter
            axes[2].plot(slope_nodes, out_struct.scatter, 'o-', lw=2, label='<z_lambda|z_true>')
            for i in range(self.config.calib_zlambda_correct_niter):
                axes[2].plot(slope_nodes, out_struct.scatter_true[:, i], 's--', alpha=0.5, label=f'<z_true|z_lambda> iter{i}')
            axes[2].set_xlabel('Redshift', fontsize=12)
            axes[2].set_ylabel('$\sigma_z$', fontsize=12)
            axes[2].set_title('Scatter Parameters', fontsize=14)
            axes[2].legend()
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plotpath, 'zlambda_correction_parameters.png'), dpi=300)
            plt.close()

        # Need to do the zred uncorr calibration, blah.
        medfitter = MedZFitter(nodes, cat_orig.z_lambda, cat_orig.zred)
        p0 = nodes
        zred_uncorr = medfitter.fit(p0)

        out_struct.zred_uncorr = zred_uncorr

        # And now save the file

        hdr = fitsio.FITSHDR()
        hdr['ZRANGE0'] = self.config.zrange[0] - 0.02
        hdr['ZRANGE1'] = self.config.zrange[1] + 0.07
        hdr['ZBINSIZE'] = self.config.zlambda_binsize
        hdr['ZLAMPIV'] = self.config.zlambda_pivot

        out_struct.to_fits_file(self.config.zlambdafile, header=hdr)
