"""Classes to describe clusters and cluster catalogs.

This module contains the logic for cluster and cluster catalog handling,
transitioning to a functional style using astropy.table.Table.
"""
import fitsio
import esutil
import numpy as np
import itertools
import scipy.optimize
import scipy.integrate
import copy
from astropy.table import Table, Row

from .solver_nfw import Solver
from .catalog import Catalog, Entry
from .utilities import chisq_pdf, calc_theta_i, schechter_pdf, nfw_pdf
from .mask import get_mask_values, calc_maskcorr
from .redsequence import read_redsequence, redsequence_zindex, redsequence_lumrefmagindex, redsequence_mstar, compute_redsequence_chisq
from esutil.cosmology import Cosmo
from .galaxy import GalaxyCatalog
from .depth_fitting import calcErrorModel
from . import depthmap
from .background import compute_background, compute_zred_background

cluster_dtype_base = [('MEM_MATCH_ID', 'i4'),
                      ('RA', 'f8'),
                      ('DEC', 'f8'),
                      ('Z', 'f4'),
                      ('REFMAG', 'f4'),
                      ('REFMAG_ERR', 'f4'),
                      ('LAMBDA', 'f4'),
                      ('LAMBDA_E', 'f4'),
                      ('Z_LAMBDA', 'f4'),
                      ('Z_LAMBDA_E', 'f4'),
                      ('CG_SPEC_Z', 'f4'),
                      ('Z_SPEC_INIT', 'f4'),
                      ('Z_INIT', 'f4'),
                      ('R_LAMBDA', 'f4'),
                      ('R_MASK', 'f4'),
                      ('SCALEVAL', 'f4'),
                      ('MASKFRAC', 'f4'),
                      ('ZRED', 'f4'),
                      ('ZRED_E', 'f4'),
                      ('ZRED_CHISQ', 'f4'),
                      ('CHISQ', 'f4'),
                      ('Z_LAMBDA_NITER', 'i2'),
                      ('EBV_MEAN', 'f4'),
                      ('LNLAMLIKE', 'f4'),
                      ('LNCGLIKE', 'f4'),
                      ('LNLIKE', 'f4'),
                      ('RA_ORIG', 'f8'),
                      ('DEC_ORIG', 'f8'),
                      ('W', 'f4'),
                      ('DLAMBDA_DZ', 'f4'),
                      ('DLAMBDA_DZ2', 'f4'),
                      ('DLAMBDAVAR_DZ', 'f4'),
                      ('DLAMBDAVAR_DZ2', 'f4'),
                      ('Z_LAMBDA_RAW', 'f4'),
                      ('Z_LAMBDA_E_RAW', 'f4'),
                      ('BKG_LOCAL', 'f4'),
                      ('LIM_EXPTIME', 'f4'),
                      ('LIM_LIMMAG', 'f4'),
                      ('LIM_LIMMAG_HARD', 'f4'),
                      ('LAMBDA_C', 'f4'),
                      ('LAMBDA_CE', 'f4'),
                      ('NCENT_GOOD', 'i2'),
                      ('MASKGAL_INDEX', 'i2')]

member_dtype_base = [('MEM_MATCH_ID', 'i4'),
                     ('ID', 'i8'),
                     ('Z', 'f4'),
                     ('RA', 'f8'),
                     ('DEC', 'f8'),
                     ('R', 'f4'),
                     ('P', 'f4'),
                     ('PFREE', 'f4'),
                     ('PCOL', 'f4'),
                     ('THETA_I', 'f4'),
                     ('THETA_R', 'f4'),
                     ('REFMAG', 'f4'),
                     ('REFMAG_ERR', 'f4'),
                     ('ZRED', 'f4'),
                     ('ZRED_E', 'f4'),
                     ('ZRED_CHISQ','f4'),
                     ('CHISQ', 'f4'),
                     ('EBV', 'f4'),
                     ('ZSPEC', 'f4')]


class Cluster(Entry):
    """
    Class for a single galaxy cluster, based on Entry (astropy Row).
    """

    def __init__(self, *args, **kwargs):
        """
        Instantiate a Cluster object.
        Supports standard Row(table, index) and legacy Cluster(cat_vals=...)
        """
        if len(args) > 0 and isinstance(args[0], Table):
            super().__init__(*args, **kwargs)
            # Row init doesn't take extra kwargs, they should be in self.table
            self.neighbors = None
            self._mstar = None
            self._mpc_scale = None
            self._redshift = None
            if self.z > 0.0:
                self.redshift = self.z
        else:
            # Legacy initialization
            cat_vals = kwargs.pop('cat_vals', None)
            dtype = kwargs.pop('dtype', None)
            config = kwargs.pop('config', None)
            
            if cat_vals is None:
                if dtype is not None:
                    cat_vals = np.zeros(1, dtype=dtype)
                else:
                    if config is not None:
                        cat_vals = np.zeros(1, dtype=config.cluster_dtype)
                    else:
                        cat_vals = np.zeros(1, dtype=cluster_dtype_base)
            
            # Create a 1-row catalog to host this cluster
            parent = ClusterCatalog(cat_vals, config=config, **kwargs)
            # We cannot easily BE that row by initializing ourselves.
            # Instead, we'll initialize as a Row of that parent.
            super().__init__(parent, 0)
            
            self.neighbors = None
            self._mstar = None
            self._mpc_scale = None
            self._redshift = None
            
            neighbors = kwargs.pop('neighbors', None)
            if neighbors is not None:
                self.set_neighbors(neighbors)
            
            if self.z > 0.0 and self.zredstr is not None:
                self.redshift = self.z

    @property
    def r0(self): return getattr(self.table, 'r0', 1.0)
    @r0.setter
    def r0(self, value): self.table.r0 = value

    @property
    def beta(self): return getattr(self.table, 'beta', 0.2)
    @beta.setter
    def beta(self, value): self.table.beta = value

    @property
    def config(self): return getattr(self.table, 'config', None)
    @config.setter
    def config(self, value): self.table.config = value

    @property
    def zredstr(self): return getattr(self.table, 'zredstr', None)
    @zredstr.setter
    def zredstr(self, value): self.table.zredstr = value

    @property
    def bkg(self): return getattr(self.table, 'bkg', None)
    @bkg.setter
    def bkg(self, value): self.table.bkg = value

    @property
    def cbkg(self): return getattr(self.table, 'cbkg', None)
    @cbkg.setter
    def cbkg(self, value): self.table.cbkg = value

    @property
    def zredbkg(self): return getattr(self.table, 'zredbkg', None)
    @zredbkg.setter
    def zredbkg(self, value): self.table.zredbkg = value

    @property
    def cosmo(self): return getattr(self.table, 'cosmo', Cosmo())
    @cosmo.setter
    def cosmo(self, value): self.table.cosmo = value

    def reset(self):
        """Reset richness and redshift."""
        self.Lambda = -1.0
        self.z_lambda = -1.0

    def set_neighbors(self, neighbors):
        """Set the neighbor galaxy catalog."""
        if (neighbors.__class__ is not GalaxyCatalog and neighbors is not None):
            raise ValueError("Cluster neighbors must be a GalaxyCatalog")

        self.neighbors = None
        if (neighbors is not None):
            self.neighbors = copy.deepcopy(neighbors)

            neighbor_extra_dtype = [('R', 'f8'),
                                    ('DIST', 'f8'),
                                    ('CHISQ', 'f8'),
                                    ('ZRED_CHISQ', 'f8'),
                                    ('PFREE', 'f8'),
                                    ('THETA_I', 'f8'),
                                    ('THETA_R', 'f8'),
                                    ('P', 'f8'),
                                    ('PCOL', 'f8'),
                                    ('PMEM', 'f8'),
                                    ('INDEX', 'i8'),
                                    ('CENTERING_CAND', 'i2')]

            dtype_augment = [dt for dt in neighbor_extra_dtype if dt[0].lower() not in self.neighbors.colnames]
            if len(dtype_augment) > 0:
                self.neighbors.add_fields(dtype_augment)

            if 'pfree' in self.neighbors.colnames:
                # We need to check if it's new. add_fields was just called.
                # Actually, if we use Catalog.add_fields, it initializes to 0.
                pass

            # In the old code, it check if 'PFREE' was in dtype_augment
            if any(dt[0].lower() == 'pfree' for dt in dtype_augment):
                self.neighbors.pfree[:] = 1.0
            
            if any(dt[0].lower() == 'zred_chisq' for dt in dtype_augment):
                self.neighbors.zred_chisq = self.neighbors.chisq

    def find_neighbors(self, radius, galcat, megaparsec=False, maxmag=None):
        """Find neighbors from a full galaxy catalog."""
        if radius is None:
            raise ValueError("A radius must be specified")
        if galcat is None:
            raise ValueError("A GalaxyCatalog object must be specified.")

        if megaparsec:
            radius_degrees = radius / self.mpc_scale
        else:
            radius_degrees = radius

        indices, dists = galcat.match_one(self.ra, self.dec, radius_degrees)

        if maxmag is not None:
            use, = np.where(galcat.refmag[indices] <= maxmag)
            indices = indices[use]
            dists = dists[use]

        self.set_neighbors(galcat[indices])
        self.neighbors.dist = dists
        self.neighbors.index = indices
        self._compute_neighbor_r()

    def update_neighbors_dist(self):
        """Update neighbor distances."""
        self.neighbors.dist = esutil.coords.sphdist(self.ra, self.dec,
                                                    self.neighbors.ra, self.neighbors.dec)
        self._compute_neighbor_r()

    def clear_neighbors(self):
        """Clear neighbors."""
        self.neighbors = None

    def _calc_radial_profile(self, idx=None, rscale=0.15):
        if idx is None:
            idx = np.arange(len(self.neighbors))
        sigx = nfw_pdf(self.neighbors.r[idx], rscale=rscale)
        return sigx

    def _calc_luminosity(self, normmag, idx=None):
        if idx is None:
            idx = np.arange(len(self.neighbors))
        zind = redsequence_zindex(self.zredstr, self._redshift)
        refind = redsequence_lumrefmagindex(self.zredstr, normmag)
        normalization = self.zredstr['lumnorm'][refind, zind]
        mstar = redsequence_mstar(self.zredstr, self._redshift)
        phi = schechter_pdf(self.neighbors.refmag[idx], alpha=self.zredstr['alpha'], mstar=mstar)
        return phi / normalization

    def calc_bkg_density(self, r, chisq, refmag):
        sigma_g = compute_background(self.bkg, np.full(r.size, self._redshift), chisq, refmag)
        return 2. * np.pi * r * (sigma_g/self.mpc_scale**2.)

    def calc_cbkg_density(self, r, col_index, col, refmag):
        from .color_background import sigma_g_diagonal
        sigma_g = sigma_g_diagonal(self.cbkg, col_index, col, refmag)
        return 2. * np.pi * r * (sigma_g / self.mpc_scale**2.)

    def calc_zred_bkg_density(self, r, zred, refmag):
        if self.zredbkg is None:
            raise AttributeError("zredbkg has not been set for this cluster")
        sigma_g = compute_zred_background(self.zredbkg, zred, refmag)
        return 2. * np.pi * r * (sigma_g / self.mpc_scale**2.)

    def compute_bkg_local(self, mask, depth):
        ras = self.ra + (mask['maskgals'].x_uniform/self.mpc_scale)/np.cos(np.deg2rad(self.dec))
        decs = self.dec + mask['maskgals'].y_uniform/self.mpc_scale

        maxmag = self.mstar - 2.5*np.log10(self.config.lval_reference)
        maskgals_mark = get_mask_values(mask['mask_data'], ras, decs, rng=mask['rng'], config=mask['config'])
        maskgals_refmag = self.mstar + mask['maskgals'].m
        if isinstance(depth, dict):
            maskgals_depth = depthmap.get_depth_values(depth, ras, decs)[0]
        else:
            from .depth_fitting import calcErrorModel
            limpars, fail = calcErrorModel(self.neighbors.refmag, self.neighbors.refmag_err, calcErr=False)
            if fail:
                maskgals_depth = depth['LIMMAG']
            else:
                maskgals_depth = limpars['LIMMAG']

        sigma_g_maskgals = compute_background(self.bkg, np.full(mask['maskgals'].chisq.size, self._redshift), mask['maskgals'].chisq, mask['maskgals'].refmag)

        bright_enough, = np.where((mask['maskgals'].refmag < maxmag) & (np.isfinite(sigma_g_maskgals)) &
                                  (mask['maskgals'].chisq_pdf > 0.0) & (mask['maskgals'].lum_pdf > 0.0) &
                                  (mask['maskgals'].refmag < maskgals_depth))

        prediction = np.sum(sigma_g_maskgals[bright_enough].astype(np.float64) / (mask['maskgals'].chisq_pdf[bright_enough].astype(np.float64) * mask['maskgals'].lum_pdf[bright_enough].astype(np.float64))) / float(bright_enough.size)

        in_annulus, = np.where((mask['maskgals'].r_uniform > self.config.bkg_local_annuli[0]) &
                               (mask['maskgals'].r_uniform < self.config.bkg_local_annuli[1]))
        in_annulus_gd, = np.where(maskgals_mark[in_annulus])
        annulus_area = (np.pi*((self.config.bkg_local_annuli[1]/self.mpc_scale)**2. -
                               (self.config.bkg_local_annuli[0]/self.mpc_scale)**2.) *
                        (float(in_annulus_gd.size) / float(in_annulus.size)))

        if isinstance(depth, dict):
            neighbors_depth = depthmap.get_depth_values(depth, self.neighbors.ra, self.neighbors.dec)[0]
        else:
            neighbors_depth = maskgals_depth

        neighbors_in_annulus, = np.where((self.neighbors.r > self.config.bkg_local_annuli[0]) &
                                         (self.neighbors.r < self.config.bkg_local_annuli[1]) &
                                         (self.neighbors.refmag < maxmag) &
                                         (self.neighbors.chisq < mask['maskgals'].chisq.max()) &
                                         (self.neighbors.refmag < neighbors_depth))
        bkg_density_in_annulus = float(neighbors_in_annulus.size) / annulus_area
        bkg_local = bkg_density_in_annulus / prediction
        return bkg_local

    def calc_richness(self, mask, calc_err=True, index=None):
        if index is not None:
            idx = index
        else:
            idx = np.arange(len(self.neighbors))

        maxmag = self.mstar - 2.5 * np.log10(self.config.lval_reference)
        self.neighbors.chisq[idx] = compute_redsequence_chisq(self.zredstr, self.neighbors[idx], self._redshift)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr['ncol'])
        nfw = self._calc_radial_profile(idx=idx)
        phi = self._calc_luminosity(maxmag, idx=idx)
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self.calc_bkg_density(self.neighbors.r[idx], self.neighbors.chisq[idx],
                                        self.neighbors.refmag[idx])

        theta_i = calc_theta_i(self.neighbors.refmag[idx], self.neighbors.refmag_err[idx],
                               maxmag, self.zredstr['limmag'])

        cpars = calc_maskcorr(mask['maskgals'], self.mstar, maxmag, self.zredstr['limmag'], mask['rng'])

        try:
            w = theta_i * self.neighbors.pfree[idx]
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)

        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts,
                              self.neighbors.r[idx], w,
                              cpars=cpars, rsig=self.config.rsig)

        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()

        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0

        if lam < 0.0 or pmem.max() == 0.0:
            lam = -1.0
            lam_err = -1.0
            self.scaleval = -1.0
        else:
            bar_pmem = np.sum(pmem**2.0)/np.sum(pmem)
            cval = np.clip(np.sum(cpars * rlam**np.arange(cpars.size, dtype=float)),
                           0.0, None)

            self.scaleval = np.absolute(lam / np.sum(pmem))
            lam_unscaled = lam / self.scaleval

            if calc_err:
                lam_cerr = self.calc_lambdacerr(mask['maskgals'], self.mstar,
                                                lam, rlam, pmem, cval, self.config.dldr_gamma)
                lam_err = np.sqrt((1-bar_pmem) * lam_unscaled * self.scaleval**2. + lam_cerr**2.)

            ucounts = rho*phi
            pcol = ucounts * lam/(ucounts * lam + bcounts)
            bad, = np.where((self.neighbors.r[idx] > rlam) | (self.neighbors.refmag[idx] > maxmag) |
                            (self.neighbors.refmag[idx] > self.zredstr['limmag']) | (~np.isfinite(pcol)))
            pcol[bad] = 0.0

            self.neighbors.theta_i[idx] = theta_i
            self.neighbors.theta_r[idx] = theta_r
            self.neighbors.p[idx] = p
            self.neighbors.pcol[idx] = pcol
            self.neighbors.pmem[idx] = pmem

        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0
        return lam

    def calc_lambdacerr(self, maskgals, mstar, lam, rlam, pmem, cval, gamma):
        limmag = self.zredstr['limmag']
        use, = np.where(maskgals.r < rlam)
        mark    = maskgals.mark[use]
        refmag  = mstar + maskgals.m[use]
        cwt     = maskgals.cwt[use]
        nfw     = maskgals.nfw[use]
        lumwt   = maskgals.lumwt[use]
        chisq   = maskgals.chisq[use]
        r       = maskgals.r[use]

        logrc   = np.log(rlam)
        norm    = np.exp(1.65169 - 0.547850*logrc + 0.138202*logrc**2. -
            0.0719021*logrc**3. - 0.0158241*logrc**4.-0.000854985*logrc**5.)
        nfw     = norm*nfw
        ucounts = cwt*nfw*lumwt

        faint, = np.where(refmag >= limmag)
        refmag_for_bcounts = np.copy(refmag)
        refmag_for_bcounts[faint] = limmag-0.01

        bcounts = self.calc_bkg_density(r, chisq, refmag_for_bcounts)
        out, = np.where((refmag > limmag) | (mark == 0))

        if out.size == 0 or cval < 0.01:
            lam_err = 0.0
        else:
            p_out = lam*ucounts[out] / (lam*ucounts[out] + bcounts[out])
            varc0 = (1./lam) * (1./use.size) * np.sum(p_out)
            sigc = np.sqrt(varc0 - varc0**2.)
            k = lam**2. / np.sum(pmem**2.)
            lam_err = k*sigc/(1. - self.beta*gamma)
        return lam_err

    def calc_richness_fit(self, mask, col_index, centcolor_in=None, rcut=0.5, mingal=5, sigint=0.05, calc_err=False):
        badlam = -10.0
        s2p = np.sqrt(2. * np.pi)
        maxmag = self.mstar - 2.5 * np.log10(self.config.lval_reference)
        minmag = self.mstar - 2.5 * np.log10(20.0)
        col = self.neighbors.galcol[:, col_index]
        col_err = self.neighbors.galcol_err[:, col_index]
        from .color_background import get_colrange
        colrange = get_colrange(self.cbkg, col_index)
        guse, = np.where((self.neighbors.refmag > minmag) &
                         (self.neighbors.refmag < maxmag) &
                         (self.neighbors.r < rcut) &
                         (col > colrange[0]) &
                         (col < colrange[1]))
        if guse.size < mingal:
            self.scaleval = -1.0
            return badlam
        if centcolor_in is None:
            ind = np.argmin(self.neighbors.r)
            test, = np.where(guse == ind)
            if test.size == 0: return badlam
            centcolor = col[ind]
        else:
            centcolor = centcolor_in
        cerr = np.sqrt(col_err**2. + sigint**2.)
        in2sig, = np.where((np.abs(col[guse] - centcolor) < 2. * cerr[guse]))
        if in2sig.size < mingal:
            self.scaleval = -1.0
            return badlam
        pivot = np.median(self.neighbors.refmag[guse])
        fit = np.polyfit(self.neighbors.refmag[guse[in2sig]] - pivot,
                         col[guse[in2sig]], 1, w=1. / col_err[guse[in2sig]])
        mpivot, bpivot = fit[0], fit[1]
        d = col - (mpivot * (self.neighbors.refmag - pivot) + bpivot)
        d_err_net = np.sqrt(col_err**2. + sigint**2.)
        d_wt = (1. / (s2p * d_err_net)) * np.exp(-(d**2.) / (2. * d_err_net**2.))
        nfw = self._calc_radial_profile()
        theta_i = calc_theta_i(self.neighbors.refmag, self.neighbors.refmag_err, maxmag, self.config.limmag_catalog)
        phi = self._calc_luminosity(maxmag)
        ucounts = (2. * np.pi * self.neighbors.r) * d_wt * nfw * phi
        bcounts = self.calc_cbkg_density(self.neighbors.r, col_index, col, self.neighbors.refmag)
        cpars = calc_maskcorr(mask['maskgals'], self.mstar, maxmag, self.config.limmag_catalog, mask['rng'])
        try:
            w = theta_i * self.neighbors.pfree
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)
        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts, self.neighbors.r, w, cpars=cpars)
        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()
        bar_pmem = np.sum(pmem**2.) / np.sum(pmem)
        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0
        if lam < 0.0:
            lam_err = -1.0
            self.scaleval = -1.0
        else:
            self.scaleval = np.absolute(lam / np.sum(pmem))
            lam_unscaled = lam / self.scaleval
            cval = np.clip(np.sum(cpars * rlam**np.arange(cpars.size, dtype=float)),
                           0.0, None)
            if calc_err:
                lam_cerr = self.calc_lambdacerr(mask['maskgals'], self.mstar,
                                               lam, rlam, pmem, cval, self.config.dldr_gamma)
                lam_err = np.sqrt((1. - bar_pmem) * lam_unscaled * self.scaleval**2. + lam_cerr**2.)
            ucounts = d_wt * phi
            bcounts = (bcounts / (2. * np.pi * self.neighbors.r)) * np.pi * rlam**2.
            pcol = ucounts * lam / (ucounts * lam + bcounts)
            bad, = np.where((self.neighbors.r > rlam) | (self.neighbors.refmag > maxmag) |
                            (self.neighbors.refmag > self.config.limmag_catalog))
            pcol[bad] = 0.0
            self.neighbors.theta_i[:] = theta_i
            self.neighbors.theta_r[:] = theta_r
            self.neighbors.p[:] = p
            self.neighbors.pcol[:] = pcol
            self.neighbors.pmem[:] = pmem
        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0
        return lam

    @property
    def redshift(self):
        if self._redshift is None and self.z > 0.0:
            # This will call the setter and initialize mstar etc.
            self.redshift = self.z
        return self._redshift

    @redshift.setter
    def redshift(self, value):
        if (value < 0.0): raise ValueError("Cannot set redshift to < 0.0")
        self._redshift = np.clip(value, 0.01, None)
        self._update_mstar()
        self._update_mpc_scale()
        self._compute_neighbor_r()

    @property
    def mstar(self):
        if self._mstar is None and self.zredstr is not None:
            self._update_mstar()
        return self._mstar

    def _update_mstar(self):
        if self.zredstr is not None and self._redshift is not None:
            self._mstar = redsequence_mstar(self.zredstr, self._redshift)

    @property
    def mpc_scale(self):
        if self._mpc_scale is None:
            self._update_mpc_scale()
        return self._mpc_scale

    def _update_mpc_scale(self):
        if self._redshift is not None:
            self._mpc_scale = np.radians(1.) * self.cosmo.Da(0, self._redshift)

    def _compute_neighbor_r(self):
        if self.neighbors is not None and self._redshift is not None:
            self.neighbors.r = np.clip(self.mpc_scale * self.neighbors.dist, 1e-6, None)

    def copy(self):
        cluster = self.__copy__()
        cluster.redshift = self.redshift
        return cluster

    def __copy__(self):
        return Cluster(cat_vals=self.as_array(),
                       r0=self.r0,
                       beta=self.beta,
                       config=self.config,
                       zredstr=self.zredstr,
                       bkg=self.bkg,
                       cbkg=self.cbkg,
                       neighbors=self.neighbors)

    def calc_richness_wo_mask(self, calc_err=True, index=None):
        if index is not None:
            idx = index
        else:
            idx = np.arange(len(self.neighbors))
        maxmag = self.mstar - 2.5 * np.log10(self.config.lval_reference)
        self.neighbors.chisq[idx] = compute_redsequence_chisq(self.zredstr, self.neighbors[idx], self._redshift)
        rho = chisq_pdf(self.neighbors.chisq[idx], self.zredstr['ncol'])
        nfw = self._calc_radial_profile(idx=idx)
        phi = self._calc_luminosity(maxmag, idx=idx)
        ucounts = (2*np.pi*self.neighbors.r[idx]) * nfw * phi * rho
        bcounts = self.calc_bkg_density(self.neighbors.r[idx], self.neighbors.chisq[idx],
                                        self.neighbors.refmag[idx])
        theta_i = calc_theta_i(self.neighbors.refmag[idx], self.neighbors.refmag_err[idx],
                               maxmag, self.zredstr['limmag'])
        try:
            w = theta_i * self.neighbors.pfree[idx]
        except AttributeError:
            w = theta_i * np.ones_like(ucounts)
        richness_obj = Solver(self.r0, self.beta, ucounts, bcounts,
                              self.neighbors.r[idx], w, rsig=self.config.rsig)
        lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()
        self.neighbors.theta_i[:] = 0.0
        self.neighbors.theta_r[:] = 0.0
        self.neighbors.p[:] = 0.0
        self.neighbors.pcol[:] = 0.0
        self.neighbors.pmem[:] = 0.0
        if lam < 0.0 or pmem.max() == 0.0:
            lam = -1.0
            lam_err = -1.0
            self.scaleval = -1.0
        else:
            bar_pmem = np.sum(pmem**2.0)/np.sum(pmem)
            self.scaleval = np.absolute(lam / np.sum(pmem))
            self.lam_unscaled = lam / self.scaleval

            if calc_err:
                lam_err = np.sqrt((1-bar_pmem) * self.lam_unscaled * self.scaleval**2.)

            ucounts = rho*phi
            pcol = ucounts * lam/(ucounts * lam + bcounts)
            bad, = np.where((self.neighbors.r[idx] > rlam) | (self.neighbors.refmag[idx] > maxmag) |
                            (self.neighbors.refmag[idx] > self.zredstr['limmag']) | (~np.isfinite(pcol)))
            pcol[bad] = 0.0
            self.neighbors.theta_i[idx] = theta_i
            self.neighbors.theta_r[idx] = theta_r
            self.neighbors.p[idx] = p
            self.neighbors.pcol[idx] = pcol
            self.neighbors.pmem[idx] = pmem
        self.Lambda = lam
        self.r_lambda = rlam
        if calc_err:
            self.Lambda_e = lam_err
        else:
            self.Lambda_e = 0.0
        return lam


class ClusterCatalog(Catalog):
    """
    Class for a catalog of clusters, based on Catalog (astropy Table).
    """
    _RowClass = Cluster
    RowClass = Cluster

    def __init__(self, *args, **kwargs):
        self.r0 = kwargs.pop('r0', 1.0)
        self.beta = kwargs.pop('beta', 0.2)
        self.zredstr = kwargs.pop('zredstr', None)
        self.config = kwargs.pop('config', None)
        self.bkg = kwargs.pop('bkg', None)
        self.cbkg = kwargs.pop('cbkg', None)
        self.zredbkg = kwargs.pop('zredbkg', None)
        dtype = kwargs.pop('dtype', None)
        
        self.cosmo = getattr(self.config, 'cosmo', Cosmo())

        super().__init__(*args, **kwargs)
        self._RowClass = Cluster
        self.RowClass = Cluster

        if dtype is not None:
            cluster_dtype = dtype
        else:
            if self.config is not None:
                cluster_dtype = self.config.cluster_dtype
            else:
                cluster_dtype = cluster_dtype_base

        dtype_augment = [dt for dt in cluster_dtype if dt[0].lower() not in self.colnames]
        if len(dtype_augment) > 0:
            self.add_fields(dtype_augment)

    @classmethod
    def from_catfile(cls, filename, **kwargs):
        cat = fitsio.read(filename, ext=1, upper=True)
        return cls(cat, **kwargs)

    @classmethod
    def zeros(cls, size, **kwargs):
        dtype = kwargs.get('dtype', None)
        if dtype is not None:
            cluster_dtype = dtype
        else:
            cluster_dtype = cluster_dtype_base
        return cls(np.zeros(size, dtype=cluster_dtype), **kwargs)
