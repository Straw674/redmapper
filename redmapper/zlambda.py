"""Classes to compute cluster photometric redshifts (z_lambda) and perform corrections.
"""
import numpy as np
import scipy.optimize
import scipy.integrate
import copy
import os
import fitsio

from .utilities import gaussFunction, cubic_spline_compute_y2, cubic_spline_interpolate
from .catalog import Entry
from .logger import logger
from .redsequence import redsequence_zindex, redsequence_mstar, compute_redsequence_chisq

def compute_zlambda(cluster, mask, zin, maxmag_in=None, calcpz=False, calc_err=True):
    """
    Calculate the z_lambda cluster photometric redshift.

    This is an iterative algorithm that ensures consistency between the
    member selection, richness, and redshift likelihood.

    Parameters
    ----------
    cluster: `redmapper.Cluster`
       Cluster to compute z_lambda.
    mask: `redmapper.Mask`
       Footprint mask for survey.
    zin: `float`
       Input redshift (starting point).
    maxmag_in: `float`, optional
       Maximum magnitude to select neighbor galaxies.  Default is None,
       which uses reference luminosity cut.
    calcpz: `bool`, optional
       Calculate p(z) as well as z_lambda.  Default is False.
    calc_err: `bool`, optional
       Calculate z_lambda error.  Default is True.

    Returns
    -------
    z_lambda: `float`
       Cluster photometric redshift.
    z_lambda_err: `float`
       Error on photometric redshift.
    pzbins: `np.array`, optional
       p(z) redshift bins.
    pz: `np.array`, optional
       p(z) values.
    niter: `int`
       Number of iterations.
    """
    z_lambda = copy.copy(zin)
    config = cluster.config
    zredstr = cluster.zredstr

    # Work on a copy of the cluster for modifications
    cluster_copy = cluster.copy()

    maxmag = redsequence_mstar(zredstr, z_lambda) - 2.5*np.log10(config.lval_reference)
    if maxmag_in is not None:
        if maxmag_in.size == 1:
            maxmag = maxmag_in

    maxrad = 1.2 * cluster_copy.r0 * 3.**cluster_copy.beta

    niter = 0
    pzdone = False

    if not calc_err:
        z_lambda_e = 0.0

    pzbins = None
    pz = None
    state = {'zlambda_fail': False, 'targval': 0.0}

    for pz_iter in range(2):
        if pzdone: break

        i = 0
        while i < config.zlambda_maxiter:
            if z_lambda < 0.0:
                break

            cluster_copy.redshift = z_lambda
            in_r, = np.where(cluster_copy.neighbors.r < maxrad)

            if in_r.size < 1:
                z_lambda = -1.0
                break

            lam = cluster_copy.calc_richness(mask, calc_err=False, index=in_r)

            if lam < config.percolation_minlambda:
                z_lambda = -1.0
                break

            wtvals_mod = cluster_copy.neighbors.pcol

            if maxmag_in is not None:
               maxmag = (redsequence_mstar(zredstr, z_lambda) -
                   2.5 * np.log10(config.lval_reference))

            _zlambda_select_neighbors(state, cluster_copy, wtvals_mod, maxrad, maxmag)

            if state['zlambda_fail']:
                z_lambda = -1.0
                z_lambda_new = -1.0
                break
            else:
                z_lambda_new = _zlambda_calcz(state, cluster_copy, z_lambda)

            if (i > 0 and (np.abs(z_lambda_new-z_lambda) < config.zlambda_tol or
                z_lambda_new < 0.0)):
                break

            z_lambda = z_lambda_new
            if z_lambda < 0.0:
                break
            i += 1

        niter = i

        if z_lambda > 0.0 and calc_err:
            cluster_copy.redshift = z_lambda

            if not calcpz:
                z_lambda_e = _zlambda_calc_gaussian_err(state, cluster_copy, z_lambda)
                if z_lambda_e < 0.0:
                    z_lambda = -1.0
                    z_lambda_e = -1.0
                pzdone = True
            else:
                pzdone, z_lambda, z_lambda_e, pzbins, pz = _zlambda_calc_pz_and_check(state, cluster_copy, z_lambda, wtvals_mod, cluster_copy.r_lambda, maxmag, convergence_warning=(pz_iter > 0))
        else:
            z_lambda_e = -1.0
            if calcpz:
                pzbins = np.zeros(config.npzbins)
                pz = np.zeros_like(pzbins)
            pzdone = True

    return z_lambda, z_lambda_e, pzbins, pz, niter

def _zlambda_select_neighbors(state, cluster, wtvals, maxrad, maxmag):
    config = cluster.config
    zredstr = cluster.zredstr
    topfrac = config.zlambda_topfrac

    nzrefmag    = zredstr['refmagbins'].size
    zrefmagbin  = np.clip(np.around(nzrefmag*(cluster.neighbors.refmag -
                                              zredstr['refmagbins'][0])/
                                    (zredstr['refmagbins'][nzrefmag-2] -
                                     zredstr['refmagbins'][0])), 0, nzrefmag-1)

    ncount = topfrac*np.sum(wtvals)
    use,   = np.where((cluster.neighbors.r < maxrad) &
                      (cluster.neighbors.refmag < maxmag) &
                      (wtvals > 0.0))

    if ncount < 3:
        ncount = 3

    state['zlambda_fail'] = False
    if use.size < 3:
        state['zlambda_fail'] = True
        return

    if use.size < ncount:
        ncount = use.size

    st = np.argsort(wtvals[use])[::-1]
    pthresh = wtvals[use[st[np.int64(np.around(ncount)-1)]]]

    pw  = 1./(np.exp((pthresh-wtvals[use])/0.04)+1)
    gd, = np.where(pw > 1e-3)

    state['in_rad'] = use[gd]
    state['pw'] = pw[gd]

def _zlambda_calcz(state, cluster, z_lambda):
    config = cluster.config
    nsteps = 10
    steps = config.zlambda_parab_step * np.arange(nsteps) + z_lambda - config.zlambda_parab_step * (nsteps - 1) / 2
    likes = np.zeros(nsteps)
    for i in range(nsteps):
         likes[i] = _zlambda_bracket_fn(state, cluster, steps[i])
    fit = np.polyfit(steps, likes, 2)

    if fit[0] > 0.0:
        z_lambda = -fit[1]/(2.0 * fit[0])
    else:
        z_lambda = -1.0

    z_lambda = np.clip(z_lambda, steps[0] - config.zlambda_parab_step,
                       steps[-1] + config.zlambda_parab_step)

    return z_lambda

def _zlambda_bracket_fn(state, cluster, z):
    zredstr = cluster.zredstr
    likelihoods = compute_redsequence_chisq(zredstr, cluster.neighbors[state['in_rad']],
                                               z, calc_lkhd=True)
    t = -np.sum(state['pw']*likelihoods)
    return t

def _zlambda_delta_bracket_fn(z, state, cluster):
    t  = _zlambda_bracket_fn(state, cluster, z)
    dt = np.abs(t-state['targval'])
    return dt

def _zlambda_calc_gaussian_err(state, cluster, z_lambda):
    minlike = _zlambda_bracket_fn(state, cluster, z_lambda)
    state['targval'] = minlike+1

    z_lambda_lo = scipy.optimize.minimize_scalar(_zlambda_delta_bracket_fn,
        args=(state, cluster),
        bracket = (z_lambda-0.1, z_lambda-0.001), method='bounded',
        bounds = (z_lambda-0.1, z_lambda-0.001))
    z_lambda_hi = scipy.optimize.minimize_scalar(_zlambda_delta_bracket_fn,
        args=(state, cluster),
        bracket = (z_lambda+0.001, z_lambda+0.1), method='bounded',
        bounds = (z_lambda+0.001, z_lambda+0.1))
    z_lambda_e = (z_lambda_hi.x-z_lambda_lo.x)/2.

    return z_lambda_e

def _zlambda_calc_pz_and_check(state, cluster, z_lambda, wtvals, maxrad, maxmag, convergence_warning=False):
    config = cluster.config
    zredstr = cluster.zredstr
    
    pzbins, pz = _zlambda_calc_pz(state, cluster, z_lambda, wtvals, maxrad, maxmag, slow=False)

    pzdone = False

    if (pz[(config.npzbins - 1) // 2] > 0.0 and
        ((pz[0] / pz[(config.npzbins - 1) // 2] > 0.01) and
         (pzbins[0] >= (zredstr['z'][0] + 0.01))) or
        ((pz[-1] >= pz[(config.npzbins-1) // 2] > 0.01) and
         (pzbins[-1] <= (zredstr['z'][-2] - 0.01)))):

        pzbins, pz = _zlambda_calc_pz(state, cluster, z_lambda, wtvals, maxrad, maxmag, slow=True)

    if pz[0] < 0:
        z_lambda = -1.0
        z_lambda_e = -1.0
    else:
        m = np.argmax(pz)
        p0 = np.array([pz[m], pzbins[m], 0.01])

        try:
            coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction,
                                                        pzbins,
                                                        pz,
                                                        p0=p0)
        except:
            coeff = [-10.0, -10.0, -10.0]

        if coeff[2] > 0 or coeff[2] > 0.2:
            z_lambda_e = coeff[2]
        else:
            z_lambda_e = _zlambda_calc_gaussian_err(state, cluster, z_lambda)

        pmind = np.argmax(pz)
        if (np.abs(pzbins[pmind] - z_lambda) < config.zlambda_tol):
            pzdone = True
        else:
            if (convergence_warning):
                logger.info('Warning: z_lambda / p(z) inconsistency detected.')

            z_lambda = pzbins[pmind]
            pzdone = False

    return pzdone, z_lambda, z_lambda_e, pzbins, pz

def _zlambda_calc_pz(state, cluster, z_lambda, wtvals, maxrad, maxmag, slow=False):
    config = cluster.config
    zredstr = cluster.zredstr
    
    minlike = _zlambda_bracket_fn(state, cluster, z_lambda)
    state['targval']=minlike+16

    if not slow:
        z_lambda_hi = scipy.optimize.minimize_scalar(_zlambda_delta_bracket_fn,
                                                     args=(state, cluster),
                                                     bracket=(z_lambda + 0.001, z_lambda + 0.15),
                                                     method='bounded',
                                                     bounds=(z_lambda + 0.001, z_lambda + 0.15),
                                                     options={'xatol':1e-5})

        dz = np.clip((z_lambda_hi.x - z_lambda), 0.005, 0.15)
        pzbinsize = 2.*dz/(config.npzbins-1)
        pzbins = pzbinsize*np.arange(config.npzbins)+z_lambda - dz

    else:
        pk = -_zlambda_bracket_fn(state, cluster, z_lambda)
        pz0 = zredstr['volume_factor'][redsequence_zindex(zredstr, z_lambda)]

        dztest = 0.05
        lowz = z_lambda - dztest
        ratio = 1.0
        while (lowz >= zredstr['z'][0] and (ratio > 0.01)):
            val = -_zlambda_bracket_fn(state, cluster, lowz)
            with np.errstate(over="raise"):
                pz_val = np.exp(val - pk) * zredstr['volume_factor'][redsequence_zindex(zredstr, lowz)]
            ratio = pz_val/pz0
            if (ratio > 0.01):
                lowz -= dztest

        lowz = np.clip(lowz, zredstr['z'][0], None)
        highz = z_lambda + dztest
        ratio = 1.0
        while (highz <= zredstr['z'][-2] and ratio > 0.01):
            val = -_zlambda_bracket_fn(state, cluster, highz)
            ln_lkhd = val - pk
            pz_val = np.exp(ln_lkhd) * zredstr['volume_factor'][redsequence_zindex(zredstr, highz)]
            ratio = pz_val / pz0
            if ratio > 0.01:
                highz += dztest

        highz = np.clip(highz, None, zredstr['z'][-2])
        pzbinsize = (highz - lowz)/(config.npzbins-1)
        pzbins = pzbinsize*np.arange(config.npzbins) + lowz
        zmind = np.argmin(np.abs(pzbins - z_lambda))
        pzbins = pzbins - (pzbins[zmind] - z_lambda)

    ln_lkhd = np.zeros(config.npzbins)
    for i in range(config.npzbins):
        ln_lkhd[i] = -_zlambda_bracket_fn(state, cluster, pzbins[i])

    ln_lkhd = ln_lkhd - np.max(ln_lkhd)
    pz = np.exp(ln_lkhd) * zredstr['volume_factor'][redsequence_zindex(zredstr, pzbins)]

    n = scipy.integrate.simpson(y=pz, x=pzbins)
    pz = pz / n

    return pzbins, pz


def read_zlambda_correction(parfile=None, pars=None, zrange=None, zbinsize=None, zlambda_pivot=None):
    """
    Read z_lambda correction parameters.

    Must specify at least one of parfile (parameter file) or pars
    (`redmapper.Entry` describing the parameters).

    Parameters
    ----------
    parfile: `str`, optional
       z_lambda correction parameters file.  Default is None.
    pars: `redmapper.Entry`, optional
       z_lambda correction parameters.  Default is None.
    zrange: array_like, optional
       Redshift range.  Default is None.  Use header info if parfile,
       must be specified if pars are input.
    zbinsize: `float`, optional
       Redshift bin size.  Default is None.  Use header info if parfile,
       must be specified if pars are input.
    zlambda_pivot: `float`, optional
       Pivot richness for correction terms.  Default is None.  Use
       header info if parfile, must be specified if pars are input.

    Returns
    -------
    zlambda_corr_data: `dict`
        Dictionary containing correction parameters.
    """
    if parfile is None and pars is None:
        raise RuntimeError("Must supply either parfile or pars")

    if parfile is not None:
        if not os.path.isfile(parfile):
            raise IOError("Could not find zlambda correction file %s" % (parfile))

        hdr = fitsio.read_header(parfile, ext=1)

        if zrange is None:
            zrange = [hdr['ZRANGE0'], hdr['ZRANGE1']]
        if zbinsize is None:
            zbinsize = hdr['ZBINSIZE']
        if zlambda_pivot is None:
            zlambda_pivot = hdr['ZLAMPIV']

        pars = Entry.from_fits_file(parfile, ext=1)
    else:
        if zrange is None:
            raise ValueError("Must specify zrange with a par structure")
        if zbinsize is None:
            raise ValueError("Must specify zbinsize with a par structure")
        if zlambda_pivot is None:
            raise ValueError("Must specify zlambda_pivot with a par structure")

    nbins = np.round((zrange[1] - zrange[0])/zbinsize).astype(np.int32)
    z = zbinsize*np.arange(nbins) + zrange[0]

    niter = 1
    try:
        niter = pars.niter_true
    except:
        pass

    extrapolated = np.zeros_like(z, dtype=bool)

    offset = np.zeros((niter, nbins))
    slope = np.zeros_like(offset)
    scatter = np.zeros_like(offset)

    loz, = np.where(z < pars.offset_z[0])
    hiz, = np.where(z > pars.offset_z[-1])

    extrapolated[loz] = True
    extrapolated[hiz] = True

    if niter == 1:
        y2 = cubic_spline_compute_y2(pars.offset_z, pars.offset_true)
        offset[0, :] = cubic_spline_interpolate(z, pars.offset_z, pars.offset_true, y2)

        y2 = cubic_spline_compute_y2(pars.slope_z, pars.slope_true)
        slope[0, :] = cubic_spline_interpolate(z, pars.slope_z, pars.slope_true, y2)

        y2 = cubic_spline_compute_y2(pars.slope_z, pars.scatter_true)
        scatter[0, :] = np.clip(cubic_spline_interpolate(z, pars.slope_z, pars.scatter_true, y2), 0.001, None)
    else:
        for i in range(niter):
            y2 = cubic_spline_compute_y2(pars.offset_z, pars.offset_true[:, i])
            offset[i, :] = cubic_spline_interpolate(z, pars.offset_z, pars.offset_true[:, i], y2)

            y2 = cubic_spline_compute_y2(pars.slope_z, pars.slope_true[:, i])
            slope[i, :] = cubic_spline_interpolate(z, pars.slope_z, pars.slope_true[:, i], y2)

            y2 = cubic_spline_compute_y2(pars.slope_z, pars.scatter_true[:, i])
            scatter[i, :] = np.clip(cubic_spline_interpolate(z, pars.slope_z, pars.scatter_true[:, i], y2), 0.001, None)

    y2 = cubic_spline_compute_y2(pars.offset_z, pars.zred_uncorr)
    zred_uncorr = cubic_spline_interpolate(z, pars.offset_z, pars.zred_uncorr, y2)

    return {
        'zrange': zrange,
        'zbinsize': zbinsize,
        'zlambda_pivot': zlambda_pivot,
        'z': z,
        'niter': niter,
        'extrapolated': extrapolated,
        'offset': offset,
        'slope': slope,
        'scatter': scatter,
        'zred_uncorr': zred_uncorr
    }

def apply_zlambda_correction(zlambda_corr_data, lam, zlam, zlam_e, pzbins=None, pzvals=None, noerr=False):
    """
    Apply the z_lambda correction to an input z_lambda.

    Parameters
    ----------
    zlambda_corr_data: `dict`
        Dictionary containing correction parameters.
    lam: `float`
       Richness of cluster to compute correction.
    zlam: `float`
       Input z_lambda.
    zlam_e: `float`
       Input z_lambda error.
    pzbins: `np.array`, optional
       Input p(z) redshift bins.  Default is None (no p(z)).
    pzvals: `np.array`, optional
       Input p(z) values.  Default is None (no p(z)).
    noerr: `bool`, optional
       Do not apply any error correction.  Default is False.

    Returns
    -------
    zlam_corr: `float`
       Output corrected z_lambda.
    zlam_e_corr: `float`
       Output corrected z_lambda error.
    pzbins_corr: `np.array`, optional
       Corrected p(z) redshift bins.
    pzvals_corr: `np.array`, optional
       Corrected p(z) values.
    """
    if pzbins is None:
        npzbins = 0
    else:
        npzbins = pzbins.size

    zlam_corr = copy.copy(zlam)
    zlam_e_corr = copy.copy(zlam_e)
    if pzbins is not None:
        pzbins_corr = copy.copy(pzbins)
        pzvals_corr = copy.copy(pzvals)
    else:
        pzbins_corr = None
        pzvals_corr = None

    for i in range(zlambda_corr_data['niter']):
        correction = zlambda_corr_data['offset'][i, :] + zlambda_corr_data['slope'][i, :] * np.log(lam / zlambda_corr_data['zlambda_pivot'])
        extra_err = np.interp(zlam_corr, zlambda_corr_data['z'], zlambda_corr_data['scatter'][i, :])

        dz = np.interp(zlam_corr, zlambda_corr_data['z'], correction)

        ozlam = copy.deepcopy(zlam_corr)

        zlam_corr += dz

        if pzbins_corr is None and not noerr:
            zlam_e_corr = np.sqrt(zlam_e_corr**2. + extra_err**2.)
        elif pzbins_corr is not None:
            offset = pzbins_corr[int((float(npzbins) - 1)/2.)] - ozlam

            opdz = pzbins_corr[1] - pzbins_corr[0]
            pdz = opdz * np.sqrt(extra_err**2. + zlam_e_corr**2.) / zlam_e_corr

            pzbins_corr = pdz * np.arange(npzbins) + zlam_corr - pdz * (npzbins - 1)/2. + offset

            n = scipy.integrate.simpson(y=pzvals_corr, x=pzbins_corr)
            pzvals_corr /= n

            zlam_e_corr = np.sqrt(zlam_e_corr**2. + extra_err**2.)

    if pzbins is None:
        return zlam_corr, zlam_e_corr
    else:
        return zlam_corr, zlam_e_corr, pzbins_corr, pzvals_corr


