import numpy as np
from ..solver_nfw import Solver
from ..mask import calc_maskcorr
from ..utilities import chisq_pdf, calc_theta_i, nfw_pdf, schechter_pdf
from ..redsequence import (redsequence_zindex, redsequence_lumrefmagindex,
                           redsequence_mstar, compute_redsequence_chisq)

def calc_radial_profile(r, rscale=0.15):
    """Pure function for computing radial profile weights."""
    return nfw_pdf(r, rscale=rscale)

def calc_luminosity_profile(refmag, redshift, redsequence_data, normmag):
    """Pure function to compute luminosity filter."""
    zind = redsequence_zindex(redsequence_data, redshift)
    refind = redsequence_lumrefmagindex(redsequence_data, normmag)
    normalization = redsequence_data['lumnorm'][refind, zind]
    mstar = redsequence_mstar(redsequence_data, redshift)
    phi = schechter_pdf(refmag, alpha=redsequence_data['alpha'], mstar=mstar)
    return phi / normalization

def calc_richness(
    cluster_data,
    neighbors,
    redshift,
    mask,
    redsequence_data,
    bkg_model,
    config,
    calc_err=True
):
    """
    Pure function to calculate the richness for a cluster.

    Parameters
    ----------
    cluster_data: structured array
        Data for a single cluster (or a row of a catalog).
    neighbors: structured array
        Catalog of neighbor galaxies.
    redshift: float
        Redshift of the cluster.
    mask: dict
        Footprint mask data dictionary.
    redsequence_data: dict
        Red-sequence model data dictionary from read_redsequence().
    bkg_model: dict
        Background model data dictionary.
    config: dict-like
        Configuration parameters.
    calc_err: bool, optional
        Whether to calculate the richness error.

    Returns
    -------
    result: dict
        Dictionary containing lam, lam_e, r_lambda, and updated neighbor probabilities.
    """
    lval_reference = config.get('lval_reference', 0.2)
    rsig = config.get('rsig', 0.05)
    r0 = config.get('r0', 1.0)
    beta = config.get('beta', 0.2)

    mstar = redsequence_mstar(redsequence_data, redshift)
    maxmag = mstar - 2.5 * np.log10(lval_reference)

    neighbor_chisq = compute_redsequence_chisq(redsequence_data, neighbors, redshift)
    rho = chisq_pdf(neighbor_chisq, redsequence_data['ncol'])

    nfw = calc_radial_profile(neighbors['r'])
    phi = calc_luminosity_profile(neighbors['refmag'], redshift, redsequence_data, maxmag)

    ucounts = (2 * np.pi * neighbors['r']) * nfw * phi * rho

    from ..background import compute_background
    sigma_g = compute_background(bkg_model, np.full(len(neighbors), redshift), neighbor_chisq, neighbors['refmag'])
    mpc_scale = cluster_data['mpc_scale'] if 'mpc_scale' in cluster_data.dtype.names else 1.0
    bcounts = 2. * np.pi * neighbors['r'] * (sigma_g / mpc_scale**2.)

    theta_i = calc_theta_i(neighbors['refmag'], neighbors['refmag_err'], maxmag, redsequence_data['limmag'])
    cpars = calc_maskcorr(mask['maskgals'], mstar, maxmag, redsequence_data['limmag'], mask['rng'])

    pfree = neighbors['pfree'] if 'pfree' in neighbors.dtype.names else np.ones(len(neighbors))
    w = theta_i * pfree

    richness_obj = Solver(r0, beta, ucounts, bcounts, neighbors['r'], w, cpars=cpars, rsig=rsig)
    lam, p, pmem, rlam, theta_r = richness_obj.solve_nfw()

    res = {
        'lam': lam,
        'rlam': rlam,
        'p': p,
        'pmem': pmem,
        'theta_r': theta_r,
        'theta_i': theta_i,
    }

    if lam < 0.0 or pmem.max() == 0.0:
        res['lam'] = -1.0
        res['lam_e'] = -1.0
    else:
        if calc_err:
            pass
        else:
            res['lam_e'] = 0.0

    return res
