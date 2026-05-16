"""Function to run the final (percolation) pass through a catalog for cluster finding.
"""
from functools import reduce

import fitsio
import numpy as np
import esutil
import copy
import sys

from .cluster import ClusterCatalog
from .background import read_background
from .mask import get_mask_values, compute_maskgals_mark
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from . import depthmap
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .redsequence import redsequence_mstar
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .depth_fitting import apply_depthlim
from .centering import CENTERING_FUNCS
from .configuration import Configuration
from .logger import logger

def _percolation_more_setup(state, cleaninput=False, keepz=False, keepid=False, specseed=False, **kwargs):
    config = state['config']
    logger.info("%s: Percolation using catfile: %s" % (state['hpix_logstr'], config.catfile))

    cat = ClusterCatalog.from_catfile(config.catfile, zredstr=state['zredstr'], config=config,
                                           bkg=state['bkg'], zredbkg=state['zredbkg'], cosmo=state['cosmo'],
                                           r0=state['r0'], beta=state['beta'])
    
    zrange = copy.copy(config.zrange)
    if keepz:
        zrange[0] -= config.calib_zrange_cushion
        zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
        zrange[1] += config.calib_zrange_cushion

    use, = np.where((cat.z > zrange[0]) &
                    (cat.z < zrange[1]) &
                    (np.isfinite(cat.lnlike)) &
                    (cat.Lambda > config.percolation_minlambda))

    if use.size == 0:
        state['cat'] = None
        logger.info("No usable inputs for percolation on pixel %s" % (state['hpix_logstr']))
        return state, False

    mstar = redsequence_mstar(state['zredstr'], cat.z[use])
    mlim = mstar - 2.5 * np.log10(state['limlum'])

    good, = np.where(cat.refmag[use] < mlim)

    if good.size == 0:
        state['cat'] = None
        logger.info("No good inputs for percolation on pixel %s" % (state['hpix_logstr']))
        return state, False

    use = use[good]

    if keepid:
        st = np.argsort(cat.mem_match_id[use])
    else:
        st = np.argsort(cat.lnlike[use])[::-1]
        cat.mem_match_id[:] = 0

    cat = cat[use[st]]

    if cleaninput:
        catmask = get_mask_values(state['mask']['mask_data'], cat.ra, cat.dec, rng=state['mask']['rng'], config=state['mask']['config'])
        cat = cat[catmask]

        if cat.size == 0:
            state['cat'] = None
            logger.info("No input cluster positions are in the mask on pixel %s" % (state['hpix_logstr']))
            return state, False

    generate_mem_match_ids(cat)

    cat.ra_orig = cat.ra
    cat.dec_orig = cat.dec

    state['cat'] = cat
    state['keepz'] = keepz
    state['specseed'] = specseed
    state['maxiter'] = config.percolation_niter
    
    return state, True

def _percolation_process_cluster(cluster, state):
    bad = False
    done = False
    config = state['config']

    minind = np.argmin(cluster.neighbors.r)
    if state['specseed']:
        specind = minind

    if cluster.neighbors.pfree[minind] < config.percolation_pbcg_cut:
        bad = True
        reset_bad_values(cluster)
        return bad

    lc, = np.where(cluster.neighbors.r < 2.05 * state['r0'] * (cluster.Lambda/100.)**state['beta'])
    if lc.size < 2:
        bad = True
        reset_bad_values(cluster)
        return bad

    lam = cluster.calc_richness(state['mask'], index=lc, calc_err=False)

    incut, = np.where((cluster.neighbors.pmem > 0.0) &
                      (cluster.neighbors.r > np.min(cluster.neighbors.r)))

    if ((cluster.Lambda/cluster.scaleval < config.percolation_minlambda) or
        (incut.size < 3)):
        bad = True
        reset_bad_values(cluster)
        return bad

    if not state['keepz']:
        z_lambda, z_lambda_e, _, _, _ = compute_zlambda(cluster, state['mask'], cluster.redshift, calc_err=False, calcpz=False)
        cluster.z_lambda = z_lambda
        cluster.z_lambda_err = z_lambda_e

        if z_lambda < 0.0:
            bad = True
            reset_bad_values(cluster)
            return bad

        cluster.redshift = z_lambda

    cent = CENTERING_FUNCS[config.centerclass](cluster, config, rng=state['mask']['rng'], zlambda_corr=state['zlambda_corr'])
    if not cent['success'] or cent['ngood']==0:
        bad = True
        reset_bad_values(cluster)
        return bad

    cluster.ra = cent['ra'][0]
    cluster.dec = cent['dec'][0]
    cluster.update_neighbors_dist()

    if cent['index'][0] >= 0:
        cluster.mag[:] = cluster.neighbors.mag[cent['index'][0], :]
        cluster.mag_err[:] = cluster.neighbors.mag_err[cent['index'][0], :]
        cluster.refmag = cluster.neighbors.refmag[cent['index'][0]]
        cluster.refmag_err = cluster.neighbors.refmag_err[cent['index'][0]]
        cluster.ebv_mean = cluster.neighbors.ebv[cent['index'][0]]
        if state['did_read_zreds']:
            cluster.zred = cluster.neighbors.zred[cent['index'][0]]
            cluster.zred_e = cluster.neighbors.zred_e[cent['index'][0]]
            cluster.zred_chisq = cluster.neighbors.zred_chisq[cent['index'][0]]

        cluster.id_cent[:] = cluster.neighbors.id[cent['index']]
        cluster.neighbors.centering_cand[cent['index']] = 1

    cluster.ncent_good = cent['ngood']
    cluster.ra_cent[:] = cent['ra']
    cluster.dec_cent[:] = cent['dec']
    cluster.p_cen[:] = cent['p_cen']
    cluster.q_cen[:] = cent['q_cen']
    cluster.p_fg[:] = cent['p_fg']
    cluster.q_miss = cent['q_miss']
    cluster.p_sat[:] = cent['p_sat']
    cluster.p_c[:] = cent['p_c']

    for i in range(state['maxiter']):
        if cluster.redshift < 0.0:
            bad = True
        if bad:
            reset_bad_values(cluster)
            return bad

        if i == 0:
            state['mask']['maskgals'].mark = compute_maskgals_mark(state['mask']['mask_data'], cluster, state['mask']['maskgals'], rng=state['mask']['rng'], config=config)

            if state['depthstr'] is None:
                apply_depthlim(state['mask']['maskgals'],
                               cluster.neighbors.refmag, cluster.neighbors.refmag_err,
                               state['depthlim_pars'])
            else:
                depthmap.compute_maskdepth(state['depthstr'], state['mask']['maskgals'],
                                           cluster.ra, cluster.dec, cluster.mpc_scale)

        rmask = state['rmask_0'] * (cluster.Lambda/100.)**state['rmask_beta'] * ((1. + cluster.redshift) / (1. + state['rmask_zpivot']))**state['rmask_gamma']

        if rmask < cluster.r_lambda:
            rmask = cluster.r_lambda

        if i == (config.percolation_niter - 1):
            lc, = np.where(cluster.neighbors.r < 1.1 * rmask)
        else:
            lc, = np.where(cluster.neighbors.r < 2.0 * cluster.r_lambda)

        if lc.size < 2:
            bad = True
            continue

        lam = cluster.calc_richness(state['mask'])

        if (((cluster.Lambda/cluster.scaleval) < config.percolation_minlambda) or
            (cluster.neighbors.pfree[cent['maxind']] < config.percolation_pbcg_cut)):
            bad = True
            continue

        if i == 0:
            z_lambda, z_lambda_e, pzbins, pz, _ = compute_zlambda(cluster, state['mask'], cluster.redshift,
                                                               calc_err=True, calcpz=True)
            cluster.z_lambda = z_lambda
            cluster.z_lambda_err = z_lambda_e
            cluster.pzbins[:] = pzbins
            cluster.pz[:] = pz

            if not state['keepz'] and z_lambda > 0.0:
                cluster.redshift = z_lambda

        if cluster.z_lambda < 0.0:
            bad = True
            continue

    if bad:
        reset_bad_values(cluster)
        return bad

    minind = np.argmin(cluster.neighbors.r)
    u, = np.where((cluster.neighbors.r > cluster.neighbors.r[minind]) &
                  (cluster.neighbors.r < cluster.r_lambda) &
                  (cluster.neighbors.p > 0.0))
    if u.size == 0:
        reset_bad_values(cluster)
        return bad

    lum = 10.**((cluster.mstar - cluster.neighbors.refmag[u]) / 2.5)
    if config.wcen_uselum:
        cluster.w = np.log(np.sum(cluster.neighbors.p[u] * lum / np.sqrt(cluster.neighbors.r[u]**2. + config.wcen_rsoft**2.)) / ((1./cluster.r_lambda) * np.sum(cluster.neighbors.p[u] * lum)))
    else:
        cluster.w = np.log(np.sum(cluster.neighbors.p[u] / np.sqrt(cluster.neighbors.r[u]**2. + config.wcen_rsoft**2.)) / ((1./cluster.r_lambda) * np.sum(cluster.neighbors.p[u])))

    cluster.lambda_cent[0] = cluster.Lambda
    cluster.zlambda_cent[0] = cluster.z_lambda
    if cluster.ncent_good > 1:
        for ce in range(1,cluster.ncent_good):
            cluster_temp = cluster.copy()
            cluster_temp.ra = cluster.ra_cent[ce]
            cluster_temp.dec = cluster.dec_cent[ce]
            cluster_temp.update_neighbors_dist()

            clc, = np.where(cluster_temp.neighbors.r < 1.5*cluster.r_lambda)
            lam = cluster_temp.calc_richness(state['mask'], calc_err=False, index=clc)
            cluster.lambda_cent[ce] = lam

            if ce == 1:
                z_lambda, _, _, _, _ = compute_zlambda(cluster_temp, state['mask'], cluster.redshift, calc_err=False, calcpz=False)
                cluster.zlambda_cent[ce] = z_lambda

        cluster.lambda_c = np.sum(cluster.p_cen * cluster.lambda_cent)
        cluster.lambda_ce = np.sqrt(np.clip(np.sum(cluster.p_cen * cluster.lambda_cent**2.) - cluster.lambda_c**2., 0.0, None))
    else:
        cluster.lambda_c = cluster.Lambda
        cluster.lambda_ce = 0.0

    return bad

def run_percolation(conf, keepz=False, keepid=False, specseed=False, cleaninput=False):
    """
    Run percolation on a catalog.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    use_memradius = False
    if config.percolation_memradius is not None and config.percolation_memradius > 1.0:
        use_memradius = True
        
    use_memlum = False
    limlum = np.clip(config.lval_reference - 0.1, 0.01, None)
    if config.lval_reference > 0.1:
        limlum = limlum if limlum > 0.1 else 0.1

    if config.percolation_memlum is not None and config.percolation_memlum > 0.0 and config.percolation_memlum < config.lval_reference:
        if config.percolation_memlum < limlum:
            limlum = config.percolation_memlum
        use_memlum = True

    if config.percolation_lmask > 0.0:
        if config.percolation_lmask < limlum:
            limlum = config.percolation_lmask

    cat, members = run_cluster_pipeline(
        config,
        runmode='percolation',
        filetype='final',
        more_setup_fn=_percolation_more_setup,
        process_cluster_fn=_percolation_process_cluster,
        read_gals=True,
        read_zreds=True,
        zreds_required=True,
        zredbkg_required=True,
        cutgals_bkgrange=True,
        cutgals_chisqmax=False,
        keepz=keepz,
        keepid=keepid,
        specseed=specseed,
        cleaninput=cleaninput,
        do_percolation_masking=True,
        do_lam_plusminus=True,
        record_members=True,
        do_correct_zlambda=True,
        do_pz=True,
        use_memradius=use_memradius,
        use_memlum=use_memlum,
        min_lambda=config.percolation_minlambda
    )


    return cat, members


