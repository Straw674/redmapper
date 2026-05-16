"""Function to run the second (likelihood) pass through a catalog for cluster finding.
"""
import fitsio
import numpy as np
import esutil
import copy
import warnings

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import read_background
from .mask import get_mask_values
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .zlambda import compute_zlambda, read_zlambda_correction, apply_zlambda_correction
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .utilities import chisq_pdf, interpol
from .redsequence import redsequence_mstar
from .configuration import Configuration
from .logger import logger

def _likelihoods_more_setup(state, cleaninput=False, keepz=False, **kwargs):
    config = state['config']
    logger.info("%s: Likelihoods using catfile: %s" % (state['hpix_logstr'], config.catfile))

    cat = ClusterCatalog.from_catfile(config.catfile, zredstr=state['zredstr'], config=config,
                                           bkg=state['bkg'], cosmo=state['cosmo'],
                                           r0=state['r0'], beta=state['beta'])
    
    zrange = copy.copy(config.zrange)
    if keepz:
        zrange[0] -= config.calib_zrange_cushion
        zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
        zrange[1] += config.calib_zrange_cushion

    use, = np.where((cat.z > zrange[0]) & (cat.z < zrange[1]) & (cat.Lambda > 0.0))

    if use.size == 0:
        logger.info("No usable inputs for likelihood on pixel %s" % (state['hpix_logstr']))
        state['cat'] = None
        return state, False

    mstar = redsequence_mstar(state['zredstr'], cat.z[use])
    mlim = mstar - 2.5 * np.log10(state['limlum'])

    good, = np.where(cat.refmag[use] < mlim)
    if good.size == 0:
        logger.info("No good inputs for likelihood on pixel %s" % (state['hpix_logstr']))
        state['cat'] = None
        return state, False

    cat = cat[use[good]]

    if cleaninput:
        catmask = get_mask_values(state['mask']['mask_data'], cat.ra, cat.dec, rng=state['mask']['rng'], config=state['mask']['config'])
        cat = cat[catmask]

        if cat.size == 0:
            logger.info("No input cluster positions are in the mask on pixel %s" % (state['hpix_logstr']))
            state['cat'] = None
            return state, False
    
    state['cat'] = cat
    return state, True

def _likelihoods_process_cluster(cluster, state):
    bad = False
    config = state['config']
    
    maxmag = cluster.mstar - 2.5*np.log10(state['limlum'])

    lam = cluster.calc_richness(state['mask'])

    minrind = np.argmin(cluster.neighbors.r)
    incut, = np.where((cluster.neighbors.pmem > 0.0) &
                      (cluster.neighbors.r > cluster.neighbors.r[minrind]))

    if cluster.Lambda < config.likelihoods_minlambda or incut.size < 3:
        cluster.lnlamlike = -1e11
        cluster.lnbcglike = -1e11
        cluster.lnlike = -1e11
        return True

    cluster.lnlamlike = (-cluster.Lambda / cluster.scaleval -
                          np.sum(np.log(1.0 - cluster.neighbors.pmem[incut])))

    if config.lnw_cen_sigma <= 0.0:
        cluster.lnbcglike = 0.0
    else:
        mbar = (cluster.mstar + config.wcen_Delta0 +
                config.wcen_Delta1 * np.log(cluster.Lambda / config.wcen_pivot))
        phi_cen = ((1. / (np.sqrt(2. * np.pi) * config.wcen_sigma_m)) *
                   np.exp(-0.5 * (cluster.neighbors.refmag[minrind] - mbar)**2. / config.wcen_sigma_m**2.))

        if config.likelihoods_use_zred:
            if state['zlambda_corr'] is not None:
                zrmod = interpol(state['zlambda_corr']['zred_uncorr'], state['zlambda_corr']['z'], cluster.z_lambda)
            else:
                zrmod = cluster.z_lambda

            g = ((1./(np.sqrt(2. * np.pi) * cluster.neighbors.zred_e[minrind])) *
                 np.exp(-0.5 * (cluster.neighbors.zred[minrind] - zrmod)**2. / cluster.neighbors.zred_e[minrind]**2.))
        else:
            g = chisq_pdf(cluster.neighbors.chisq[minrind], state['zredstr']['ncol'])

        lum = 10.**((cluster.mstar - cluster.neighbors.refmag) / 2.5)
        u, = np.where((cluster.neighbors.r > 1e-5) & (cluster.neighbors.pmem > 0.0))
        w = np.log(np.sum(cluster.neighbors.pmem[u] * lum[u] /
                          np.sqrt(cluster.neighbors.r[u]**2. + config.wcen_rsoft**2.)) / ((1. / cluster.r_lambda) * np.sum(cluster.neighbors.pmem[u] * lum[u])))
        sig = config.lnw_cen_sigma / np.sqrt(((np.clip(cluster.Lambda, None, config.wcen_maxlambda)) / cluster.scaleval) / config.wcen_pivot)
        fw = (1. / (np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - config.lnw_cen_mean)**2. / (sig**2.))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cluster.lnbcglike = np.log(phi_cen * np.clip(g, 1e-10, None) * fw)

        cluster.lnlike = cluster.lnbcglike + cluster.lnlamlike

    return False

def _likelihoods_postprocess(cat, members, state):
    use, = np.where(cat.lnlamlike > -1e11)
    cat = ClusterCatalog(cat._ndarray[use])
    if members is not None:
        a, b = esutil.numpy_util.match(cat.mem_match_id, members.mem_match_id)
        members = Catalog(members._ndarray[b])
    return cat, members

def run_likelihoods(conf, keepz=False, cleaninput=False):
    """
    Run likelihoods on a catalog.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    read_zreds = config.likelihoods_use_zred
    zreds_required = config.likelihoods_use_zred

    cat, members = run_cluster_pipeline(
        config,
        runmode='likelihoods',
        filetype='like',
        more_setup_fn=_likelihoods_more_setup,
        process_cluster_fn=_likelihoods_process_cluster,
        postprocess_fn=_likelihoods_postprocess,
        read_gals=True,
        read_zreds=read_zreds,
        zreds_required=zreds_required,
        cutgals_bkgrange=True,
        cutgals_chisqmax=True,
        keepz=keepz,
        cleaninput=cleaninput,
        do_lam_plusminus=False,
        match_centers_to_galaxies=False,
        record_members=False,
        do_correct_zlambda=False,
        do_pz=False,
        min_lambda=config.likelihoods_minlambda
    )


    return cat, members


