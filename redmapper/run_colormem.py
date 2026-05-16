"""Function to compute richnesses on a catalog by fitting a linear red-sequence
model, for use in the first part of training before a red-sequence model has
been found.

"""
import fitsio
import numpy as np
import esutil
import copy
import sys
import matplotlib.pyplot as plt

from .cluster import ClusterCatalog
from .color_background import read_color_background
from .mask import get_mask
from .galaxy import GalaxyCatalog
from .catalog import Catalog, Entry
from .cluster import Cluster
from .cluster_runner import run_cluster_pipeline, output_cluster_catalog, generate_mem_match_ids, reset_bad_values
from .utilities import cubic_spline_compute_y2, cubic_spline_interpolate
from .configuration import Configuration
from .logger import logger

def _colormem_more_setup(state, **kwargs):
    config = state['config']
    state['rmask_0'] = config.calib_colormem_r0
    state['rmask_beta'] = config.calib_colormem_beta

    cat = ClusterCatalog.from_catfile(config.redgalfile,
                                           zredstr=state['zredstr'],
                                           config=config,
                                           cbkg=state['cbkg'],
                                           cosmo=state['cosmo'],
                                           r0=state['r0'],
                                           beta=state['beta'])

    use, = np.where((cat.z > config.zrange[0]) &
                    (cat.z < config.zrange[1]))
    cat = cat[use]

    cat.z_init = cat.z

    cat.add_fields([('redcolor', 'f4', config.nmag - 1),
                         ('p_bcg', 'f4')])

    redmodel = Entry.from_fits_file(config.redgalmodelfile)
    for j in range(config.nmag - 1):
        y2 = cubic_spline_compute_y2(redmodel.nodes, redmodel.meancol[:, j])
        cat.redcolor[:, j] = cubic_spline_interpolate(cat.z, redmodel.nodes, redmodel.meancol[:, j], y2)

    state['zbounds'] = np.concatenate([np.array([config.zrange[0] - 0.011]),
                                   config.calib_colormem_zbounds,
                                   np.array([config.zrange[1] + 0.011])])

    generate_mem_match_ids(cat)
    state['cat'] = cat
    return state, True

def _colormem_process_cluster(cluster, state):
    bad = False
    config = state['config']
    m = 0
    found = False
    while ((m < state['zbounds'].size - 1) and (not found)):
        if (cluster.z > state['zbounds'][m]) and (cluster.z <= state['zbounds'][m + 1]):
            found = True
            mode = config.calib_colormem_colormodes[m]
        else:
            m += 1
    if (not found):
        raise RuntimeError("Programmer error with illegal mode")

    lam = cluster.calc_richness_fit(state['mask'], mode, calc_err=False,
                                    centcolor_in=cluster.redcolor[mode])

    ind = np.argmin(cluster.neighbors.r)
    cluster.p_bcg = cluster.neighbors.pmem[ind]

    if (lam / cluster.scaleval < config.calib_colormem_minlambda):
        bad = True
        reset_bad_values(cluster)
        return bad

    if cluster.p_bcg < config.calib_pcut:
        bad = True
        reset_bad_values(cluster)
        return bad

    return bad

def _colormem_postprocess(cat, members, state):
    config = state['config']
    use, = np.where((cat.Lambda/cat.scaleval >= config.calib_colormem_minlambda) & (cat.scaleval > 0.0) & (cat.maskfrac < config.max_maskfrac))

    cat = ClusterCatalog(cat._ndarray[use])

    if members is not None:
        a, b = esutil.numpy_util.match(cat.mem_match_id, members.mem_match_id)
        members = Catalog(members._ndarray[b])

        if config.calib_colormem_smooth > 0.0:
            rng = np.random.RandomState(config.randomseed)
            members.z += rng.normal(scale=config.calib_colormem_smooth, size=members.size)
            
    return cat, members

def run_colormem(conf):
    """
    Run colormem on a catalog.
    """
    if not isinstance(conf, Configuration):
        config = Configuration(conf)
    else:
        config = conf

    def doublerun_sort_fn(cat):
        st = np.argsort(cat.Lambda)[::-1]
        return cat[st]

    cat, members = run_cluster_pipeline(
        config,
        runmode="calib_colormem",
        filetype="colormem",
        more_setup_fn=_colormem_more_setup,
        process_cluster_fn=_colormem_process_cluster,
        postprocess_fn=_colormem_postprocess,
        read_gals=True,
        read_zreds=False,
        zreds_required=False,
        use_colorbkg=True,
        use_parfile=False,
        cutgals_bkgrange=False,
        cutgals_chisqmax=False,
        do_lam_plusminus=False,
        record_members=False,
        doublerun=True,
        doublerun_sort_fn=doublerun_sort_fn,
        min_lambda=config.calib_colormem_minlambda
    )

    if config.more_qa_plots and cat is not None and cat.size > 0:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(cat.z, cat.Lambda, c='r', marker='o', s=16)
        ax.set_xlabel(r'$z$', fontsize=16)
        ax.set_ylabel(r'$\lambda$', fontsize=16)
        ax.set_title('Colormem Lambda vs Redshift')
        fig.savefig(config.redmapper_filename('colormem_qa', paths=[config.plotpath], filetype='png'))
        plt.close(fig)

        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.hist([item.p_bcg for item in cat], bins=20, range=(0, 1))
            ax.set_xlabel(r'$P_{\mathrm{BCG}}$', fontsize=16)
            ax.set_ylabel(r'$N_{\mathrm{clusters}}$', fontsize=16)
            ax.set_title('Colormem P_BCG')
            fig.savefig(config.redmapper_filename('colormem_pbcg_qa', paths=[config.plotpath], filetype='png'))
            plt.close(fig)
        except Exception as e:
            logger.warning("Could not make P_BCG plot: {}".format(e))
        finally:
            plt.close(1)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.hist(cat.Lambda, bins=50, range=(0, 200))
        ax.set_xlabel(r'$\lambda$', fontsize=16)
        ax.set_ylabel(r'$N_{\mathrm{clusters}}$', fontsize=16)
        ax.set_title('Colormem Lambda')
        fig.savefig(config.redmapper_filename('colormem_lambda_qa', paths=[config.plotpath], filetype='png'))
        plt.close(fig)

    return cat, members


