import fitsio
import esutil
import numpy as np
from redmapper.utilities import gaussFunction
from redmapper.utilities import interpol

def _init_centering_result(maxcen):
    return {
        'ra': np.zeros(maxcen) - 400.0,
        'dec': np.zeros(maxcen) - 400.0,
        'ngood': 0,
        'index': np.zeros(maxcen, dtype=np.int32) - 1,
        'maxind': -1,
        'lnlamlike': -1.0,
        'lnbcglike': -1.0,
        'p_cen': np.zeros(maxcen),
        'q_cen': np.zeros(maxcen),
        'p_fg': np.zeros(maxcen),
        'q_miss': 0.0,
        'p_sat': np.zeros(maxcen),
        'p_c': np.zeros(maxcen),
        'success': False
    }

def compute_centering_bcg(cluster, config, rng=None, **kwargs):
    maxcen = config.percolation_maxcen
    res = _init_centering_result(maxcen)
    
    pmem_cut = 0.8
    z_neighbors = cluster.neighbors.zred
    z_neighbors_e = cluster.neighbors.zred_e
    
    if config.centering_use_zspec:
        has_zspec, = np.where(cluster.neighbors.zspec > 0.0)
        z_neighbors[has_zspec] = cluster.neighbors.zspec[has_zspec]
        z_neighbors_e[has_zspec] = cluster.neighbors.zspec_err[has_zspec]

    use, = np.where((cluster.neighbors.r < cluster.r_lambda) &
                    ((cluster.neighbors.pmem > pmem_cut) |
                     (np.abs(z_neighbors - cluster.redshift) < 2.0 * z_neighbors_e)))

    if use.size == 0:
        return res

    mind = np.argmin(cluster.neighbors.refmag[use])

    res['maxind'] = use[mind]
    res['ra'][0] = cluster.neighbors.ra[res['maxind']]
    res['dec'][0] = cluster.neighbors.dec[res['maxind']]
    res['ngood'] = 1
    res['index'][0] = res['maxind']
    res['p_cen'][0] = 1.0
    res['q_cen'][0] = 1.0
    res['p_sat'][0] = 0.0
    res['p_c'][0] = 1.0
    res['success'] = True
    return res

def compute_centering_wcen_zred(cluster, config, zlambda_corr=None, rng=None, **kwargs):
    maxcen = config.percolation_maxcen
    res = _init_centering_result(maxcen)

    z_neighbors = cluster.neighbors.zred
    z_neighbors_e = cluster.neighbors.zred_e
    chisq_neighbors = cluster.neighbors.zred_chisq
    
    if config.centering_use_zspec:
        has_zspec, = np.where(cluster.neighbors.zspec > 0.0)
        z_neighbors[has_zspec] = cluster.neighbors.zspec[has_zspec]
        z_neighbors_e[has_zspec] = cluster.neighbors.zspec_err[has_zspec]
        chisq_neighbors[has_zspec] = 1.0

    use, = np.where((cluster.neighbors.r < cluster.r_lambda) &
                    (cluster.neighbors.pfree >= config.percolation_pbcg_cut) &
                    (chisq_neighbors < config.wcen_zred_chisq_max) &
                    ((cluster.neighbors.pmem > 0.0) |
                     (np.abs(cluster.redshift - z_neighbors) < 5.0 * z_neighbors_e)))

    mbar = cluster.mstar + config.wcen_Delta0 + config.wcen_Delta1 * np.log(cluster.Lambda / config.wcen_pivot)
    phi_cen = gaussFunction(cluster.neighbors.refmag[use],
                            1. / (np.sqrt(2. * np.pi) * config.wcen_sigma_m),
                            mbar,
                            config.wcen_sigma_m)

    if zlambda_corr is not None:
        zrmod = interpol(zlambda_corr['zred_uncorr'], zlambda_corr['z'], cluster.redshift)
        gz = gaussFunction(z_neighbors[use],
                           1. / (np.sqrt(2. * np.pi) * z_neighbors_e[use]),
                           zrmod,
                           z_neighbors_e[use])
    else:
        gz = gaussFunction(z_neighbors[use],
                           1. / (np.sqrt(2. * np.pi) * z_neighbors_e[use]),
                           cluster.redshift,
                           z_neighbors_e[use])

    u, = np.where(cluster.neighbors.p > 0.0)
    maxrad = 1.1 * cluster.r_lambda / cluster.mpc_scale

    htm_matcher = esutil.htm.Matcher(cluster.neighbors.depth,
                                     cluster.neighbors.ra[use],
                                     cluster.neighbors.dec[use])
    i2, i1, dist = htm_matcher.match(cluster.neighbors.ra[u],
                                     cluster.neighbors.dec[u],
                                     maxrad, maxmatch=0)

    subdifferent, = np.where(~(use[i1] == u[i2]))
    i1 = i1[subdifferent]
    i2 = i2[subdifferent]
    pdis = dist[subdifferent] * cluster.mpc_scale
    pdis = np.sqrt(pdis**2. + config.wcen_rsoft**2.)

    lum = 10.**((cluster.mstar - cluster.neighbors.refmag) / (2.5))
    w = np.zeros(use.size) + 1e-3
    for i in range(use.size):
        subgal, = np.where(i1 == i)
        if subgal.size > 0:
            inside, = np.where(pdis[subgal] < cluster.r_lambda)
            if inside.size > 0:
                indices = u[i2[subgal[inside]]]
                if config.wcen_uselum:
                    w[i] = np.log(np.sum(cluster.neighbors.p[indices] * lum[indices] /
                                         pdis[subgal[inside]]) /
                                  ((1. / cluster.r_lambda) *
                                   np.sum(cluster.neighbors.p[indices] * lum[indices])))
                else:
                    w[i] = np.log(np.sum(cluster.neighbors.p[indices] /
                                         pdis[subgal[inside]]) /
                                  ((1. / cluster.r_lambda) *
                                   np.sum(cluster.neighbors.p[indices])))

    sigscale = np.sqrt((np.clip(cluster.Lambda, None, config.wcen_maxlambda) / cluster.scaleval) / config.wcen_pivot)
    sig = config.lnw_cen_sigma / sigscale
    fw = gaussFunction(np.log(w),
                       1. / (np.sqrt(2. * np.pi) * sig),
                       config.lnw_cen_mean,
                       sig)

    ucen = phi_cen * gz * fw
    lo, = np.where(ucen < 1e-10)
    ucen[lo] = 0.0

    maxmag = cluster.mstar - 2.5 * np.log10(config.lval_reference)
    phi_sat = cluster._calc_luminosity(maxmag, idx=use)

    satsig = config.lnw_sat_sigma / sigscale
    fsat = gaussFunction(np.log(w),
                         1. / (np.sqrt(2. * np.pi) * satsig),
                         config.lnw_sat_mean,
                         satsig)

    usat = phi_sat * gz * fsat
    lo, = np.where(usat < 1e-10)
    usat[lo] = 0.0

    fgsig = config.lnw_fg_sigma / sigscale
    ffg = gaussFunction(np.log(w),
                        1. / (np.sqrt(2. * np.pi) * fgsig),
                        config.lnw_fg_mean,
                        fgsig)

    rtest = np.zeros(use.size) + 0.1
    bcounts = ffg * (cluster.calc_zred_bkg_density(rtest,
                                                        z_neighbors[use],
                                                        cluster.neighbors.refmag[use]) /
                     (2. * np.pi * rtest)) * np.pi * cluster.r_lambda**2.

    Pcen_basic = np.clip(cluster.neighbors.pfree[use] * (ucen / (ucen + (cluster.Lambda / cluster.scaleval - 1.0) * usat + bcounts)), None, 0.99999)

    bad, = np.where(~np.isfinite(Pcen_basic))
    Pcen_basic[bad] = 0.0

    okay, = np.where(Pcen_basic > 0.0)
    if okay.size == 0:
        res['q_miss'] = 1.0
        good = np.atleast_1d(np.argmin(cluster.neighbors.r[use]))
        maxind = use[good[0]]
        Pcen = np.zeros(use.size)
        Qcen = np.zeros(use.size)
    else:
        Pcen_unnorm = np.zeros(use.size)
        ok, = np.where(Pcen_basic > 0)
        st = np.argsort(Pcen_basic[ok])[::-1]
        if st.size < config.percolation_maxcen:
            good = ok[st]
        else:
            good = ok[st[0: config.percolation_maxcen]]

        res['ngood'] = good.size

        for i in range(res['ngood']):
            Pcen0 = Pcen_basic[good[i]]
            Pcen_basic[good[i]] = 0.0
            Pcen_unnorm[good[i]] = Pcen0 * np.prod(1.0 - Pcen_basic[good])
            Pcen_basic[good[i]] = Pcen0

        Qmiss = np.prod(1.0 - Pcen_basic[good])
        KQ = 1./(Qmiss + np.sum(Pcen_unnorm))
        KP = 1./np.sum(Pcen_unnorm)

        Pcen = KP * Pcen_unnorm
        Qcen = KQ * Pcen_unnorm
        maxind = use[good[0]]

    Pfg_basic = bcounts[good] / ((cluster.Lambda - 1.0) * usat[good] + bcounts[good])
    inf, = np.where(~np.isfinite(Pfg_basic))
    Pfg_basic[inf] = 0.0
    Pfg = (1.0 - Pcen[good]) * Pfg_basic

    Psat_basic = (cluster.Lambda - 1.0) * usat[good] / ((cluster.Lambda - 1.0) * usat[good] + bcounts[good])
    inf, = np.where(~np.isfinite(Psat_basic))
    Psat_basic[inf] = 0.0
    Psat = (1.0 - Pcen[good]) * Psat_basic

    if okay.size > 0:
        res['ra'][0: good.size] = cluster.neighbors.ra[use[good]]
        res['dec'][0: good.size] = cluster.neighbors.dec[use[good]]
        res['maxind'] = use[good[0]]
        res['index'][0: good.size] = use[good]
        res['p_cen'][0: good.size] = Pcen[good]
        res['q_cen'][0: good.size] = Qcen[good]
        res['p_fg'][0: good.size] = Pfg
        res['p_sat'][0: good.size] = Psat
        res['p_c'][0: good.size] = Pcen_basic[good]

    res['success'] = True
    return res

def compute_centering_random(cluster, config, rng=None, **kwargs):
    if rng is None:
        rng = np.random.RandomState()
    maxcen = config.percolation_maxcen
    res = _init_centering_result(maxcen)
    
    r = cluster.r_lambda * np.sqrt(rng.random(size=1))
    phi = 2. * np.pi * rng.random(size=1)

    x = r * np.cos(phi) / (cluster.mpc_scale)
    y = r * np.sin(phi) / (cluster.mpc_scale)

    ra_cen = cluster.ra + x / np.cos(np.radians(cluster.dec))
    dec_cen = cluster.dec + y

    res['ra'][0] = ra_cen[0]
    res['dec'][0] = dec_cen[0]
    res['ngood'] = 1
    res['index'][0] = -1
    res['maxind'] = -1
    res['p_cen'][0] = 1.0
    res['q_cen'][0] = 1.0
    res['p_sat'][0] = 0.0
    res['p_fg'][0] = 0.0
    res['p_c'][0] = 1.0
    res['success'] = True
    return res

def compute_centering_random_satellite(cluster, config, rng=None, **kwargs):
    if rng is None:
        rng = np.random.RandomState()
    maxcen = config.percolation_maxcen
    res = _init_centering_result(maxcen)

    st = np.argsort(cluster.neighbors.pmem)[::-1]
    pdf = cluster.neighbors.pmem[st]
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf, dtype=np.float64)
    cdfi = (cdf * st.size).astype(np.int32)

    rand = (rng.uniform(size=1) * st.size).astype(np.int32)
    ind = np.where(cdfi >= rand[0])[0][0]
    maxind = st[ind]

    res['ra'][0] = cluster.neighbors.ra[maxind]
    res['dec'][0] = cluster.neighbors.dec[maxind]
    res['index'][0] = maxind
    res['maxind'] = maxind
    res['ngood'] = 1
    res['p_cen'][0] = 1.0
    res['q_cen'][0] = 1.0
    res['p_sat'][0] = 0.0
    res['p_fg'][0] = 0.0
    res['p_c'][0] = 1.0
    res['success'] = True
    return res

CENTERING_FUNCS = {
    'CenteringBCG': compute_centering_bcg,
    'CenteringWcenZred': compute_centering_wcen_zred,
    'CenteringRandom': compute_centering_random,
    'CenteringRandomSatellite': compute_centering_random_satellite
}
