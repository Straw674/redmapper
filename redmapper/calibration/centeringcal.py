"""Functions related to calibrating the centering model
"""
import os
import numpy as np
import fitsio
import scipy.optimize
import warnings

from ..utilities import sample_from_pdf, histoGauss, chisq_pdf
from ..redsequence import read_redsequence, redsequence_mstar, compute_redsequence_chisq
from ..background import read_background, compute_background
from ..cluster import ClusterCatalog
from ..galaxy import GalaxyCatalog

import matplotlib.pyplot as plt

def fit_wcen_fg(w, lscale, p0):
    """
    Fit the foreground centering parameters

    Parameters
    ----------
    w: `np.array`
       Float array of w values
    lscale: `np.array`
       Richness scaled by richness pivot value (lambda / lambda_pivot)
    p0: `list`
       Initial parameters
       p0[0]: log-mean w of foreground galaxies
       p0[1]: log-sigma w of foreground galaxies

    Returns
    -------
    pars: `list`
       Best-fit parameters
    """
    def cost(pars):
        sig = pars[1] * lscale
        f = (1./(np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - pars[0])**2. / (sig**2.))
        t = -np.sum(np.log(f))
        if pars[1] < 0.0:
            t += 1000.0
        return t

    pars = scipy.optimize.fmin(cost, p0, disp=False, xtol=1e-5, ftol=1e-5)
    return pars

def fit_wcen_c(pcen, psat, mstar, lamscale, refmag, cwt, phi1, bcounts, p0):
    """
    Fit the mean magnitude model of the central galaxies.

    Parameters
    ----------
    pcen: `np.array`
       Float array of probability of being the correct center
    psat: `np.array`
       Float array of probability of being a satellite galaxy
    mstar: `np.array`
       Float array of mstar for the galaxies
    lamscale: `np.array`
       Float array of lambda/pivot for the galaxies
    refmag: `np.array`
       Float array of Total magnitude in the reference band
    cwt: `np.array`
       Float array of chi-squared weight from chisq_pdf()
    phi1: `np.array`
       Float array of Gaussian pdf of brightest galaxy sampled
       from a Schechter function
    bcounts: `np.array`
       Float array of background probability, assuming uniform
       background (not nfw)
    p0: `list`
       p0[0]: Delta0
       p0[1]: Delta1
       p0[2]: sigma_m
       mean mag mbar = mstar + Delta0 + delta1 * log(lambda / pivot)

    Returns
    -------
    pars: `list`
       Best fit parameters
    """
    def cost(pars):
        mbar = mstar + pars[0] + pars[1] * lamscale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phicen = (1./(np.sqrt(2.*np.pi) * pars[2])) * np.exp(-0.5*(refmag - mbar)**2. / (pars[2]**2.))
            rho = pcen * phicen * cwt + psat * phi1 * cwt + (1. - pcen - psat) * bcounts
            bad, = np.where((rho < 1e-5) | (~np.isfinite(rho)))
            rho[bad] = 1e-5

        t = -np.sum(np.log(rho))
        if pars[2] < 0.0: t += 1000
        return t

    pars = scipy.optimize.fmin(cost, p0, disp=False, xtol=1e-5, ftol=1e-5)
    return pars

def fit_wcen_cw(pcen, psat, wcen, ffg, fsat, lscale, p0):
    """
    Fit f(w) model for central galaxies

    Parameters
    ----------
    pcen: `np.array`
       Float array of probability of being the correct center
    psat: `np.array`
       Float array of probability of being a satellite galaxy
    wcen: `np.array`
       Float array of w connectivity from previous iteration
    ffg: `np.array`
       Float array of f_fg(w) for foreground galaxies
    fsat: `np.array`
       Float array of f_sat(w) for satellite galaxies
    lscale: `np.array`
       Float array of lambda/pivot for the galaxies
    p0: `list`
       p0[0]: wcen_mean
       p0[1]: wcen_sigma

    Returns
    -------
    pars: `list`
       Best fit parameters
    """
    def cost(pars):
        sig = pars[1] * lscale
        fcen = (1. / (np.sqrt(2.*np.pi)*sig)) * np.exp(-0.5*(np.log(wcen) - pars[0])**2. / (sig**2.))
        f = pcen * fcen + psat * fsat + (1. - pcen - psat) * ffg
        t = -np.sum(np.log(f))
        if pars[1] < 0.0: t+=1000.0
        return t

    pars = scipy.optimize.fmin(cost, p0, disp=False, xtol=1e-5, ftol=1e-5)
    return pars

def schechter_montecarlo_calib(config, rng=None, testing=False):
    """
    Calibrate the brightest galaxy sampled from a schechter function with a
    simple monte carlo.

    m1mstar is the magnitude of the brightest galaxy sampled from a
    schechter function minus mstar.

    The functional form of the parametrizations are:

    mmstar1 = phi1_mmstar_m * log(lambda/pivot)**phi1_mmstar_slope
    msig1 = phi1_msig_m * log(lambda/pivot)**phi1_msig_slope

    Such that if you want to sample from the brightest galaxy of a
    schechter function for a given richness, you sample from a Gaussian of
    mean mmstar1 and sigma msig1.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    rng: `np.random.RandomState`, optional
       Random number generator.
    testing: `bool`, optional
       Run in testing mode.  Used for unit tests.  Default is False.

    Returns
    -------
    phi1_mmstar_m: `float`
    phi1_mmstar_slope: `float`
    phi1_msig_m: `float`
    phi1_msig_slope: `float`
    """
    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    if testing:
        nmag = 1000
        ntrial = 100
        nlambdas = 3
    else:
        nmag = 100000
        ntrial = 5000
        nlambdas = 9

    mag = np.zeros(nmag)

    mstar = 0.0
    mrange = -2.5 * np.log10(np.array([10.0, 0.2]))
    step = 0.002

    def schechter(x, alpha=-1.0, mstar=0.0):
        return 10.**(0.4*(alpha + 1.0)*(mstar - x)) * np.exp(-10.**(0.4*(mstar - x)))

    mag = sample_from_pdf(schechter, mrange, step, nmag, rng, alpha=config.calib_lumfunc_alpha, mstar=mstar)

    # We want to sample lambda galaxies from a schechter function...
    # And figure out the 3 brightest galaxies (m1, m2, m3)

    lambdas = np.linspace(20, 100, num=nlambdas, dtype=np.int32)

    m1 = np.zeros((nlambdas, ntrial))
    m2 = np.zeros_like(m1)
    m3 = np.zeros_like(m1)

    for i in range(ntrial):
        r = rng.rand(nmag)
        st = np.argsort(r)

        for j in range(nlambdas):
            u = st[0:lambdas[j] - 1]
            st2 = np.argsort(mag[u])

            m1[j, i] = mag[u[st2[0]]]
            m2[j, i] = mag[u[st2[1]]]
            m3[j, i] = mag[u[st2[2]]]

    mmstar1_mean = np.zeros(nlambdas)
    mmstar1_sigma = np.zeros(nlambdas)
    mmstar2_mean = np.zeros(nlambdas)
    mmstar2_sigma = np.zeros(nlambdas)
    mmstar3_mean = np.zeros(nlambdas)
    mmstar3_sigma = np.zeros(nlambdas)

    for i in range(nlambdas):
        coeff = histoGauss(None, m1[i, :])
        mmstar1_mean[i] = coeff[1]
        mmstar1_sigma[i] = coeff[2]

        coeff = histoGauss(None, m2[i, :])
        mmstar2_mean[i] = coeff[1]
        mmstar2_sigma[i] = coeff[2]

        coeff = histoGauss(None, m3[i, :])
        mmstar3_mean[i] = coeff[1]
        mmstar3_sigma[i] = coeff[2]

    fit = np.polyfit(np.log(lambdas / config.wcen_pivot), mmstar1_mean, 1)

    phi1_mmstar_m = fit[1]
    phi1_mmstar_slope = fit[0]

    fit = np.polyfit(np.log(lambdas / config.wcen_pivot), mmstar1_sigma, 1)
    phi1_msig_m = fit[1]
    phi1_msig_slope = fit[0]

    return phi1_mmstar_m, phi1_mmstar_slope, phi1_msig_m, phi1_msig_slope


def calibrate_wcen(config, iteration, randcatfile=None, randsatcatfile=None, rng=None, testing=False):
    """
    Run the wcen calibration routine.

    Parameters
    ----------
    config: `redmapper.Configuration`
       Configuration object
    iteration: `int`
       Iteration number.  If iteration==1, then must set randcatfile and
       randsatfile to calibrate foreground and satellite functions
    randcatfile: `str`, optional
       Catalog file with richness information on random (foreground)
       points. Default is None, but must be set if iteration==1.
    randsatcatfile: `str`, optional
       Catalog file with richness information on randomly selected
       satellites.  Default is None, but must be set if iteration==1.
    rng : `np.random.RandomState`, optional
        Random number generator.
    testing: `bool`, optional
       Run in fast testing mode, for unit tests.  Default is False.

    Returns
    -------
    wcenstr: `np.ndarray`
       Array of calibration parameters written to file.
    """
    if rng is None:
        rng = np.random.RandomState(config.randomseed)

    if iteration == 1:
        if randcatfile is None:
            if config.lnw_fg_sigma < 0:
                raise RuntimeError("randcatfile must be set on iteration 1, or lnw_fg_mean and lnw_fg_sigma must be set in configuration")
        if randsatcatfile is None:
            if config.lnw_sat_sigma < 0:
                raise RuntimeError("randsatcatfile must be set on iteration 1, or lnw_sat_mean and lnw_sat_sigma must be set in configuration")

    # Calibrate the brightest galaxy from the schechter function
    if config.phi1_mmstar_m < -1000.0:
        # Need to run the schechter calibration
        phi1_mmstar_m, phi1_mmstar_slope, phi1_msig_m, phi1_msig_slope = schechter_montecarlo_calib(config, rng=rng, testing=testing)
    else:
        # We have the numbers already
        phi1_mmstar_m = config.phi1_mmstar_m
        phi1_mmstar_slope = config.phi1_mmstar_slope
        phi1_msig_m = config.phi1_msig_m
        phi1_msig_slope = config.phi1_msig_slope

    # Read in the parameters (fine steps)
    zredstr = read_redsequence(config.parfile, fine=True)

    # Read in the background
    bkg = read_background(config.bkgfile)

    # Read in the catalog
    cat = ClusterCatalog.from_catfile(config.catfile, cosmo=config.cosmo)

    # We set the redshift according to the initial spec redshift for training
    cat.z = cat.z_spec_init

    # Select clusters for wcen training
    use, = np.where((cat.Lambda/cat.scaleval > config.wcen_minlambda) &
                    (cat.Lambda/cat.scaleval < config.wcen_maxlambda) &
                    (cat.w > 0.0) &
                    (cat.z > config.wcen_cal_zrange[0]) &
                    (cat.z < config.wcen_cal_zrange[1]) &
                    (cat.maskfrac < config.max_maskfrac))

    cat = cat[use]

    randfiles = [randcatfile, randsatcatfile]
    # note that the config variables might already be set...
    for randfile in randfiles:
        if randfile is None:
            continue

        rcat = ClusterCatalog.from_catfile(randfile, cosmo=config.cosmo)
        rcat.z = rcat.z_spec_init

        use, = np.where((rcat.Lambda/rcat.scaleval > config.wcen_minlambda) &
                        (rcat.Lambda/rcat.scaleval < config.wcen_maxlambda) &
                        (rcat.w > 0.0) &
                        (rcat.z > config.wcen_cal_zrange[0]) &
                        (rcat.z < config.wcen_cal_zrange[1]) &
                        (rcat.maskfrac < config.max_maskfrac))
        rcat = rcat[use]

        lscalefg = 1./np.sqrt((rcat.Lambda / rcat.scaleval) / config.wcen_pivot)

        p0 = np.array([np.mean(np.log(rcat.w)), np.std(np.log(rcat.w))])
        p = fit_wcen_fg(rcat.w, lscalefg, p0)

        if randfile == randcatfile:
            config.lnw_fg_mean = p[0]
            config.lnw_fg_sigma = p[1]
        else:
            config.lnw_sat_mean = p[0]
            config.lnw_sat_sigma = p[1]

    # Prepare the model fits
    mstars = redsequence_mstar(zredstr, cat.z)

    # Get the starting values...
    def _linfunc(p, x, y):
        return (p[1] + p[0] * x) - y

    fit = scipy.optimize.least_squares(_linfunc, [0.0, 0.0], loss='soft_l1',
                                       args=(np.log(cat.Lambda / config.wcen_pivot),
                                             cat.refmag - mstars))
    Delta0 = fit.x[1]
    Delta1 = fit.x[0]

    resid = (cat.refmag - mstars) - (Delta0 + Delta1*np.log(cat.Lambda / config.wcen_pivot))
    sigma_m = np.std(resid)

    # This needs to be made more elegant if this is a common use case.
    chisqs = compute_redsequence_chisq(zredstr, GalaxyCatalog(cat._ndarray), cat.z)

    cwt = chisq_pdf(chisqs, zredstr['ncol'])

    mmstar1 = mstars + phi1_mmstar_m + phi1_mmstar_slope * np.log(cat.Lambda / config.wcen_pivot)
    phisig1 = phi1_msig_m + phi1_msig_slope * np.log(cat.Lambda / config.wcen_pivot)
    phi1 = (1./(np.sqrt(2.*np.pi)*phisig1)) * np.exp(-0.5*(cat.refmag - mmstar1)**2. / (phisig1**2.))

    sigma_g = compute_background(bkg, cat.z, chisqs, cat.refmag)
    mpc_scale = np.radians(1.) * config.cosmo.Da(0, cat.z)
    bcounts = (sigma_g / mpc_scale**2.) * np.pi * cat.r_lambda**2.

    lscale = np.log((cat.Lambda / cat.scaleval) / config.wcen_pivot)

    p0 = np.array([Delta0, Delta1, sigma_m])
    p = fit_wcen_c(cat.p_cen[:, 0], cat.p_sat[:, 0], mstars, lscale, cat.refmag, cwt, phi1, bcounts, p0)

    fgsig = config.lnw_fg_sigma / np.sqrt((cat.Lambda / cat.scaleval) / config.wcen_pivot)
    ffg = (1./(np.sqrt(2.*np.pi) * fgsig)) * np.exp(-0.5*(np.log(cat.w) - config.lnw_fg_mean)**2. / (fgsig**2.))
    satsig = config.lnw_sat_sigma / np.sqrt((cat.Lambda / cat.scaleval) / config.wcen_pivot)
    fsat = (1./(np.sqrt(2.*np.pi) * satsig)) * np.exp(-0.5*(np.log(cat.w) - config.lnw_sat_mean)**2. / (satsig**2.))
    lscalefg = 1./np.sqrt((cat.Lambda / cat.scaleval) / config.wcen_pivot)

    p0 = np.array([np.mean(np.log(cat.w)), np.std(np.log(cat.w))])
    wp = fit_wcen_cw(cat.p_cen[:, 0], cat.p_sat[:, 0], cat.w, ffg, fsat, lscalefg, p0)

    # QA plotting
    _make_wcen_qa_plots(config, cat, ffg, fsat, config.lnw_fg_mean, config.lnw_fg_sigma, config.lnw_sat_mean, config.lnw_sat_sigma, wp)

    # and save this...
    wcenstr = np.zeros(1, dtype=[('delta0', 'f8'),
                                 ('delta1', 'f8'),
                                 ('sigma_m', 'f8'),
                                 ('pivot', 'f8'),
                                 ('lnw_fg_mean', 'f8'),
                                 ('lnw_fg_sigma', 'f8'),
                                 ('lnw_sat_mean', 'f8'),
                                 ('lnw_sat_sigma', 'f8'),
                                 ('lnw_cen_mean', 'f8'),
                                 ('lnw_cen_sigma', 'f8'),
                                 ('phi1_mmstar_m', 'f8'),
                                 ('phi1_mmstar_slope', 'f8'),
                                 ('phi1_msig_m', 'f8'),
                                 ('phi1_msig_slope', 'f8')])
    wcenstr['delta0'] = p[0]
    wcenstr['delta1'] = p[1]
    wcenstr['sigma_m'] = p[2]
    wcenstr['pivot'] = config.wcen_pivot
    wcenstr['lnw_fg_mean'] = config.lnw_fg_mean
    wcenstr['lnw_fg_sigma'] = config.lnw_fg_sigma
    wcenstr['lnw_sat_mean'] = config.lnw_sat_mean
    wcenstr['lnw_sat_sigma'] = config.lnw_sat_sigma
    wcenstr['lnw_cen_mean'] = wp[0]
    wcenstr['lnw_cen_sigma'] = wp[1]
    wcenstr['phi1_mmstar_m'] = phi1_mmstar_m
    wcenstr['phi1_mmstar_slope'] = phi1_mmstar_slope
    wcenstr['phi1_msig_m'] = phi1_msig_m
    wcenstr['phi1_msig_slope'] = phi1_msig_slope

    fitsio.write(config.wcenfile, wcenstr, clobber=True)
    return wcenstr

def _make_wcen_qa_plots(config, cat, ffg, fsat, lnw_fg_mean, lnw_fg_sigma, lnw_sat_mean, lnw_sat_sigma, wp):
    """
    Make QA plots for wcen calibration.
    """
    if config.plotpath is None:
        return

    # Let's plot the distributions of log(w) for cen, sat, and fg
    fig, ax = plt.subplots(figsize=(8, 6))

    # Actual histograms
    bins = np.linspace(-3, 3, 50)
    ax.hist(np.log(cat.w), bins=bins, density=True, alpha=0.5, label='All w', color='k')

    x = np.linspace(-3, 3, 200)

    # Plot the functional forms based on the fitted parameters
    fg_dist = (1. / (np.sqrt(2. * np.pi) * lnw_fg_sigma)) * np.exp(-0.5 * (x - lnw_fg_mean)**2 / lnw_fg_sigma**2)
    sat_dist = (1. / (np.sqrt(2. * np.pi) * lnw_sat_sigma)) * np.exp(-0.5 * (x - lnw_sat_mean)**2 / lnw_sat_sigma**2)
    cen_dist = (1. / (np.sqrt(2. * np.pi) * wp[1])) * np.exp(-0.5 * (x - wp[0])**2 / wp[1]**2)

    ax.plot(x, fg_dist, 'r--', label='Foreground Model')
    ax.plot(x, sat_dist, 'b--', label='Satellite Model')
    ax.plot(x, cen_dist, 'g--', label='Central Model')

    ax.set_xlabel('log(w)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    ax.set_title('Centering Calibration QA')

    fig.tight_layout()
    fig.savefig(os.path.join(config.plotpath, '%s_wcen_qa.png' % (config.outbase)))
    plt.close(fig)