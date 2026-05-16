"""Functions to compute zred with a color-based red sequence model.
"""
import numpy as np
import scipy.integrate
import scipy.interpolate
import copy

from .utilities import interpol, sample_from_pdf
from .redsequence import compute_redsequence_chisq_redshifts, compute_redsequence_chisq

def compute_zreds(zredstr, galaxies, sigint=0.001, do_correction=True,
                  use_photoerr=True, zrange=None, rng=None):
    """
    Compute zreds for a catalog of galaxies.

    Will set galaxies['zred'], galaxies['zred_e'], etc.

    Parameters
    ----------
    zredstr: `dict`
       Red sequence parametrization (dictionary)
    galaxies: `redmapper.GalaxyCatalog` or `astropy.table.Table`
       Catalog of galaxies to compute zred.
    sigint: `float`, optional
       Intrinsic scatter floor.  Default is 0.001.
    do_correction: `bool`, optional
       Apply zred correction terms?  Default is True.
    use_photoerr: `bool`, optional
       Use photometric errors in computing zred?  Default is True.
    zrange: `list`, optional
       Redshift range.  Useful for testing.  Default is None (use
       zredstr redshift range).
    rng: `np.random.RandomState`, optional
       Random number generator. Default is None.
    """
    if rng is None:
        rng = np.random.RandomState()

    for galaxy in galaxies:
        compute_zred(zredstr, galaxy, sigint=sigint, do_correction=do_correction,
                     use_photoerr=use_photoerr, zrange=zrange, rng=rng, no_corrections=True)

    if do_correction:
        # Bulk processing
        ok, = np.where(galaxies['zred_uncorr'] > 0)

        if ok.size > 0:
            olddzs = np.zeros(ok.size)
            dzs = np.zeros_like(olddzs)
            iteration = 0

            pivotmags = interpol(zredstr['pivotmag'], zredstr['z'], galaxies['zred_uncorr'][ok])

            while (iteration < 5):
                olddzs[:] = dzs
                dzs[:] = (interpol(zredstr['corr'], zredstr['z'], galaxies['zred_uncorr'][ok] + olddzs) +
                          (galaxies['refmag'][ok] - pivotmags) *
                          interpol(zredstr['corr_slope'], zredstr['z'], galaxies['zred_uncorr'][ok] + olddzs))
                iteration += 1

            galaxies['zred'][ok] = galaxies['zred_uncorr'][ok] + dzs
            galaxies['zred_e'][ok] = galaxies['zred_uncorr_e'][ok] * interpol(zredstr['corr_r'], zredstr['z'], galaxies['zred'][ok])

            dz2s = interpol(zredstr['corr2'], zredstr['z'], galaxies['zred_uncorr'][ok])
            r2s = interpol(zredstr['corr2_r'], zredstr['z'], galaxies['zred_uncorr'][ok])

            galaxies['zred2'][ok] = galaxies['zred_uncorr'][ok] + dz2s
            galaxies['zred2_e'][ok] = galaxies['zred_uncorr_e'][ok] * r2s

def compute_zred(zredstr, galaxy, sigint=0.001, do_correction=True,
                 use_photoerr=True, zrange=None, rng=None, no_corrections=False):
    """
    Compute zred for a single galaxy.

    Will set galaxy['zred'], galaxy['zred_e'], etc.

    Parameters
    ----------
    zredstr: `dict`
       Red sequence parametrization
    galaxy: `astropy.table.Row`
       Galaxy to compute zred
    sigint: `float`, optional
       Intrinsic scatter floor.  Default is 0.001.
    do_correction: `bool`, optional
       Apply zred correction terms?  Default is True.
    use_photoerr: `bool`, optional
       Use photometric errors in computing zred?  Default is True.
    zrange: `list`, optional
       Redshift range.  Useful for testing.  Default is None (use
       zredstr redshift range).
    rng: `np.random.RandomState`, optional
       Random number generator.
    no_corrections: `bool`, optional
       Do not apply redshift corrections.  Default is False.
    """
    if rng is None:
        rng = np.random.RandomState()

    nz = zredstr['z'].size - 1
    notextrap, = np.where(~zredstr['extrapolated'])

    lndist = np.zeros(nz) - 1e12
    chisq = np.zeros(nz) + 1e12

    zbins_limited, = np.where((galaxy['refmag'] < zredstr['maxrefmag']) &
                              (galaxy['refmag'] > zredstr['minrefmag']) &
                              (zredstr['z'] < 100.0))

    if zbins_limited.size < 2:
        _reset_bad_values(galaxy)
        return

    neighbors = 10
    zbins = np.arange(*np.clip([zbins_limited[0] - neighbors,
                                zbins_limited[-1] + neighbors],
                               0, nz))

    lndist[zbins], chisq[zbins] = _calculate_lndist(zredstr, galaxy, zbins)

    # move from log space to regular space
    maxlndist = np.max(lndist[zbins])
    dist = np.zeros_like(lndist)
    with np.errstate(invalid='ignore', over='ignore'):
        dist[zbins] = np.exp(lndist[zbins] - maxlndist)

    # fix infinities and NaNs
    bad, = np.where(~np.isfinite(dist))
    dist[bad] = 0.0

    # take the maximum where not extrapolated
    ind_temp = np.argmax(dist[notextrap])
    ind = notextrap[ind_temp]

    calcinds, = np.where(dist > 1e-5)

    if calcinds.size >= 3:
        tdist = scipy.integrate.trapezoid(dist[calcinds], zredstr['z'][calcinds])
        zred_temp = scipy.integrate.trapezoid(dist[calcinds] * zredstr['z'][calcinds],
                                          zredstr['z'][calcinds]) / tdist
        zred_e = scipy.integrate.trapezoid(dist[calcinds] * zredstr['z'][calcinds]**2.,
                                       zredstr['z'][calcinds]) / tdist - zred_temp**2.
    else:
        tdist = np.sum(dist[calcinds])
        zred_temp = np.sum(dist[calcinds] * zredstr['z'][calcinds]) / tdist
        zred_e = np.sum(dist[calcinds] * zredstr['z'][calcinds]**2.) / tdist - zred_temp**2.

    if zred_e < 0.0:
        zred_e = 1.0
    else:
        zred_e = np.sqrt(zred_e)

    zred_e = zred_e if zred_e > 0.005 else 0.005

    # Now fit a parabola to get the perfect zred
    ind_temp = np.argmax(dist[notextrap])
    ind = notextrap[ind_temp]

    zred = zred_temp.copy()

    neighbors = 2
    use, = np.where(lndist > -1e10)
    if use.size >= neighbors * 2 + 1:
        # If it hits a wall, then move in the other direction to ensure we have at least neighbors*2+1 points
        minuse = use.min()
        maxuse = use.max()

        minindex = minuse if minuse > ind - neighbors else ind - neighbors
        maxindex = maxuse if maxuse < ind + neighbors else ind + neighbors
        if minindex == minuse:
            maxindex = np.clip(minuse + 2 * neighbors, None, maxuse)
        elif maxindex == maxuse:
            minindex = np.clip(maxuse - 1 - 2*neighbors, minuse, None)

        if ((maxindex - minindex + 1) >= 5):
            X = np.zeros((maxindex - minindex + 1, 3))
            X[:, 1] = zredstr['z'][minindex:maxindex + 1]
            X[:, 0] = X[:, 1] * X[:, 1]
            X[:, 2] = 1
            y = lndist[minindex: maxindex + 1]

            fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

            if fit[0] < 0.0:
                ztry = -fit[1] / (2.0 * fit[0])
                # Don't let it move too far, or it's a bad fit
                if (np.abs(ztry - zred) < 2.0*zred_e):
                    zred = ztry

    # And compute values at the real zred peak
    x = (zredstr['z'] - zred) / zred_e
    newdist = np.exp(-0.5 * x * x)

    bad, = np.where((lndist < -1e10) | (~np.isfinite(lndist)))
    newdist[bad] = 0.0
    lndist[bad] = -1e11

    if calcinds.size >= 3:
        # Note there maybe should be a distcorr here, but this is not
        #  actually computed in the IDL code (bug?)
        lkhd = scipy.integrate.trapezoid(newdist[calcinds] * (lndist[calcinds]), zredstr['z'][calcinds]) / scipy.integrate.trapezoid(newdist[calcinds], zredstr['z'][calcinds])
    else:
        lkhd = np.sum(newdist[calcinds] * lndist[calcinds]) / np.sum(newdist[calcinds])

    # Get chisq at the closest bin position
    zbin = np.argmin(np.abs(zred - zredstr['z']))
    chisq_val = chisq[zbin]

    if not np.isfinite(lkhd):
        _reset_bad_values(galaxy)
        return

    # And sample the uncorrected p(z)
    gdzbins, = np.where((dist > 1e-10) & (np.isfinite(dist)))

    if gdzbins.size < 3:
        # We cannot do a proper p(z)
        zred_samp = np.zeros(galaxy['zred_samp'].size) + zred
    else:
        pz = dist.copy()
        n = scipy.integrate.simpson(y=pz[gdzbins], x=zredstr['z'][gdzbins])
        pz /= n

        pdf = scipy.interpolate.interp1d(zredstr['z'][gdzbins], pz[gdzbins], kind='quadratic',
                                         bounds_error=False, fill_value=0.0)
        zred_samp = sample_from_pdf(pdf,
                                    [zredstr['z'][gdzbins[0]], zredstr['z'][gdzbins[-1]]],
                                    0.0001,
                                    galaxy['zred_samp'].size,
                                    rng)

    galaxy['zred_samp'] = zred_samp
    galaxy['chisq'] = chisq_val
    galaxy['lkhd'] = lkhd

    zred2 = zred
    zred2_e = zred_e
    zred_uncorr = zred
    zred_uncorr_e = zred_e

    if do_correction and not no_corrections:
        olddz = -1.0
        dz = 0.0
        iteration = 0

        pivotmag = interpol(zredstr['pivotmag'], zredstr['z'], zred)

        while np.abs(olddz - dz) > 1e-3 and iteration < 10:
            olddz = copy.copy(dz)
            dz = interpol(zredstr['corr'], zredstr['z'], zred + olddz) + (galaxy['refmag'] - pivotmag) * interpol(zredstr['corr_slope'], zredstr['z'], zred + olddz)
            iteration += 1

        zred = zred + dz

        # evaluate error correction at "z_true"
        zred_e *= interpol(zredstr['corr_r'], zredstr['z'], zred)

        # And the zred2 correction
        dz = interpol(zredstr['corr2'], zredstr['z'], zred2) + (galaxy['refmag'] - pivotmag) * (interpol(zredstr['corr2_slope'], zredstr['z'], zred2))
        # this is evaluated at zred0
        r2 = interpol(zredstr['corr2_r'], zredstr['z'], zred2)

        zred2 += dz
        zred2_e *= r2

    # Finally store the values

    galaxy['zred'] = zred
    galaxy['zred_e'] = zred_e
    galaxy['zred2'] = zred2
    galaxy['zred2_e'] = zred2_e
    galaxy['zred_uncorr'] = zred_uncorr
    galaxy['zred_uncorr_e'] = zred_uncorr_e

def _calculate_lndist(zredstr, galaxy, zbins):
    """
    Calculate the log-likelihood for a list of redshift bins.

    Parameters
    ----------
    zredstr: `dict`
       Red sequence parametrization
    galaxy: `astropy.table.Row`
       Galaxy to compute likelihoods
    zbins: `np.array`
       Integer array of redshift bins

    Returns
    -------
    lndist: `np.array`
       Log-likelihood for the redshift bins
    chisq: `np.array`
       Fit chi-squared for the redshift bins
    """

    # Note we need to deal with photoerr...
    if zbins.size > 1:
        # we have many bins...
        chisq = compute_redsequence_chisq_redshifts(zredstr, galaxy, zbins, z_is_index=True, calc_lkhd=False)
    else:
        # we have a single bin... hack this
        chisq = compute_redsequence_chisq(zredstr, galaxy, np.array([zbins[0], zbins[0]]), z_is_index=True, calc_lkhd=False)[0]

    lndist = -0.5 * chisq

    #with np.errstate(invalid='ignore'):
    lndistcorr = np.log((10.**(0.4 * (zredstr['alpha'] + 1.0) *
                               (zredstr['_mstar'][zbins] - galaxy['refmag'])) *
                         np.exp(-10.**(0.4 * (zredstr['_mstar'][zbins] - galaxy['refmag'])))) *
                        zredstr['volume_factor'][zbins])

    lndist += lndistcorr

    bad, = np.where(~np.isfinite(lndist))
    lndist[bad] = -1e11

    return (lndist, chisq)

def _reset_bad_values(galaxy):
    """
    Reset all galaxy zred values to "bad" (-1s)

    Parameters
    ----------
    galaxy: `astropy.table.Row`
       Galaxy to set zred values.
    """

    galaxy['lkhd'] = -1000.0
    galaxy['zred'] = -1.0
    galaxy['zred_e'] = -1.0
    galaxy['zred2'] = -1.0
    galaxy['zred2_e'] = -1.0
    galaxy['zred_uncorr'] = -1.0
    galaxy['zred_uncorr_e'] = -1.0
    galaxy['chisq'] = -1.0
