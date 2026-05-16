import os
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from ..core.io import read_fits_catalog, write_fits_catalog
from ..utilities import make_nodes, cubic_spline_compute_y2, cubic_spline_interpolate, interpol, read_redgal_initial_colors, get_redgal_initial_color
from ..fitters import fit_med_z, fit_red_sequence, fit_ecgmm
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog, Entry


def select_spec_seeds(config, usetrain=True, logger=None):
    """
    Pure-ish function to match a galaxy catalog to a spectroscopic catalog to
    create seeds for cluster training.

    Parameters
    ----------
    config: dict-like
        Configuration dictionary.
    usetrain: bool, optional
        Use spectra from specfile_train rather than specfile. Default is True.
    logger: logging.Logger, optional
        Logger instance.
    """
    if logger:
        logger.info("Selecting spectroscopic seeds...")

    # 1. Read in the galaxies, check for zreds
    zredfile = config.get("zredfile")
    if zredfile is not None and not os.path.isfile(zredfile):
        zredfile = None

    has_zreds = zredfile is not None

    hpix = config.get("hpix", [])
    nside = config.get("nside", 0)
    border = config.get("border", 0.0)

    gals = GalaxyCatalog.from_galfile(
        config["galfile"], nside=nside, hpix=hpix, border=border, zredfile=zredfile
    )

    # 2. Read in the spectroscopic catalog
    if usetrain:
        specfile = config["specfile_train"]
    else:
        specfile = config["specfile"]

    spec = Catalog.from_fits_file(specfile)

    # 3. Limit the redshift range
    from ..configuration import get_zrange_cushioned
    zrange_cushioned = get_zrange_cushioned(config)
    calib_spec_max_zerr = config.get("calib_spec_max_zerr", 0.001)

    # 4. Select good spectra
    (use,) = np.where(
        (spec.z >= zrange_cushioned[0])
        & (spec.z <= zrange_cushioned[1])
        & (spec.z_err < calib_spec_max_zerr)
    )
    spec = spec[use]

    # 5. Match spectra to galaxies
    i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3.0 / 3600.0, maxmatch=1)

    gals = gals[i1]
    spec = spec[i0]

    # 6. Ensure it has a valid zred
    if has_zreds:
        (use,) = np.where(gals.zred > 0.0)
    else:
        (use,) = np.where(gals.refmag > 0.0)

    # 7. Create output catalog
    nmag = config["nmag"]
    cat = Catalog(
        np.zeros(
            use.size,
            dtype=[
                ("ra", "f8"),
                ("dec", "f8"),
                ("mag", "f4", nmag),
                ("mag_err", "f4", nmag),
                ("refmag", "f4"),
                ("refmag_err", "f4"),
                ("zred", "f4"),
                ("zred_e", "f4"),
                ("zred_chisq", "f4"),
                ("zspec", "f4"),
                ("ebv", "f4"),
            ],
        )
    )
    cat.ra[:] = gals.ra[use]
    cat.dec[:] = gals.dec[use]
    cat.mag[:, :] = gals.mag[use, :]
    cat.mag_err[:, :] = gals.mag_err[use, :]
    cat.refmag[:] = gals.refmag[use]
    cat.refmag_err[:] = gals.refmag_err[use]

    if has_zreds:
        cat.zred[:] = gals.zred[use]
        cat.zred_e[:] = gals.zred_e[use]
        cat.zred_chisq[:] = gals.chisq[use]
    else:
        cat.zred[:] = -1.0
        cat.zred_e[:] = -1.0
        cat.zred_chisq[:] = -1.0

    cat.zspec[:] = spec.z[use]
    cat.ebv[:] = gals.ebv[use]

    if logger:
        logger.info(f"Saving {cat.size} seeds to {config['seedfile']}")

    cat.to_fits_file(config["seedfile"])
    return config["seedfile"]


def select_spec_red_galaxies(config, logger=None):

    """
    Pure-ish function to select red galaxies from a spectroscopic catalog.

    Parameters
    ----------
    config: dict-like
        Configuration dictionary.
    logger: logging.Logger, optional
        Logger instance.
    """
    # 1. Read galaxies
    # For now, still using GalaxyCatalog.from_galfile as it's complex
    # but passing parameters from the config dict.
    hpix = config.get("hpix", [])
    nside = config.get("nside", 0)
    border = config.get("border", 0.0)

    gals = GalaxyCatalog.from_galfile(
        config["galfile"], nside=nside, hpix=hpix, border=border
    )

    # 2. Read spectra
    spec = Catalog.from_fits_file(config["specfile_train"])

    # 3. Select good spectra
    calib_spec_max_zerr = config.get("calib_spec_max_zerr", 0.001)
    (use,) = np.where(spec.z_err < calib_spec_max_zerr)
    spec = spec[use]

    # 4. Match spectra to galaxies
    i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3.0 / 3600.0, maxmatch=1)

    # 5. Make a specific galaxy table
    gals = gals[i1]
    gals.add_fields([("z", "f4")])
    gals.z = spec[i0].z

    if logger:
        logger.info("Calibrating with %d galaxies with spectra" % (gals.size))

    # 6. Set the redshift range
    from ..configuration import get_zrange_cushioned
    zrange_cushioned = get_zrange_cushioned(config)
    zrange = config.get("zrange")

    limmag_ref = config.get("limmag_ref")
    (use,) = np.where(
        (gals.z > zrange_cushioned[0])
        & (gals.z < zrange_cushioned[1])
        & (gals.refmag < limmag_ref)
    )
    gals = gals[use]

    galcolor = gals.galcol
    galcolor_err = gals.galcol_err
    ncol = config["nmag"] - 1
    bands = config["bands"]

    # 7. Make nodes
    calib_pivotmag_nodesize = config.get("calib_pivotmag_nodesize", 0.1)
    nodes = make_nodes(zrange_cushioned, calib_pivotmag_nodesize)

    # 8. Initialization for fitting
    medcol = np.zeros((nodes.size, ncol))
    medcol_width = np.zeros_like(medcol)
    meancol = np.zeros_like(medcol)
    meancol_scatter = np.zeros_like(meancol)

    redgal_data = read_redgal_initial_colors(config["calib_redgal_template"])
    calib_colormem_colormodes = np.atleast_1d(config["calib_colormem_colormodes"])
    nmodes = calib_colormem_colormodes.size

    # 9. Loop over modes (colors)
    for m in range(nmodes):
        j = calib_colormem_colormodes[m]

        if logger:
            logger.info("Working on color %d" % (j))

        c = get_redgal_initial_color(redgal_data, bands[j], bands[j + 1], gals.z)
        delta = galcolor[:, j] - c

        st = np.argsort(delta)
        delta5 = delta[st[int(0.05 * delta.size)]]
        delta99 = delta[st[int(0.99 * delta.size)]]

        (u,) = np.where((delta > delta5) & (delta < delta99))

        wt, mu, sigma = fit_ecgmm(delta[u], galcolor_err[u, j], [0.2], [-0.5, 0.0], [0.2, 0.05], offset=2.0)

        mvals = get_redgal_initial_color(redgal_data, bands[j], bands[j + 1], nodes) + mu[1]
        scvals = np.zeros(nodes.size) + sigma[1]

        # Fit median and median-width
        y2_sc = cubic_spline_compute_y2(nodes, scvals)
        width = cubic_spline_interpolate(gals.z, nodes, scvals, y2_sc)
        y2_m = cubic_spline_compute_y2(nodes, mvals)
        med = cubic_spline_interpolate(gals.z, nodes, mvals, y2_m)
        (u,) = np.where(
            (galcolor[:, j] > (med - 2.0 * width))
            & (galcolor[:, j] < (med + 2.0 * width))
        )

        mvals = fit_med_z(nodes, gals.z[u], galcolor[u, j], mvals)

        y2_m = cubic_spline_compute_y2(nodes, mvals)
        med = cubic_spline_interpolate(gals.z, nodes, mvals, y2_m)
        (u,) = np.where(
            (galcolor[:, j] > (med - 2.0 * width))
            & (galcolor[:, j] < (med + 2.0 * width))
        )

        scvals = fit_med_z(nodes, gals.z[u], np.abs(galcolor[u, j] - med[u]), scvals)

        medcol[:, j] = mvals
        medcol_width[:, j] = 1.4826 * scvals

        # Fit mean and scatter
        y2_med = cubic_spline_compute_y2(nodes, medcol[:, j])
        med = cubic_spline_interpolate(gals.z, nodes, medcol[:, j], y2_med)
        y2_width = cubic_spline_compute_y2(nodes, medcol_width[:, j])
        width = cubic_spline_interpolate(gals.z, nodes, medcol_width[:, j], y2_width)

        nsig = 1.5
        (u,) = np.where(np.abs(galcolor[:, j] - med) < nsig * width)

        trunc = nsig * width[u]
        mag_err = gals.mag_err[u, j : j + 2]

        mvals_list = fit_red_sequence(
            nodes,
            gals.z[u],
            galcolor[u, j],
            mag_err,
            medcol[:, j], np.zeros_like(mvals), medcol_width[:, j],
            fit_mean=True,
            trunc=trunc,
            use_scatter_prior=False,
        )
        mvals = mvals_list[0]

        scvals_list = fit_red_sequence(
            nodes,
            gals.z[u],
            galcolor[u, j],
            mag_err,
            mvals, np.zeros_like(mvals), medcol_width[:, j],
            fit_scatter=True,
            trunc=trunc,
            use_scatter_prior=False,
        )
        scvals = scvals_list[0]

        mvals, scvals = fit_red_sequence(
            nodes,
            gals.z[u],
            galcolor[u, j],
            mag_err,
            mvals, np.zeros_like(mvals), scvals,
            fit_mean=True, fit_scatter=True,
            trunc=trunc,
            use_scatter_prior=False,
        )


        meancol[:, j] = mvals
        meancol_scatter[:, j] = scvals

    # 10. Select red galaxies
    calib_colormem_zbounds = np.atleast_1d(config["calib_colormem_zbounds"])
    zbounds = np.concatenate(
        [
            np.array([zrange_cushioned[0] - 0.011]),
            calib_colormem_zbounds,
            np.array([zrange_cushioned[1] + 0.011]),
        ]
    )

    mark = np.zeros(gals.size, dtype=bool)
    calib_redspec_nsig = config.get("calib_redspec_nsig", 2.0)

    for m in range(nmodes):
        (u,) = np.where((gals.z > zbounds[m]) & (gals.z < zbounds[m + 1]))
        j = calib_colormem_colormodes[m]

        if u.size > 0:
            y2_mean = cubic_spline_compute_y2(nodes, meancol[:, j])
            mn = cubic_spline_interpolate(gals.z[u], nodes, meancol[:, j], y2_mean)
            y2_scatter = cubic_spline_compute_y2(nodes, meancol_scatter[:, j])
            sc = cubic_spline_interpolate(gals.z[u], nodes, meancol_scatter[:, j], y2_scatter)

            (gd,) = np.where(
                (
                    np.abs(galcolor[u, j] - mn)
                    < calib_redspec_nsig * np.sqrt(galcolor_err[u, j] ** 2.0 + sc**2.0)
                )
            )
            mark[u[gd]] = True

    # 11. Output results
    (use,) = np.where(mark)
    redgalfile = config["redgalfile"]
    gals.to_fits_file(redgalfile, indices=use, clobber=True)

    redgalmodelfile = config["redgalmodelfile"]
    model = Entry(
        np.zeros(
            1,
            dtype=[
                ("nodes", "f4", nodes.size),
                ("meancol", "f4", meancol.shape),
                ("meancol_scatter", "f4", meancol_scatter.shape),
                ("medcol", "f4", medcol.shape),
                ("medcol_width", "f4", medcol_width.shape),
            ],
        )
    )
    model.nodes = nodes
    model.meancol = meancol
    model.meancol_scatter = meancol_scatter
    model.medcol = medcol
    model.medcol_width = medcol_width
    model.to_fits_file(redgalmodelfile, clobber=True)

    # 12. Plotting (Functional-ish)
    plotpath = config.get("plotpath", "")
    outpath = config.get("outpath", "./")
    outbase = config.get("outbase")

    for m in range(nmodes):
        j = calib_colormem_colormodes[m]

        # ... (plotting code from selectspecred.py adapted to use config dict)
        # I'll keep it simple for now to avoid too much clutter,
        # but I should implement at least one plot to match functionality.

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        (not_use,) = np.where(~mark)
        fit_result_color = "#1f77b4"

        ax.scatter(
            gals.z[not_use],
            galcolor[not_use, j],
            c="#cccccc",
            marker="o",
            s=4,
            edgecolors="none",
            alpha=0.5,
            label="Rejected",
        )
        ax.scatter(
            gals.z[use],
            galcolor[use, j],
            c="#ff7f0e",
            marker="o",
            s=4,
            edgecolors="none",
            alpha=0.5,
            label="Selected",
        )

        xvals = np.arange(zrange_cushioned[0], zrange_cushioned[1], 0.01)
        y2_mean = cubic_spline_compute_y2(nodes, meancol[:, j])
        mean_model = cubic_spline_interpolate(xvals, nodes, meancol[:, j], y2_mean)
        y2_scatter = cubic_spline_compute_y2(nodes, meancol_scatter[:, j])
        scatter_vals = cubic_spline_interpolate(xvals, nodes, meancol_scatter[:, j], y2_scatter)
        nsig = calib_redspec_nsig

        ax.fill_between(
            xvals,
            mean_model - nsig * scatter_vals,
            mean_model + nsig * scatter_vals,
            color=fit_result_color,
            alpha=0.15,
            edgecolor="none",
            label=rf"{nsig}$\sigma$ Region",
        )
        ref_vals = get_redgal_initial_color(redgal_data, bands[j], bands[j + 1], xvals)
        ax.plot(
            xvals,
            ref_vals,
            color=fit_result_color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Initial Template",
        )
        ax.plot(
            xvals,
            mean_model,
            color=fit_result_color,
            lw=2.5,
            linestyle="-",
            label="Fitted Mean Model",
        )
        ax.scatter(
            nodes,
            meancol[:, j],
            color=fit_result_color,
            marker="X",
            s=60,
            label="Nodes",
        )

        ax.set_xlim(zrange)
        ax.set_xlabel(r"$z_{\mathrm{spec}}$")
        ax.set_ylabel(f"{bands[j]} - {bands[j + 1]}")
        ax.legend()

        plot_filename = os.path.join(
            outpath,
            plotpath,
            f"{outbase}_redgals_selection_{bands[j]}-{bands[j + 1]}.png",
        )
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        fig.savefig(plot_filename, bbox_inches="tight")
        plt.close(fig)

    return redgalfile, redgalmodelfile
