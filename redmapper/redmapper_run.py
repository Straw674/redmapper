"""Class for performing a local redmapper run with multiprocessing.

This class is typically used during training.
"""
import numpy as np
import copy
import fitsio
import re
import os
import esutil
import hpgeom as hpg
from esutil.cosmology import Cosmo

import multiprocessing

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

from .utilities import _pickle_method

from .catalog import Catalog, Entry
from .run_firstpass import RunFirstPass
from .run_likelihoods import RunLikelihoods
from .run_percolation import RunPercolation
from .utilities import getMemoryString

class RedmapperRun(object):
    """
    Class to run various stages of the redmapper finder using multiprocessing.

    This is typically run during the training.  In production runs should be
    farmed out to a compute cluster.
    """

    def __init__(self, config):
        """
        Instantiate a RedmapperRun object.

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        """
        self.config = config.copy()

        # Record the cosmology parameters because we need to rebuild the object
        # in the multiprocessing.
        self._H0 = config.cosmo.H0()
        self._omega_l = config.cosmo.omega_l()
        self._omega_m = config.cosmo.omega_m()
        self.config.cosmo = None

    def run(self, specmode=False, seedfile=None, check=True,
            percolation_only=False, consolidate_like=False, keepz=False, cleaninput=False,
            consolidate=True):
        """
        Run the redmapper cluster finder using multiprocessing.

        Parameters
        ----------
        specmode: `bool`, optional
           Run with spectroscopic mode (firstpass uses zspec as seeds).
           Default is False.
        seedfile: `str`, optional
           File containing spectroscopic seeds.  Default is None.
        check: `bool`, optional
           Check if files already exist, and skip if so.  Default is True.
        percolation_only: `bool`, optional
           Only run the percolation phase.  Default is False.
        consolidate_like: `bool`, optional
           Consolidate the pixel runs for the likelihood files?  Default is False.
        keepz: `bool`, optional
           Keep input redshifts or replace with z_lambda?  Default is False.
        cleaninput: `bool`, optional
           Processing stage should clean out bad clusters?  Default is False.
        consolidate: `bool`, optional
           Consolidate the pixel runs for the percolated files?  Default is True.

        Returns
        -------
        finalfile: `str`
           Filename for the final consolidated percolation file.
        likefile: `str`
           Filename for the final consolidated likelihood file
           (if consolidate_like == True)
        """
        self.config.start_file_logging()

        self.specmode = specmode
        self.seedfile = seedfile
        self.check = check
        self.percolation_only = percolation_only
        self.keepz = keepz
        self.cleaninput = cleaninput

        if self.specmode and not self.keepz:
            raise RuntimeError("Must set keepz=True when specmode=True")
        if self.percolation_only and self.specmode:
            raise RuntimeError("Cannot set both percolation_only=True and specmode=True")

        nside_split, pixels_split = self._get_pixel_splits()

        self.config.logger.info("Running on %d pixels" % (len(pixels_split)))

        # run each individual one
        nside_orig = self.config.d.nside
        self.config.d.nside = nside_split

        orig_seedfile = self.config.seedfile
        if seedfile is not None:
            # Use the specified seedfile if desired
            self.config.seedfile = seedfile
        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=self.config.calib_run_nproc)
        if self.percolation_only:
            retvals = pool.map(self._percolation_only_worker, pixels_split, chunksize=1)
            #retvals = list(map(self._percolation_only_worker, pixels_split))
        else:
            retvals = pool.map(self._worker, pixels_split, chunksize=1)
            #retvals = list(map(self._worker, pixels_split))
        pool.close()
        pool.join()

        # Reset the seedfile
        self.config.seedfile = orig_seedfile

        # Consolidate (adds additional mask cuts)
        hpixels_like = [x[0] for x in retvals if x[2] is not None]
        likefiles = [x[2] for x in retvals if x[2] is not None]
        hpixels_perc = [x[0] for x in retvals if x[3] is not None]
        percfiles = [x[3] for x in retvals if x[3] is not None]

        # Allow for runs without consolidation
        if consolidate:
            finalfile = self._consolidate(hpixels_perc, percfiles, 'final', members=True, check=check)
        else:
            finalfile = None

        if consolidate_like:
            likefile = self._consolidate(hpixels_like, likefiles, 'like', members=False, check=check)

        # Reset the nside in the config file
        self.config.d.nside = nside_orig

        # Generate QA plots if requested
        if hasattr(self.config, 'more_qa_plots') and self.config.more_qa_plots and consolidate:
            self._make_qa_plots(finalfile)

        # And done
        if consolidate_like:
            return (finalfile, likefile)
        else:
            return finalfile

    def _make_qa_plots(self, finalfile):
        """
        Generate QA plots for the redmapper run.

        Parameters
        ----------
        finalfile: `str`
           Path to the final consolidated catalog file
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.config.logger.warning("matplotlib not available, skipping QA plots")
            return

        if not hasattr(self.config, 'plotpath') or self.config.plotpath is None:
            self.config.logger.warning("plotpath not set in config, skipping QA plots")
            return

        if not os.path.exists(self.config.plotpath):
            os.makedirs(self.config.plotpath)

        if finalfile is None or not os.path.isfile(finalfile):
            self.config.logger.warning("Final catalog file not found, skipping QA plots")
            return

        self.config.logger.info("Generating redmapper run QA plots...")

        # Read the catalog
        cat = Catalog.from_fits_file(finalfile)

        if cat.size == 0:
            self.config.logger.warning("Empty catalog, skipping QA plots")
            return

        # Plot 1: Lambda distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cat.Lambda, bins=50, alpha=0.7, edgecolor='black', range=(0, 200))
        ax.set_xlabel(r'$\lambda$ (Richness)', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Cluster Richness Distribution', fontsize=14)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_lambda_dist.png'), dpi=300)
        plt.close()

        # Plot 2: Redshift distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(cat.z_lambda, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(r'$z_\lambda$ (Cluster Redshift)', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Cluster Redshift Distribution', fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_z_dist.png'), dpi=300)
        plt.close()

        # Plot 3: Lambda vs z scatter plot

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hexbin(cat.z_lambda, cat.Lambda, gridsize=(50, 50), cmap='Reds', mincnt=1, yscale='log')
        ax.set_xlabel(r'$z_\lambda$ (Cluster Redshift)', fontsize=12)
        ax.set_ylabel(r'$\lambda$ (Richness)', fontsize=12)
        ax.set_title('Richness vs Redshift', fontsize=14)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Number of Clusters', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_lambda_vs_z.png'), dpi=300)
        plt.close()

        # Plot 4: Spatial distribution
        # Calculate RA and Dec ranges to set aspect ratio = 1
        ra_range = cat.ra.max() - cat.ra.min()
        dec_range = cat.dec.max() - cat.dec.min()
        aspect_ratio = ra_range / dec_range if dec_range > 0 else 1.0
        fig_height = 8
        fig_width = fig_height * aspect_ratio
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        scatter = ax.scatter(cat.ra, cat.dec, c=cat.z_lambda, s=10, alpha=0.6, cmap='Reds')
        ax.set_xlabel('RA (degrees)', fontsize=12)
        ax.set_ylabel('Dec (degrees)', fontsize=12)
        ax.set_title('Spatial Distribution of Clusters (colored by redshift)', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_xaxis()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$z_\lambda$', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_spatial_dist.png'), dpi=300)
        plt.close()

        # Plot 5: Redshift error distribution
        if hasattr(cat, 'z_lambda_e'):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(cat.z_lambda_e, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(r'$\sigma_{z_\lambda}$ (Redshift Error)', fontsize=12)
            ax.set_ylabel('Number of Clusters', fontsize=12)
            ax.set_title('Cluster Redshift Error Distribution', fontsize=14)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_z_err_dist.png'), dpi=300)
            plt.close()

        # Plot 6: Maskfrac distribution
        if hasattr(cat, 'maskfrac'):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(cat.maskfrac, bins=50, alpha=0.7, edgecolor='black', range=(0, 1))
            ax.set_xlabel('Mask Fraction', fontsize=12)
            ax.set_ylabel('Number of Clusters', fontsize=12)
            ax.set_title('Mask Fraction Distribution', fontsize=14)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plotpath, f'{self.config.d.outbase}_maskfrac_dist.png'), dpi=300)
            plt.close()

        self.config.logger.info("QA plots saved to %s" % self.config.plotpath)

    def _get_pixel_splits(self):
        """
        Get the subpixels on which to run to optimally split the input catalog
        based on the number of cores for the run.

        Returns
        -------
        nside_split: `int`
           Healpix nside for the split pixels
        pixels_split: `list`
           Integer list of healpix pixel numbers (ring format)
        """

        # need to redo logic here.  Dang.
        tab = Entry.from_fits_file(self.config.galfile, ext=1)

        if self.config.calib_run_nproc == 1:
            if self.config.d.nside > self.config.calib_run_min_nside:
                nside_test = self.config.run_min_nside
            else:
                nside_test = np.clip(self.config.d.nside, 1, None)
            subpixels = self._get_subpixels(nside_test, tab)
            return (nside_test, subpixels)

        # start with the pixel and resolution in the config file
        if len(self.config.d.hpix) == 0:
            nside_splits = [self.config.calib_run_min_nside]
            pixels_splits = [self._get_subpixels(nside_splits[0], tab)]
        else:
            if self.config.d.nside > self.config.calib_run_min_nside:
                nside_splits = [self.config.d.nside]
            else:
                nside_splits = [self.config.calib_run_min_nside]
            pixels_splits = [self._get_subpixels(nside_splits[0], tab)]

        nsplit = [len(pixels_splits[0])]

        nside_test = nside_splits[0]
        while nside_test < tab.nside:
            # increment nside_test
            nside_test *= 2

            pixels_test = self._get_subpixels(nside_test, tab)
            nsplit.append(pixels_test.size)
            nside_splits.append(nside_test)
            pixels_splits.append(pixels_test)

        test, = np.where(np.array(nsplit) <= self.config.calib_run_nproc*2)
        if test.size == 0:
            nside_split = nside_splits[0]
            pixels_split = pixels_splits[0]
        else:
            nside_split = nside_splits[test[-1]]
            pixels_split = pixels_splits[test[-1]]

        return (nside_split, pixels_split)

    def _get_subpixels(self, nside_test, galtab):
        """
        Get all the pixels from a galaxy table corresponding to a given nside.

        Note that this takes into account the subregion of galtab that is being
        processed.

        Parameters
        ----------
        nside_test: `int`
           Nside of the desired grouping of galtab.
        galtab: `redmapper.Entry`
           Galaxy table summary information.

        Returns
        -------
        pixels: `np.array`
           Integer array of pixels that cover galtab.
        """

        # generate all the pixels
        pixels = np.arange(hpg.nside_to_npixel(int(nside_test)))

        # Which of these match the parent?
        if len(self.config.d.hpix) > 0:
            theta, phi = hpg.pixel_to_angle(nside_test, pixels, lonlat=False, nest=False)
            hpix_test = hpg.angle_to_pixel(self.config.d.nside, theta, phi, lonlat=False, nest=False)
            a, b = esutil.numpy_util.match(self.config.d.hpix, hpix_test)
            pixels = pixels[b]

        # And which match the galaxies?
        theta, phi = hpg.pixel_to_angle(galtab.nside, galtab.hpix, lonlat=False, nest=False)
        hpix_test = hpg.angle_to_pixel(nside_test, theta, phi, lonlat=False, nest=False)
        a, b = esutil.numpy_util.match(pixels, hpix_test)

        return np.unique(pixels[a])

    def _consolidate(self, hpixels, filenames, filetype, members=False, check=True):
        """
        Consolidate pixel run files.

        Parameters
        ----------
        hpixels: `np.array`
           Integer array of healpix pixels to consolidate
        filenames: `list`
           List of strings of filenames to consolidate
        filetype: `str`
           Type of file (final, like)
        members: `bool`, optional
           Consolidte members as well?  Default is False.
        check: `bool`, optional
           Check to see if consolidated files exist (and exit if so).
           Default is False

        Returns
        -------
        outfile: `str`
           Output filename
        """

        outfile = self.config.redmapper_filename(filetype)
        memfile = self.config.redmapper_filename(filetype+'_members')

        if check:
            outfile_there = os.path.isfile(outfile)
            memfile_there = os.path.isfile(memfile)

            if (outfile_there and memfile_there and members):
                # All files are accounted for
                return outfile
            if outfile_there and not members:
                return outfile

        # How many clusters are there?  (This is the maxmimum before cuts)
        ncluster = 0
        for f in filenames:
            hdr = fitsio.read_header(f, ext=1)
            ncluster += hdr['NAXIS2']

        element = Entry.from_fits_file(filenames[0], ext=1, rows=0)
        dtype = element._ndarray.dtype

        ubercat = Catalog(np.zeros(ncluster, dtype=dtype))
        ctr = 0

        ubermem = None

        for hpix, f in zip(hpixels, filenames):
            cat = Catalog.from_fits_file(f, ext=1)

            # Cut to minlambda, maxfrac, and within a pixel
            if self.config.d.nside > 0:
                ipring = hpg.angle_to_pixel(self.config.d.nside, cat.ra, cat.dec, nest=False)
            else:
                # Set all the pixels to 0
                ipring = np.zeros(cat.size, dtype=np.int32)

            use, = np.where((ipring == hpix) &
                            (cat.maskfrac < self.config.max_maskfrac) &
                            (cat.Lambda / cat.scaleval > self.config.percolation_minlambda))

            # Make sure we have surviving clusters
            if use.size == 0:
                continue

            cat = cat[use]

            if members:
                parts = f.split('.fit')
                mem = Catalog.from_fits_file(parts[0] + '_members.fit')

                # We are going to replace the mem_match_ids in the consolidated catalog,
                # because the ones generated in the pixels aren't unique
                new_ids = np.arange(use.size, dtype=np.int32) + ctr + 1

                a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)
                cat.mem_match_id = new_ids
                mem.mem_match_id[b] = cat.mem_match_id[a]

                # and we only want to store the members that matched!
                mem = mem[b]

            # And copy the fields
            for n in dtype.names:
                ubercat._ndarray[n][ctr: ctr + cat.size] = cat._ndarray[n]

            if members:
                if ubermem is None:
                    ubermem = mem
                else:
                    ubermem.append(mem)

            ctr += cat.size

        # Crop the catalog to the range that we had clusters
        ubercat = ubercat[0:ctr]

        # Now we need a final sorting by likelihood and mem_match_id replacement
        if members:
            st = np.argsort(ubercat.lnlike)[::-1]
            ubercat = ubercat[st]

            a, b = esutil.numpy_util.match(ubercat.mem_match_id, ubermem.mem_match_id)

            ubercat.mem_match_id = np.arange(ubercat.size, dtype=np.int32) + 1
            ubermem.mem_match_id[b] = ubercat.mem_match_id[a]

        # And write out...
        # We can clobber because if it was already there and we wanted to check
        # that already happened
        ubercat.to_fits_file(outfile, clobber=True)

        if members:
            ubermem.to_fits_file(memfile, clobber=True)

        return outfile

    def _worker(self, hpix):
        """
        Do the run on one pixel (for multiprocessing).

        Parameters
        ----------
        hpix: `int`
           Healpix ring pixel to run.

        Outputs
        -------
        hpix: `int`
           Healpix ring number that was run.
        firstpass_filename: `str`
           Filename for firstpass file.
        like_filename: `str`
           Filename for likelihood file
        perc_filename: `str`
           Filename for percolation file.
        """

        self.config.logger.info("Running on pixel %d" % (hpix))

        # Not sure what to do with cosmo...
        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        # Set the specific config stuff here
        config.d.hpix = [hpix]

        config.d.outbase = '%s_%d_%05d' % (self.config.d.outbase, self.config.d.nside, hpix)

        # Need to add checks about success, and whether a file was output
        #  (especially border pixels in sims)
        firstpass = RunFirstPass(config, specmode=self.specmode)

        if not os.path.isfile(firstpass.filename) or not self.check:
            firstpass.run(keepz=self.keepz, cleaninput=self.cleaninput)

            if (firstpass.cat is None or
                (firstpass.cat is not None and firstpass.cat.size == 0)):
                # We did not get a firstpass catalog
                self.config.logger.info("Did not produce a firstpass catalog for pixel %d" % (hpix))
                return (hpix, None, None, None)

            firstpass.output(savemembers=False, withversion=False, clobber=True)
        else:
            self.config.logger.info("Firstpass file %s already present.  Skipping..." % (firstpass.filename))

        firstpass_filename = firstpass.filename
        config.catfile = firstpass_filename

        # Clear out memory
        del firstpass

        like = RunLikelihoods(config)

        if not os.path.isfile(like.filename) or not self.check:
            like.run(keepz=self.keepz)

            if (like.cat is None or
                (like.cat is not None and like.cat.size == 0)):
                # We did not get a likelihood catalog
                self.config.logger.info("Did not produce a likelihood catalog for pixel %d" % (hpix))
                return (hpix, firstpass_filename, None, None)

            like.output(savemembers=False, withversion=False, clobber=True)
        else:
            self.config.logger.info("Likelihood file %s already present.  Skipping..." % (like.filename))

        like_filename = like.filename
        config.catfile = like_filename

        # Clear out memory
        del like

        perc = RunPercolation(config)

        if not os.path.isfile(perc.filename) or not self.check:
            perc.run(keepz=self.keepz)

            if (perc.cat is None or
                (perc.cat is not None and perc.cat.size == 0)):
                # We did not get a percolation catalog
                self.config.logger.info("Did not produce a percolation catalog for pixel %d" % (hpix))
                return (hpix, firstpass_filename, like_filename, None)

            perc.output(savemembers=True, withversion=False, clobber=True)
        else:
            self.config.logger.info("Percolation file %s already present.  Skipping..." % (perc.filename))

        perc_filename = perc.filename

        return (hpix, firstpass_filename, like_filename, perc_filename)

    def _percolation_only_worker(self, hpix):
        """
        Do a percolation only run on one pixel (for multiprocessing).

        Parameters
        ----------
        hpix: `int`
           Healpix ring pixel to run.

        Outputs
        -------
        hpix: `int`
           Healpix ring number that was run.
        firstpass_filename: `str`
           None
        like_filename: `str`
           None
        perc_filename: `str`
           Filename for percolation file.
        """

        self.config.logger.info("Running percolation on pixel %d" % (hpix))

        config = self.config.copy()
        config.cosmo = Cosmo(H0=self._H0, omega_l=self._omega_l, omega_m=self._omega_m)

        config.d.hpix = [hpix]

        config.d.outbase = '%s_%05d' % (self.config.d.outbase, hpix)

        perc = RunPercolation(config)
        if not os.path.isfile(perc.filename) or not self.check:
            perc.run(keepz=self.keepz, cleaninput=self.cleaninput)

            if (perc.cat is None or
                (perc.cat is not None and perc.cat.size == 0)):
                # we did not get a percolation catalog
                self.config.logger.info("Did not produce a percolation catalog for pixel %d" % (hpix))
                return (hpix, None, None, None)
            perc.output(savemembers=True, withversion=False, clobber=True)

        return (hpix, None, None, perc.filename)
