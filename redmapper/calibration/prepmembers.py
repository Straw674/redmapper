"""Classes related to preparing members for the next calibration iteration
"""
import os
import numpy as np
import esutil
import matplotlib.pyplot as plt

from ..catalog import Entry, Catalog
from ..galaxy import GalaxyCatalog
from ..utilities import read_members
from ..configuration import Configuration

class PrepMembers(object):
    """
    Class to prepare members for input to the next calibration iteration.
    """

    def __init__(self, conf, rng=None):
        """
        Instantiate a PrepMembers object.

        Parameters
        ----------
        conf: `str` or `redmapper.Configuration`
           Config filename or configuration object
        rng : `np.random.RandomState`, optional
            Random number generator.
        """
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

        if rng is None:
            rng = np.random.RandomState(self.config.randomseed)
        self.rng = rng

    def run(self, mode):
        """
        Run the member preparation.

        Output members are put into self.config.zmemfile.

        Parameters
        ----------
        mode: `str`
           May be "z_init": use initial cluster seed redshift as member redshift or
           may be "cg": use the most likely central spec_z as member redshift

        Raises
        ------
        RuntimeError: If mode is not "z_init" or "cg".
        """

        cat = Catalog.from_fits_file(self.config.catfile)

        if mode == 'z_init':
            cat_z = cat.z_init
        elif mode == 'cg':
            cat_z = cat.cg_spec_z
        else:
            raise RuntimeError("Unsupported mode %s" % (mode))

        mem = read_members(self.config.catfile)

        # Cut the clusters
        use, = np.where((cat.Lambda / cat.scaleval > self.config.calib_minlambda) &
                        (cat.scaleval > 0.0) &
                        (np.abs(cat_z - cat.z_lambda) < self.config.calib_zlambda_clean_nsig * cat.z_lambda_e))
        cat = cat[use]
        cat_z = cat_z[use]

        # Cut the members
        use, = np.where((mem.p * mem.theta_i * mem.theta_r > self.config.calib_pcut) |
                        (mem.pcol > self.config.calib_pcut))

        mem = mem[use]

        # Match cut clusters to members
        a, b = esutil.numpy_util.match(cat.mem_match_id, mem.mem_match_id)

        newmem = Catalog(np.zeros(b.size, dtype=[('z', 'f4'),
                                                 ('z_lambda', 'f4'),
                                                 ('p', 'f4'),
                                                 ('pcol', 'f4'),
                                                 ('central', 'i2'),
                                                 ('ra', 'f8'),
                                                 ('dec', 'f8'),
                                                 ('mag', 'f4', self.config.nmag),
                                                 ('mag_err', 'f4', self.config.nmag),
                                                 ('refmag', 'f4'),
                                                 ('refmag_err', 'f4'),
                                                 ('ebv', 'f4')]))

        newmem.ra[:] = mem.ra[b]
        newmem.dec[:] = mem.dec[b]
        newmem.p[:] = mem.p[b]
        newmem.pcol[:] = mem.pcol[b]
        newmem.mag[:, :] = mem.mag[b, :]
        newmem.mag_err[:, :] = mem.mag_err[b, :]
        newmem.refmag[:] = mem.refmag[b]
        newmem.refmag_err[:] = mem.refmag_err[b]
        newmem.ebv[:] = mem.ebv[b]

        cent, = np.where(mem.r[b] < 0.0001)
        newmem.central[cent] = 1

        newmem.z[:] = cat_z[a]
        newmem.z_lambda = cat.z_lambda[a]

        if self.config.calib_smooth > 0.0:
            newmem.z[:] += self.config.calib_smooth * self.rng.normal(size=newmem.size)

        newmem.to_fits_file(self.config.zmemfile)

        # Generate QA plots if requested
        if self.config.more_qa_plots:
            self._make_qa_plots(newmem, mode)

    def _make_qa_plots(self, newmem, mode):
        """
        Generate QA plots for member preparation.

        Parameters
        ----------
        newmem : `Catalog`
            Prepared member catalog
        mode : `str`
            Mode used for member preparation
        """

        os.makedirs(self.config.plotpath, exist_ok=True)

        # Plot 1: z vs z_lambda scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hexbin(newmem.z_lambda, newmem.z, gridsize=50, cmap='Reds', mincnt=1)
        ax.plot([newmem.z_lambda.min(), newmem.z_lambda.max()],
                [newmem.z_lambda.min(), newmem.z_lambda.max()], 'r--', lw=2, label='1:1')
        ax.set_xlabel('z_lambda (cluster)', fontsize=12)
        ax.set_ylabel(f'z (member, mode={mode})', fontsize=12)
        ax.set_title(f'Member Redshift vs Cluster Redshift (N={newmem.size})', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'prepmembers_{mode}_z_scatter.png'), dpi=300)
        plt.close()

        # Plot 2: Probability distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(newmem.p, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(self.config.calib_pcut, color='red', ls='--', lw=2, label=f'pcut={self.config.calib_pcut}')
        axes[0].set_xlabel('p (membership probability)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Membership Probability Distribution', fontsize=12)
        axes[0].legend()
        axes[0].set_yscale('log')

        axes[1].hist(newmem.pcol, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(self.config.calib_pcut, color='red', ls='--', lw=2, label=f'pcut={self.config.calib_pcut}')
        axes[1].set_xlabel('pcol (color probability)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Color Probability Distribution', fontsize=12)
        axes[1].legend()
        axes[1].set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'prepmembers_{mode}_probability.png'), dpi=300)
        plt.close()

        # Plot 3: Color-magnitude diagram (ref band)
        fig, ax = plt.subplots(figsize=(8, 6))
        if self.config.nmag >= 2:
            color = newmem.mag[:, 0] - newmem.mag[:, 1]
            ax.hexbin(color, newmem.refmag, gridsize=50, cmap='coolwarm', mincnt=1)
            ax.set_xlabel(f'{self.config.bands[0]} - {self.config.bands[1]} color', fontsize=12)
        else:
            ax.hexbin(newmem.refmag, newmem.z, gridsize=50, cmap='coolwarm', mincnt=1)
            ax.set_xlabel('refmag', fontsize=12)
        ax.set_ylabel('refmag (i-band)', fontsize=12)
        ax.invert_yaxis()
        ax.set_title('Color-Magnitude Diagram', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'prepmembers_{mode}_cmd.png'), dpi=300)
        plt.close()

        # Plot 4: Central vs satellite comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        central_mask = newmem.central == 1
        satellite_mask = newmem.central == 0

        axes[0].hist(newmem.refmag[central_mask], bins=30, alpha=0.7, label=f'Central (N={central_mask.sum()})', color='red')
        axes[0].hist(newmem.refmag[satellite_mask], bins=30, alpha=0.7, label=f'Satellite (N={satellite_mask.sum()})', color='blue')
        axes[0].set_xlabel('refmag', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Magnitude Distribution', fontsize=12)
        axes[0].legend()

        axes[1].hist(newmem.p[central_mask], bins=30, alpha=0.7, label='Central', color='red')
        axes[1].hist(newmem.p[satellite_mask], bins=30, alpha=0.7, label='Satellite', color='blue')
        axes[1].set_xlabel('p (membership probability)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Probability Distribution', fontsize=12)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'prepmembers_{mode}_central_vs_satellite.png'), dpi=300)
        plt.close()

        # Plot 5: Redshift distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(newmem.z, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('z (member redshift)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Member Redshift Distribution (mode={mode})', fontsize=14)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plotpath, f'prepmembers_{mode}_z_hist.png'), dpi=300)
        plt.close()


