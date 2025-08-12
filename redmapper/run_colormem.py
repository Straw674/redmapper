"""Class to compute richnesses on a catalog by fitting a linear red-sequence
model, for use in the first part of training before a red-sequence model has
been found.

"""
import fitsio
import numpy as np
import esutil
import copy
import sys

from .cluster import ClusterCatalog
from .color_background import ColorBackground
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .catalog import Catalog, Entry
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .cluster_runner import ClusterRunner
from .utilities import CubicSpline

###################################################
# Order of operations:
#  __init__()
#    _additional_initialization() [override this]
#  run()
#    _setup()
#    _more_setup() [override this]
#    _process_cluster() [override this]
#    _postprocess() [override this]
#  output()
###################################################

class RunColormem(ClusterRunner):
    """
    The RunColormem class is derived from ClusterRunner, and will compute a
    richness and membership probabilities using only an individual color per
    cluster.  The central galaxy is assumed to be close to the mean color of
    the cluster, and the red sequence is fit in a single color-magnitude space
    for a first richness estimate.
    """

    def _additional_initialization(self, **kwargs):
        """
        Additional initialization for RunColormem
        """
        self.runmode = "calib_colormem"
        self.filetype = "colormem"
        self.use_colorbkg = True
        self.use_parfile = False
        self.cutgals_bkgrange = False
        self.cutgals_chisqmax = False

    def run(self, *args, **kwargs):
        """
        Run a catalog through RunColormem.

        Loop over all clusters and perform RunColormem computations on each cluster.
        """

        return super(RunColormem, self).run(*args, **kwargs)

    def _more_setup(self, *args, **kwargs):
        """
        More setup for RunColormem.
        """
        self.rmask_0 = self.config.calib_colormem_r0
        self.rmask_beta = self.config.calib_colormem_beta

        self.cat = ClusterCatalog.from_catfile(self.config.redgalfile,
                                               zredstr=self.zredstr,
                                               config=self.config,
                                               cbkg=self.cbkg,
                                               cosmo=self.cosmo,
                                               r0=self.r0,
                                               beta=self.beta)

        use, = np.where((self.cat.z > self.config.zrange[0]) &
                        (self.cat.z < self.config.zrange[1]))
        self.cat = self.cat[use]

        self.cat.z_init = self.cat.z

        # need to insert red model and get colormodes...
        self.cat.add_fields([('redcolor', 'f4', self.config.nmag - 1)])

        redmodel = Entry.from_fits_file(self.config.redgalmodelfile)
        for j in range(self.config.nmag - 1):
            spl = CubicSpline(redmodel.nodes, redmodel.meancol[:, j])
            self.cat.redcolor[:, j] = spl(self.cat.z)

        self.zbounds = np.concatenate([np.array([self.config.zrange[0] - 0.011]),
                                       self.config.calib_colormem_zbounds,
                                       np.array([self.config.zrange[1] + 0.011])])

        self._generate_mem_match_ids()

        self.do_lam_plusminus = False
        self.record_members = False

        self.doublerun = True

        return True

    def _doublerun_sort(self):
        """
        Sort the catalog for when doing a second pass with percolation masking.
        """
        st = np.argsort(self.cat.Lambda)[::-1]
        self.cat = self.cat[st]

    def _process_cluster(self, cluster):
        """
        Process a single cluster with RunColormem.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute richness.
        """

        bad = False

        m = 0
        found = False
        while ((m < self.zbounds.size - 1) and (not found)):
            if (cluster.z > self.zbounds[m]) and (cluster.z <= self.zbounds[m + 1]):
                found = True
                mode = self.config.calib_colormem_colormodes[m]
            else:
                m += 1
        if (not found):
            raise RuntimeError("Programmer error with illegal mode")

        lam = cluster.calc_richness_fit(self.mask, mode, calc_err=False,
                                        centcolor_in=cluster.redcolor[mode])

        ind = np.argmin(cluster.neighbors.r)
        cluster.p_bcg = cluster.neighbors.pmem[ind]

        if (lam / cluster.scaleval < self.config.calib_colormem_minlambda):
            bad = True
            self._reset_bad_values(cluster)
            return bad

        if cluster.p_bcg < self.config.calib_pcut:
            bad = True
            self._reset_bad_values(cluster)
            return bad

        return bad

    def _postprocess(self):
        """
        RunColormem post-processing.

        This will select good clusters that are above the configured thresholds
        (self.config.calib_colormem_minlambda), and smooth the member redshifts
        with a configurable gaussian kernel
        (self.config.calib_colormem_smooth).
        """

        use, = np.where((self.cat.Lambda/self.cat.scaleval >= self.config.calib_colormem_minlambda) & (self.cat.scaleval > 0.0) & (self.cat.maskfrac < self.config.max_maskfrac))

        # Make a new catalog that doesn't have the extra memory usage
        # from catalogs and neighbors
        self.cat = ClusterCatalog(self.cat._ndarray[use])

        a, b = esutil.numpy_util.match(self.cat.mem_match_id, self.members.mem_match_id)
        self.members = Catalog(self.members._ndarray[b])

        if self.config.calib_colormem_smooth > 0.0:
            self.members.z += self.rng.normal(scale=self.config.calib_colormem_smooth, size=self.members.size)

    def output_training(self):
        """
        Output the training members and catalog.

        Member catalog is given by config.zmemfile, and cluster catalog is of
        'colorcat' type.
        """

        use, = np.where(self.members.pcol > self.config.calib_pcut)

        savemem = self.members[use]

        savemem.to_fits_file(self.config.zmemfile)

        # This should get a better name...
        self.cat.to_fits_file(self.config.redmapper_filename('colorcat'))
