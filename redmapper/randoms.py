"""Classes for generating redmapper randoms.
"""

import fitsio
import esutil
import re
import copy
import numpy as np
import healsparse
import warnings

from .catalog import Catalog, Entry
from .galaxy import GalaxyCatalog, GalaxyCatalogMaker
from .cluster import ClusterCatalog
from .utilities import make_nodes, CubicSpline
from .fitters import MedZFitter
from .volumelimit import VolumeLimitMask

class GenerateRandoms(object):
    """
    Class to generate redmapper raw randoms using a redmapper volume limit mask.
    """

    def __init__(self, config, vlim_mask=None, vlim_lstar=None, redmapper_cat=None, rng=None):
        """
        Instantiate a GenerateRandoms object, to generate seed randoms for redmapper.

        Parameters
        ----------
        config : `redmapper.Configuration`
        vlim_mask : `redmapper.VolumeLimitMask`, optional
           Volume limit mask, or else it will be read/generated from config
        vlim_lstar : `float`, optional
           Volume limit lstar, or else it is from config.vlim_lstar
        redmapper_cat : `redmapper.ClusterCatalog`, optional
           Redmapper catalog, or else it will be read from config.catfile
        rng : `np.random.RandomState`, optional
            Random number generator.
        """
        self.config = config

        if rng is None:
            rng = np.random.RandomState(self.config.randomseed)
        self.rng = rng

        if self.config.randfile is None:
            raise RuntimeError("Must set randfile in config to run GenerateRandoms.")

        if vlim_lstar is None:
            self.vlim_lstar = self.config.vlim_lstar
        else:
            self.vlim_lstar = vlim_lstar

        if vlim_mask is None:
            self.vlim_mask = VolumeLimitMask(self.config, self.vlim_lstar)
        else:
            self.vlim_mask = vlim_mask

        if redmapper_cat is None:
            self.redmapper_cat = ClusterCatalog.from_fits_file(self.config.catfile)
        else:
            self.redmapper_cat = redmapper_cat

    def generate_randoms(self, nrandoms, rng=None):
        """
        Generate seed randoms for use in redmapper, applying redshift mask.

        Parameters
        ----------
        nrandoms : `int`
           Number of random points to generate.
        rng : `np.random.RandomState`, optional
           Pre-set random number generator.  Default is None.
        """
        if rng is None:
            rng = self.rng

        min_gen = 10000
        max_gen = 1000000

        n_left = copy.copy(nrandoms)
        ctr = 0

        dtype = [('id', 'i4'),
                 ('ra', 'f8'),
                 ('dec', 'f8'),
                 ('z', 'f4'),
                 ('lambda', 'f4'),
                 ('id_input', 'i4')]

        info_dict = {}
        # Get outbase from self.config.randfile
        m = re.search(r'(.*)\_master\_table.fit$', self.config.randfile)
        if m is None:
            raise RuntimeError("Config has randfile of incorrect format.  Must end in _master_table.fit")
        outbase = m.groups()[0]
        maker = RandomCatalogMaker(outbase, info_dict, nside=self.config.galfile_nside)

        self.config.logger.info("Generating %d randoms to %s" % (n_left, outbase))

        while (n_left > 0):
            n_gen = np.clip(n_left * 3, min_gen, max_gen)
            ra_rand, dec_rand = healsparse.make_uniform_randoms(self.vlim_mask.sparse_vlimmap,
                                                                n_gen, rng=rng)

            zmax, fracgood = self.vlim_mask.calc_zmax(ra_rand, dec_rand, get_fracgood=True)

            r = rng.uniform(size=n_gen)
            gd, = np.where(r < fracgood)

            if gd.size == 0:
                continue

            tempcat = Catalog(np.zeros(gd.size, dtype=dtype))
            tempcat.ra = ra_rand[gd]
            tempcat.dec = dec_rand[gd]
            tempcat.z = -1.0

            r = rng.choice(np.arange(self.redmapper_cat.size), size=gd.size, replace=True)
            zz = self.redmapper_cat.z_lambda[r]
            ll = self.redmapper_cat.Lambda[r]
            ii = self.redmapper_cat.mem_match_id[r]

            # zctr counts the number of successfully placed randoms
            # while i counts index through tempcat, many of which will be rejected.
            zctr = 0
            for i in range(tempcat.size):
                if (zz[zctr] < zmax[i]):
                    # This is in a location that is within the volume limit.
                    tempcat.z[i] = zz[zctr]
                    tempcat.Lambda[i] = ll[zctr]
                    tempcat.id_input[i] = ii[zctr]
                    zctr += 1

            # Which of the tempcat were actually placed?
            gd, = np.where(tempcat.z > 0.0)
            n_good = gd.size

            if n_good == 0:
                continue

            if n_good > n_left:
                n_good = n_left
                gd = gd[0: n_good]

            tempcat = tempcat[gd]
            tempcat.id = np.arange(ctr + 1, ctr + n_good + 1)

            maker.append_randoms(tempcat._ndarray[: n_good])

            ctr += n_good
            n_left -= n_good
            self.config.logger.info("There are %d randoms remaining..." % (n_left))

        maker.finalize_catalog()


class RandomCatalog(GalaxyCatalog):
    """
    """

    @classmethod
    def from_randfile(cls, filename, nside=0, hpix=[], border=0.0):
        """
        """

        return super(RandomCatalog, cls).from_galfile(filename, nside=nside, hpix=hpix, border=border)

    @classmethod
    def from_galfile(cls, filename, zredfile=None, nside=0, hpix=[], border=0.0, truth=False):
        raise NotImplementedError("Cannot call from_galfile on a RandomCatalog")

    @property
    def galcol(self):
        raise NotImplementedError("Cannot call galcol on a RandomCatalog")

    @property
    def galcol_err(self):
        raise NotImplementedError("Cannot call galcol_err on a RandomCatalog")

    @property
    def add_zred_fields(self):
        raise NotImplementedError("Cannot call add_zred_fields on a RandomCatalog")


class RandomCatalogMaker(GalaxyCatalogMaker):
    """
    """

    def __init__(self, outbase, info_dict, nside=32, maskfile=None, mask_mode=0, parallel=False):
        """
        """

        if 'LIM_REF' not in info_dict:
            info_dict['LIM_REF'] = 0.0
        if 'REF_IND' not in info_dict:
            info_dict['REF_IND'] = 0
        if 'AREA' not in info_dict:
            info_dict['AREA'] = 0.0
        if 'NMAG' not in info_dict:
            info_dict['NMAG'] = 0
        if 'MODE' not in info_dict:
            info_dict['MODE'] = 'NONE'
        if 'ZP' not in info_dict:
            info_dict['ZP'] = 0.0

        super(RandomCatalogMaker, self).__init__(outbase, info_dict, nside=nside, maskfile=maskfile, mask_mode=mask_mode, parallel=parallel)

    def split_randoms(self, rands):
        """
        """

        if self.is_finalized:
            raise RuntimeError("Cannot split randoms for an already finalized catalog.")
        if os.path.isfile(self.filename):
            raise RuntimeError("Cannot split randoms when final file %s already exists." % (self.filename))

        self.append_randoms(rands)
        self.finalize_catalog()

    def append_randoms(self, rands):
        """
        """

        self.append_galaxies(rands)

    def _check_galaxies(self, rands):
        # These always come back true.
        return True


class RandomWeigher(object):
    """
    Class to compute random weights and effective area.
    """
    def __init__(self, config, randcatfile, vlim_mask=None, vlim_lstar=None, redmapper_cat=None):
        """
        Instantiate a RandomWeigher object, to generate weighted randoms.

        Parameters
        ----------
        config : `redmapper.Configuration`
        randcatfile : `str`
           Consolidated random catalog with scaleval, maskfrac
        vlim_mask : `redmapper.VolumeLimitMask`, optional
           Volume limit mask, or else it will be read/generated from config
        vlim_lstar : `float`, optional
           Volume limit lstar, or else it is from config.vlim_lstar
        redmapper_cat : `redmapper.ClusterCatalog`, optional
           Redmapper catalog, or else it will be read from config.catfile
        """
        self.config = config

        self.randcat = Catalog.from_fits_file(randcatfile)

        if vlim_lstar is None:
            self.vlim_lstar = self.config.vlim_lstar
        else:
            self.vlim_lstar = vlim_lstar

        if vlim_mask is None:
            self.vlim_mask = VolumeLimitMask(self.config, self.vlim_lstar)
        else:
            self.vlim_mask = vlim_mask

        if redmapper_cat is None:
            self.redmapper_cat = ClusterCatalog.from_fits_file(self.config.catfile)
        else:
            self.redmapper_cat = redmapper_cat

    def weight_randoms(self, minlambda, zrange=None, lambdabin=None):
        """
        Compute random weights.

        Parameters
        ----------
        minlambda : `float`
           Minimum lambda to use in computations
        zrange : `np.ndarray`, optional
           2-element list of redshift range.  Default is full range.
        lambdabin : `np.ndarray`, optional
           2-element list of lambda range.  Default is full range.
        """
        if zrange is None:
            zrange = np.array([self.config.zrange[0], self.config.zrange[1]])

        zname = 'z%03d-%03d' % (int(self.config.zrange[0]*100),
                                int(self.config.zrange[1]*100))
        vlimname = 'vl%02d' % (int(self.vlim_lstar*10))
        if lambdabin is None:
            lamname = 'lgt%03d' % (int(minlambda))
            lambdabin = np.array([0.0, 1000.0])
        else:
            lamname = 'lgt%03d_l%03d-%03d' % (int(minlambda), int(lambdabin[0]), int(lambdabin[1]))

        zuse, = np.where((self.randcat.z > zrange[0]) &
                         (self.randcat.z < zrange[1]))

        if zuse.size == 0:
            raise RuntimeError("No random points in specified redshift range %.2f < z < %.2f" %
                               (zrange[0], zrange[1]))

        st = np.argsort(self.randcat.id_input[zuse])
        uid = np.unique(self.randcat.id_input[zuse[st]])

        a, b = esutil.numpy_util.match(self.redmapper_cat.mem_match_id, uid)

        if b.size < uid.size:
            raise RuntimeError("IDs in randcat do not match those of corresponding redmapper catalog.")

        a, b = esutil.numpy_util.match(self.redmapper_cat.mem_match_id, self.randcat.id_input[zuse])

        if self.config.select_scaleval:
            luse, = np.where((self.redmapper_cat.Lambda[a]/self.redmapper_cat.scaleval[a] > minlambda) &
                             (self.redmapper_cat.Lambda[a] > lambdabin[0]) &
                             (self.redmapper_cat.Lambda[a] <= lambdabin[1]))
        else:
            luse, = np.where((self.redmapper_cat.Lambda[a] > minlambda) &
                             (self.redmapper_cat.Lambda[a] > lambdabin[0]) &
                             (self.redmapper_cat.Lambda[a] <= lambdabin[1]))

        if luse.size == 0:
            raise RuntimeError("No random points in specified richness range %0.2f < lambda < %0.2f and lambda > %.2f" %
                               (lambdabin[0], lambdabin[1], minlambda))

        alluse = zuse[b[luse]]

        randpoints = Catalog.zeros(luse.size, dtype=[('ra', 'f8'),
                                                     ('dec', 'f8'),
                                                     ('ztrue', 'f4'),
                                                     ('lambda_in', 'f4'),
                                                     ('avg_lambdaout', 'f4'),
                                                     ('weight', 'f4')])
        randpoints.ra = self.randcat.ra[alluse]
        randpoints.dec = self.randcat.dec[alluse]
        randpoints.ztrue = self.randcat.z[alluse]
        randpoints.lambda_in = self.randcat.lambda_in[alluse]
        randpoints.avg_lambdaout = self.randcat.lambda_in[alluse]

        h, rev = esutil.stat.histogram(self.randcat.id_input[alluse], rev=True)
        ok, = np.where(h > 0)

        for i in ok:
            i1a = rev[rev[i]: rev[i + 1]]

            if self.config.select_scaleval:
                gd, = np.where((self.randcat.lambda_in[alluse[i1a]]/self.randcat.scaleval[alluse[i1a]] > minlambda) &
                               (self.randcat.maskfrac[alluse[i1a]] < self.config.max_maskfrac) &
                               (self.randcat.lambda_in[alluse[i1a]] > lambdabin[0]) &
                               (self.randcat.lambda_in[alluse[i1a]] <= lambdabin[1]))
            else:
                gd, = np.where((self.randcat.lambda_in[alluse[i1a]] > minlambda) &
                               (self.randcat.maskfrac[alluse[i1a]] < self.config.max_maskfrac) &
                               (self.randcat.lambda_in[alluse[i1a]] > lambdabin[0]) &
                               (self.randcat.lambda_in[alluse[i1a]] <= lambdabin[1]))

            if gd.size > 0:
                randpoints.weight[i1a[gd]] = float(i1a.size)/float(gd.size)

        # And only save the randpoints with weight > 0.0
        use, = np.where(randpoints.weight > 0.0)

        fname_base = 'weighted_randoms_%s_%s_%s' % (zname, lamname, vlimname)
        randfile_out = self.config.redmapper_filename(fname_base, withversion=True)

        randpoints.to_fits_file(randfile_out, indices=use)

        # And now we need to compute the associated area.

        # area = full_area(z < zmax) * (P/(P+Q)) where
        #   P = number of good points with z < zmax
        #   Q = number of bad points with z < zmax

        # get the default area structure

        astr = self.vlim_mask.get_areas()

        # Make the fitting nodes
        nodes = make_nodes(self.config.zrange, self.config.area_nodesize)

        zbinsize = self.config.area_coarsebin
        zbins = np.arange(self.config.zrange[0], self.config.zrange[1], zbinsize)

        st = np.argsort(self.randcat.z[alluse])
        ind1 = np.searchsorted(self.randcat.z[alluse[st]], zbins)
        if self.config.select_scaleval:
            gd, = np.where((self.randcat.lambda_in[alluse[st]]/self.randcat.scaleval[alluse[st]] > minlambda) &
                           (self.randcat.maskfrac[alluse[st]] < self.config.max_maskfrac) &
                           (self.randcat.lambda_in[alluse[st]] > lambdabin[0]) &
                           (self.randcat.lambda_in[alluse[st]] < lambdabin[1]))
        else:
            gd, = np.where((self.randcat.lambda_in[alluse[st]] > minlambda) &
                           (self.randcat.maskfrac[alluse[st]] < self.config.max_maskfrac) &
                           (self.randcat.lambda_in[alluse[st]] > lambdabin[0]) &
                           (self.randcat.lambda_in[alluse[st]] < lambdabin[1]))
        ind2 = np.searchsorted(self.randcat.z[alluse[st[gd]]], zbins)

        xvals = (zbins[0: -2] + zbins[1: -1])/2.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yvals = np.nan_to_num(ind2[1: -1].astype(np.float64) / ind1[1: -1].astype(np.float64))

        fitter = MedZFitter(nodes, xvals, yvals)
        p0 = np.ones(nodes.size)
        # Do an extra fit here for stability
        pars0 = fitter.fit(p0)
        pars = fitter.fit(pars0)

        spl = CubicSpline(nodes, pars)
        corrs = np.clip(spl(astr.z), 0.0, 1.0)
        astr.area = corrs*astr.area

        areafile_out = self.config.redmapper_filename(fname_base + '_area', withversion=True)
        astr.to_fits_file(areafile_out)

        return (randfile_out, areafile_out)
