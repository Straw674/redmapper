The redMaPPer How-To
====================

This is growing documentation on how to run redMaPPer on your data.  These
instructions are incomplete, as is the documentation.  Please see [the main
page](../README.md) for installation instructions.

Requirements
------------

This section details the data requirements.

### Photometric Galaxy Catalog

The first thing that you need is a photometric galaxy catalog.  The
requirements for redMaPPer are quite simple: unique identifying integer;
position; total magnitude and error in the reference band; color-optimized
magnitude and error in each individual band; E(B-V) from Galactic reddening for
systematics checks (not yet implemented).

If you have a simulated catalog, you can also add in the true redshift; the
mass of the host halo; if the galaxy is a central; and the halo id.  These are
just used for comparisons, not as input.  (Many of these comparisons have not
been ported to python yet.)

Note that the canonical example of a different "total" magnitude and
"color-optimized" magnitude is from SDSS: the cmodel magnitudes are good for
measuring total flux (though not great, just ask the people who do detailed
modeling of galaxy outskirts!), and the model magnitudes are good for colors.

The band used as the reference band is up to the user.  For SDSS clusters,
redMaPPer was run with i-band (which is good and bright up to z\~0.6), and for
DES and LSST it is run with z-band (which avoids getting cut by the 4000A break
until z\~1).

For the expected datatype, please see `redmapper_dtype` in the python code
below.

In addition, you will need the maximum magnitude in the reference band, keys
for all the indices, as described in the python code below.  If you magnitudes
are actually arcsinh luptitudes, then you need to supply an array of "b"
softening parameters and a zeropoint to do the magnitude/luptitude/flux
conversions.

```python

import redmapper

filename_base = '/path/to/files/foo'
# files will show up as '/path/to/files/foo_XXXXX.fits' (where XXXXX is a
# healpix number) and the mail galaxy file will be
# '/path/to/files/foo_master_table.fits'

redmapper_dtype = [('id', 'i8'),             # galaxy id number (unique)
                   ('ra', 'f8'),             # right ascension (degrees)
                   ('dec', 'f8'),            # declination (degrees)
                   ('refmag', 'f4'),         # total magnitude in reference band
                   ('refmag_err', 'f4'),     # error in total reference mag
                   ('mag', 'f4', nmag),      # mag array
                   ('mag_err', 'f4', nmag),  # magnitude error array
                   ('ebv', 'f4'),            # E(B-V) (systematics checking)
                   ('ztrue', 'f4'),          # ztrue if from a simulated catalog
                   ('m200', 'f4'),           # m200 of halo if from a simulated catalog
                   ('central', 'i2'),        # central? 1 if yes (if from sims)
                   ('halo_id', 'i8')]        # halo_id if from a simulated catalog

info_dict = {}
info_dict['LIM_REF'] = limiting_mag_reference_band
info_dict['REF_IND'] = index_for_reference_band
info_dict['AREA'] = catalog_area
info_dict['NMAG'] = number_of_mags
info_dict['MODE'] = file_mode # currently SDSS, DES, or LSST
info_dict['ZP'] = reference_zeropoint_for_mag_to_flux
info_dict['B'] = b_array # if magnitudes are actually luptitudes
info_dict['G_IND'] = 0 # g-band index
info_dict['R_IND'] = 1 # r-band index
# (etc, for the rest of the bands)

maker = redmapper.GalaxyCatalogMaker(filename_base, info_dict)

for input_file in input_files:
    # insert code to translate to file format

    maker.append_galaxies(galaxies)

maker.finalize_catalog()
```

If you have a set of galaxies and a geometric mask (see below), and the
geometric mask has *not* been applied to the set of galaxies, these can be
applied at this stage, for example:

```
maker = redmapper.GalaxyCatalogMaker(filename_base, info_dict,
maskfile='maskfile', mask_mode=3)
```

And the mask will then be applied every time galaxies are appended to the catalog.

### Spectroscopic Training Galaxies

In addition to the photometric galaxies, you also need a spectroscopic sample
that includes cluster centrals over the redshift range for which you want to
run the cluster finder.  For SDSS, these are easily obtained, but it may
require some thought for other surveys.  These need not be complete, but for
adequate performance you want at least 40 clusters per 0.05 redshift bin (see
Appendix A of [Rykoff et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...785..104R)).

The spectroscopic catalog should have the following format:

```python

spec_dtype = [('ra', 'f8'),        # right ascension (degrees)
              ('dec', 'f8'),       # declination (degrees)
              ('z', 'f4'),         # spectrosopic redshift
              ('z_err', 'f4')]     # error on spec-z
```

### Survey Geometry Mask (Strongly Recommended)

Although redMaPPer does not require a survey geometry mask in order to run, it
is strongly recommended when running on real data.  For certain types of simulated
data that cover a large contiguous area with limited boundaries and no
star-holes or bad fields, you can definitely get away without a survey geometry
mask!

The only type of mask currently supported in the `redmapper` package is a
healpix geometry mask as described by the
[`healsparse`](https://github.com/lsstdesc/healsparse) format.  Other types of
mask could have supported added provided they have reasonably efficient ways of doing a
position lookup.

The `healsparse` format is much more memory efficient and faster to read than a
standard `healpy` healpix file or the older `redmapper` format.  The format of
the mask is a standard `healsparse` map with a float value for `fracgood` (the
fractional "good" coverage) in each pixel.  Using `fracgood` allows you to
approximate a higher resolution mask via sub-sampling.  If you do not have
`fracgood`, then you should set each pixel that is in the mask to 1.0, and each
pixel that is outside the mask to `healpy.UNSEEN` (the default value).

If you have an old-style `redmapper` mask, this can be converted to the
`healsparse` format with the executable
`redmapper_convert_mask_to_healsparse.py`.

### Survey Depth Maps (Recommended)

For best performance, you can specify a depth map with a format similar to the
survey geometry masks above.  Depth maps make it possible to have better
richness measurements for higher redshift clusters that are near the threshold,
and add the ability to make a volume-limited catalog (see Section 3.4 of [Rykoff
et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJS..224....1R)).  If the
depth map is not specified, the code will do its best to approximate it with a
fit to the galaxies (see Appendix B of [Rozo et
al. (2015)](http://adsabs.harvard.edu/abs/2015MNRAS.453...38R)).

The depth map should be in
[`healsparse`](https://github.com/lsstdesc/healsparse) format.  Other types of
mask could have supported added provided they have reasonably efficient ways of doing a
position lookup (as with the survey geometry masks).

The depth map `healsparse` file has the format described below, with values
consistent with the model in Section 3 of [Rykoff et
al. (2015)](http://adsabs.harvard.edu/abs/2015arXiv150900870R).

```python
import fitsio

hdr = fitsio.FITSHDR()
hdr['ZP'] = reference_zeropoint
hdr['NSIG'] = signal_to_noise_at_limmag
hdr['NBAND'] = 1  # not used
hdr['W'] = 0.0  # not used
hdr['EFF'] = 1.0 # not used

depth_dtype = [('exptime', 'f4'),  # effective exposure time
               ('limmag', 'f4'),   # limited magnitude at nsig (in header)
               ('m50', 'f4'),      # Should be same as LIMMAG (for now)
               ('fracgood', 'f4')] # fraction of good coverage (see mask above)
```

If you have an old-style `redmapper` depth file, this can be converted to the
`healsparse` format with the executable
`redmapper_convert_depthfile_to_healsparse.py`.

### m\* as a Function of Redshift

As described in Section 3.2 of [Rykoff et
al. (2016)](http://adsabs.harvard.edu/abs/2016ApJS..224....1R) we need to know
the value of m\* (as defined in a Schechter luminosity function) as a function
of redshift for the reference band.  As a default this is dervied from a
[Bruzual & Charlot (2003,
BC03)](http://adsabs.harvard.edu/abs/2003MNRAS.344.1000B) model.

For convenience, the following files are included with the `redmapper` package, and
more can be added on request (and I will add the code to generate these files
at some point in the near future):

* `mstar_sdss_r03.fit`
* `mstar_sdss_i03.fit`
* `mstar_sdss_z03.fit`
* `mstar_des_i03.fit`
* `mstar_des_z03.fit`
* `mstar_hsc_i03.fit`
* `mstar_hsc_z03.fit`
* `mstar_lsst_i03.fit`
* `mstar_lsst_r03.fit`
* `mstar_lsst_z03.fit`

(The "03" in the names refer to them being generated from a BC03 model.)

The format of the files is as follows:

```python

import fitsio

hdr = fitsio.FITSHDR()
hdr['SURVEY'] = survey_name       # e.g., 'lsst'
hdr['BAND'] = band_name_from_file # e.g., 'z03'

mstar_dtype = [('Z', 'f4'),       # redshift
               ('MSTAR', 'f4')]   # mstar at redshift z
```

### Initial Guess for Red Sequence as a Function of Redshift

The final ingredient is the initial guess for the red-sequence as a function of
redshift.  This does not need to be super-precise, but it is helpful to be
close in each redshift range of interest.  For example, the g-r color should be
reasonable at z < 0.4; the r-i color should be reasonable at 0.3 < r-i < 0.8;
and the i-z color reasonable at 0.7 < i-z.

As described in Section 3.3 of [Rykoff et
al. (2016)](http://adsabs.harvard.edu/abs/2016ApJS..224....1R) this is also
derived similarly to m\*(z) using BC03 models, though any reasonable method can
work.

For convenience, the following files are included with the `redmapper` package, and
more can be added on request (and I will add the code to generate these files
at some point in the near future):

* `bc03_colors_sdss.fit`
* `bc03_colors_des.fit`
* `bc03_colors_hsc.fit`
* `bc03_colors_lsst.fit`

The format of the files is as follows:

```python

import fitsio

hdr = fitsio.FITSHDR()
hdr['SURVEY'] = survey_name        # e.g., 'lsst'
hdr['BAND'] = band_names_from_file # e.g., 'grizy'

mstar_dtype = [('Z', 'f4'),               # redshift
               ('COLOR', 'f4', n_color)]  # colors at redshift z
```

Red-Sequence Calibration
------------------------

With all these ingredients, it is easy to run the red-sequence calibration:

```python

redmapper_calibrate.py -c cal.yml
```

Please see the example [cal_example.yml](cal_example.yml) file in this
directory for all the configuration settings, and what they mean.

The red-sequence calibration code will run on a local machine that has enough
memory to hold the full galaxy catalog (at least for the calibration area,
which is configurable), and you can set the number of cores to run on (using
python multiprocessing).  It will produce a large number of files in the
current directory, diagnostic plots in the `plots/` directory, and may take
more than a day (depending on the area and depth of the catalog).

Typically, my directory tree looks like the following:

```
/path/to/stuff/redmapper_v0.2.1py/
/path/to/stuff/redmapper_v0.2.1py/cal/cal.yml
```

And after the calibration is run in the
`/path/to/stuff/redmapper_v0.2.1py/cal/` directory, it will create a pre-made
config file for the full cluster finder run with all the calibration outputs pre-filled:

```
/path/to/stuff/redmapper_v0.2.1py/
/path/to/stuff/redmapper_v0.2.1py/cal/cal.yml
/path/to/stuff/redmapper_v0.2.1py/run/run_default.yml
```

Note that you can name your calibration directory and config yaml file whatever
you want, but if you stick with the `cal` or `cal_foo` convention the code will
automatically replace `cal` with `run` when creating the run files.

Running the Cluster Finder
--------------------------

There are two phases to running the cluster finder.  The first is to compute
the `zred` photometric redshifts for all the galaxies, and the second is to run
the actual cluster finder.  In between, you may want to recompute the global
background.

Note that if you run the calibration on the full catalog, and not a subregion,
then you can skip the `zred` photometric redshift run and background
recalculation because these have already been run.

### Step 1: Set up the `run.yml` configuration file

The first step is to copy the `run_default.yml` configuration file to something like
`run.yml` and to take a look at it to ensure that everything looks okay.  The
main things to look at are:

* `zredfile`: Does this point to a file in the calibration directory (already
  done!) or in the run directory (to do!)
* `bkgfile`: Does this point to a file in the calibration directory (already
  done!) or in the run directory (to do!)

Unfortunately, the background file generation has not yet been optimized for memory
usage.  If you have a very large footprint, you may want to set the
`calib_make_full_bkg` to `False` in the calibration configuration file.

### Step 2: Compute zred photometric redshifts (if necessary)

The second step (if the calibration was not performed on the full footprint) is
to compute the zred photometric redshifts.  This can be done one of two ways.
If you have a compute cluster, then you should be able to run:

```bash

redmapper_batch.py -c run.yml -r 1 -b BATCHMODE -w WALLTIME -n NSIDE
```

On first run, the code will create a configuration file called
`$HOME/.redmapper_batch.yml`.  You must edit this, with the following format:

```yaml

batchmode:
   setup: 'environment_setup_script_to_call'
   batch: 'lsf'
   requirements: 'linux64'
```

You can name `batchmode` whatever you want, and you can define more than one
section if you run on multiple clusters.  Currently, the only batch system that
is supported is `lsf`, but I'll be adding `slurm` support soon, and please let
me know if you have other needs.  Finally, `requirements` is the batch system
requirements (`-R` in `lsf`).

When run after setup, this will create directory called `jobs/` and an `lsf`
batch file called `foo_zred_1.job`.  You can specify the walltime for the
compute cluster (default is 5 hours) and the nside splitting (larger number
means more jobs that will run faster).  Default is nside=8.

Alternatively, if you do not have a compute cluster, this can be run locally in
the following way:

```python

import redmapper

config = redmapper.Configuration('run.yml')
zredRunpix = redmapper.ZredRunPixels(config)
# This will use python multiprocessing to run on config.calib_nproc cores
zredRunpix.run()
```

### Step 3: Compute zred background (if necessary)

The third step (if the calibration was not performed on the full footprint, and
you have asked for a computation of the new background) is to compute the zred
background.  This needs to be streamlined, but can be accomplished with:

```bash

redmapper_make_zred_bkg.py -c run.yml
```

### Step 4: Run the Cluster Finder

The fourth step is to run the full cluster finder.  It is strongly recommended
to run this on a compute cluster, but it is possible to run it locally (though
not fully tested yet...).

See above for setting up the batch system.  Then you do:

```bash

redmapper_batch.py -c run.yml -r 0 -b BATCHMODE -w WALLTIME -n NSIDE
```

When run after setup, this will create a directory called `jobs/` and an `lsf`
batch file called `foo_run_1.job`.  You can specify the walltime for the
compute cluster (default is 72 hours) and the nside splitting (larger number
means more jobs that will run faster).  Default is nside=4.  Higher nside is
smaller pixels which run faster and use less memory, but there are diminishing
returns because you still have to run the buffer region between pixels.

Alternatively, if you do not have a compute cluster, this can be run locally in
the following way:

```python

import redmapper
import numpy as np

config = redmapper.Configuration('run.yml')
config.run_min_nside = 4  # nside desired
config.border = config.compute_border()
# This will use python multiprocessing to run on config.calib_run_nproc cores
redmapper.redmapper_run(config, consolidate=False)
```

### Step 5: Consolidate the Catalog Pixels

The final step is to consolidate the catalog pixels that were run.  This is
done with:

```bash

redmapper_consolidate_run.py -c run.yml [-l LAMBDA_CUTS] [-v VLIM_LSTARS]
```

The `lambda_cuts` (typically lambda > 5 and lambda > 20) can also be specified
in `run.yml`, but the command line will override it.  The `vlim_lstars`
arguments specify the "fraction of L* for which the galaxy catalog depth should
be bright enough to find a typical red galaxy at redshift z".  This can only be
specified if you have depth maps for all the relevant bands.  Typically, this
is set to 0.2 (the same as the luminosity cut for the richness estimation) for
a volume limited catalog, and to some large number (I use 5.0) for a "full"
catalog that covers as much redshift as possible, but is not volume-limited.

Some diagnostic plots will be placed in the `plots/` subdirectory of the run.

And that's it, for creating a cluster catalog!

But wait, there's more...

### Step 6: Generating redMaPPer Random Points

If you want to generate redMaPPer random points (see Section 3.6 of [Rykoff et
al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJS..224....1R)), it is a bit
clunky at the moment, but it does work.  There are four stages to generating
the redMaPPer randoms, and these need to be performed separately for each
volume-limited catalog that you have.  The three stages are:

1. Generate raw uniform random points with appropriate redshift/richness
distribution (`redmapper_generate_randoms.py`).
2. Run random points to estimate local geometric/depth corrections
(`redmapper_run_zmask_pixel.py`/`redmapper_batch.py`).
3. Compute random weights on sub-selections (`redmapper_weight_randoms.py`).

#### Step 6.1: Generate Raw Uniform Randoms

The first step is to use an input redMaPPer cluster catalog to generate a set
of spatially uniform randoms with the same redshift/richness distribution as
the the cluster catalog, while taking into account the "zmax" maximum redshift
mask.  This step needs to be done once for each volume limit selected in Step
5, and preferentially in a separate directory for each run.

You first need to create a new directory called, for example "zmask02" for a
0.2L* volume limited catalog.  Copy the `run.yml` file from the run and call it
`run_zmask02.yml`, and edit the `catfile` to be the full path to the
appropriate catalog file that you want randoms for.  Typically, if you have
output catalogs with multiple richness cuts (defaults being lambda>5 and
lambda>20), then you want to use the catalog with the lowest richness cut
(which includes all the clusters with the higher richness cut), and down-select
the randoms at the end, as described in Section 6.3.

Additionally, edit the `randfile` to be a full path to a new raw random catalog
that will be generated.  Typically, this would be something like
`/path/to/files/zmask02/rands/name_of_cat_randoms_master_table.fit`.
Additionally, you must symlink into this directory the appropriate zmask file
(in this case, called `outbase_redmapper_version_vl02_vlim_zmask.fit`).  This
dance will be simplified in the future.

Then, call `redmapper_generate_randoms.py`.  The following will generate 50
million random points.

```bash

redmapper_generate_randoms.py -c run_zmask02.yml -n 50000000
```

#### Step 6.2: Run random points through redMaPPer code

The easiest way to do this is via the batch system:

```bash

redmapper_batch.py -c run_zmask02.yml -r 3 -b BATCHMODE [-w WALLTIME] [-n
NSIDE]
```

After it is done, to consolidate you do:

```bash

redmapper_consolidate_runcat.py -c run_zmask02.yml -r
```

Alternatively, you can run locally (not distributed) with:

```python

import redmapper

config = redmapper.Configuration('run_zmask02.yml')
rand_zmask = redmapper.RunRandomsZmask(config)
rand_zmask.run()
rand_zmask.output(savemembers=False)
```

#### Step 6.3: Compute random weights on sub-selections

The randoms must now be weighted according to their detection rate.  These
weights must be computed separately for each richness cut/bin chosen.  Right
now, the command-line code only easily computes weights for a full catalog with
a set of richness cuts (e.g., 5 and 20).  This is done with:

```bash

redmapper_weight_randoms.py -c run_zmask02.yml -r name_of_full_randfile.fit -l 5 20
```

This will output files of the form
`outbase_weighted_randoms_zMIN-MAX_lgt005_vl02.fit` (the randoms with
appropriate weights), and
`outbase_weighted_randoms_zMIN-MAX_lgt005_vl02_area.fit` (the area as a
function of redshift, with geometric corrections computed from the randoms).

If you want to weight specific redshift or richness bins, you can do:

```python

import redmapper

config = redmapper.Configuration('run_zmask02.yml')
weigher = redmapper.RandomWeigher(config, 'randfile.fit')
wt_randfile, wt_areafile = weighter.weight_randoms(lambda_cut, zrange=[minz,
maxz], lambdabin=[lambda_min, lambda_max])
```

Note that `lambda_cut` should be the minimum richness of the parent catalog (5
or 20) to ensure that scatter around the lambda_min cut for the richness bin is
accounted for properly.


### Step 7: Calibrating and Generating redMaGiC Catalogs

After a redMaPPer catalog is computed, you can calibrate and generate redMaGiC
catalogs (see [Rozo et
al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.1431R)).

First, make a new directory called (for example) `redmagic`.  Copy in the
`run.yml` from the run directory into the redmagic directory, and rename it
`redmagic_cal.yml`.  You'll need to set the following fields:

* `catfile`: The full path of the redMaPPer catalog with the "best" redshifts
  for calibration, usually the lambda>20 catalog.
* `hpix`, `nside`: Set these if you want to calibrate over a sub-area of the
  survey (for speed and to reserve regions for testing).
* `redmagic_etas`: The luminosity cuts in units of L*, typically `[0.5, 1.0]`
* `redmagic_n0s`: The comoving density target in 10^-4 gal/Mpc^3, typcially
  `[10.0, 4.0]`
* `redmagic_names`: The names of the redMaGiC selections, typically
  `['highdens', 'highlum']`
* `redmagic_zmaxes`: The maximum redshifts for the two selections.  For DES
  this is `[0.75, 0.95]`
* `redmagicfile`: The calibration file to output, typically
  `outbase_redmagic_calib.fit`

Then you can run the redMaGiC calibration and selection step:

```bash

redmagic_calibrate.py -c redmagic_cal.yml
```

And voila!  After some time you will get the redMaGiC catalogs output with
calibration and run QA plots in the `plots/` directory.  Associated randoms
will also be generated, with 10x the number density as the redMaGiC galaxies.

If you already have a redMaGiC calibration and want to make a selection on a
new catalog (e.g., for multiple sim runs with the same red-sequence model), you
can use:

```bash

redmagic_run.py -c redmagic_run.yml [-C CLOBBER] [-n NRANDOMS]
```

If you don't specify `-n` then you'll get 10x the number of randoms as the
number of galaxies selected.  Note that you have to have the `zreds` already
available to make the redMaGiC selection.


