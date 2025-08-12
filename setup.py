from setuptools import setup, Extension
import numpy
import os


scripts = ['bin/redmapper_run_zred_pixel.py',
           'bin/redmapper_run_redmapper_pixel.py',
           'bin/redmapper_batch.py',
           'bin/redmapper_make_zred_bkg.py',
           'bin/redmapper_calibrate.py',
           'bin/redmapper_consolidate_run.py',
           'bin/redmagic_calibrate.py',
           'bin/redmagic_run.py',
           'bin/redmapper_convert_mask_to_healsparse.py',
           'bin/redmapper_convert_depthfile_to_healsparse.py',
           'bin/redmapper_run_many_pixels_same_node.py',
           'bin/redmapper_build_docker.py',
           'bin/redmapper_consolidate_runcat.py',
           'bin/redmapper_runcat_pixel.py',
           'bin/redmapper_run_zmask_pixel.py',
           'bin/redmapper_run_zscan_pixel.py',
           'bin/redmapper_generate_randoms.py',
           'bin/redmapper_weight_randoms.py',
           'bin/redmapper_predict_memory.py']

solver_nfw_ext = Extension(
    "redmapper.solver_nfw._solver_nfw_pywrap",
    extra_compile_args=["-std=gnu99"],
    sources=[
        "redmapper/solver_nfw/solver_nfw_pywrap.c",
        "redmapper/solver_nfw/solver_nfw.c",
        "redmapper/solver_nfw/nfw_weights.c",
    ],
)

if os.environ.get("GSLI", None):
    if os.environ.get("CONDA_PREFIX", None):
        os.environ["GSLI"] = os.path.expandvars("${CONDA_PREFIX}/include")
        os.environ["GSLL"] = os.path.expandvars("${CONDA_PREFIX}/lib")

chisq_dist_ext = Extension(
    "redmapper.chisq_dist._chisq_dist_pywrap",
    extra_compile_args=["-std=gnu99", os.path.expandvars("-I${GSLI}")],
    extra_link_args=[os.path.expandvars("-L${GSLL}")],
    libraries=["gsl", "gslcblas"],
    sources=[
        "redmapper/chisq_dist/chisq_dist.c",
        "redmapper/chisq_dist/chisq_dist_pywrap.c",
    ],
)

setup(
    ext_modules=[solver_nfw_ext, chisq_dist_ext],
    include_dirs=numpy.get_include(),
    scripts=scripts,
)
