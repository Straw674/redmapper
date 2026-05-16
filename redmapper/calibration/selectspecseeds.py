"""Select seed galaxies
"""
from ..core.calibration import select_spec_seeds
from ..logger import logger

def select_spec_seeds_wrapper(config, usetrain=True):
    """
    Wrapper for the functional select_spec_seeds.
    Maintains compatibility with legacy code.
    """
    return select_spec_seeds(config, usetrain=usetrain, logger=logger)


