import logging
import sys
import os

def get_logger(name="redmapper", level=logging.INFO):
    """
    Get a logger with the specified name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
            
    return logger

# Global logger for the redmapper package
logger = get_logger()

def set_log_level(level):
    """Set the logging level for the redmapper logger."""
    logger.setLevel(level)

def add_file_handler(log_file):
    """Add a file handler to the redmapper logger."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def remove_file_handlers():
    """Remove all file handlers from the redmapper logger."""
    handlers = logger.handlers[:]
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
