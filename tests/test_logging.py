import unittest
import tempfile
import shutil
import os

from redmapper import Configuration
from redmapper import logger

class LoggingTestCase(unittest.TestCase):
    """Tests for writing log files.
    """
    def test_log_printonly(self):
        """Test default logging, only print to stdout."""
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))

        # Note I do not know how to confirm this works other than running
        # by hand.
        logger.info("Testing!")

    def test_logfile_filename(self):
        """Test file logging, with a specific filename."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))
        config.outpath = self.test_dir
        config.printlogging = False

        config.start_file_logging("testlog.log")
        logger.info("Testing!")
        config.stop_file_logging()

        logfile = os.path.join(config.outpath, 'logs', 'testlog.log')
        self.assertTrue(os.path.exists(logfile))

    def test_logfile_multipix(self):
        """Test file logging, with a config with multiple pixels set."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))
        config.outpath = self.test_dir
        config.printlogging = False

        config.hpix = [1, 2]

        config.start_file_logging()
        logger.info("Testing!")
        config.stop_file_logging()

        logfile = os.path.join(config.outpath, 'logs',
                               f'redmapper_{config.outbase}_0001.log')
        self.assertTrue(os.path.exists(logfile))

    def test_logfile_singlepix(self):
        """Test file logging, with a config with a single pixel set."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))
        config.outpath = self.test_dir
        config.printlogging = False

        config.hpix = [23]

        config.start_file_logging()
        logger.info("Testing!")
        config.stop_file_logging()

        logfile = os.path.join(config.outpath, 'logs',
                               f'redmapper_{config.outbase}_0023.log')
        self.assertTrue(os.path.exists(logfile))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, ignore_errors=True)


if __name__=='__main__':
    unittest.main()
