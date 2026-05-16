import unittest
import numpy as np
from astropy.table import Table
from redmapper.core.models import (get_galaxy_schema, get_cluster_schema, 
                                   get_member_schema, get_zred_schema)

class CoreModelsTestCase(unittest.TestCase):
    def test_galaxy_schema(self):
        nmag = 5
        schema = get_galaxy_schema(nmag)
        tab = Table(dtype=schema)
        self.assertEqual(len(tab.colnames), 8)
        self.assertIn('mag', tab.colnames)
        self.assertEqual(tab['mag'].shape[1], nmag)

        schema_truth = get_galaxy_schema(nmag, truth=True)
        tab_truth = Table(dtype=schema_truth)
        self.assertIn('ztrue', tab_truth.colnames)
        self.assertIn('m200', tab_truth.colnames)

    def test_cluster_schema(self):
        schema = get_cluster_schema()
        tab = Table(dtype=schema)
        self.assertIn('mem_match_id', tab.colnames)
        self.assertIn('lambda', tab.colnames)
        # Verify lowercase
        self.assertTrue(all(col == col.lower() for col in tab.colnames))

    def test_member_schema(self):
        schema = get_member_schema()
        tab = Table(dtype=schema)
        self.assertIn('mem_match_id', tab.colnames)
        self.assertIn('p', tab.colnames)
        self.assertTrue(all(col == col.lower() for col in tab.colnames))

    def test_zred_schema(self):
        nsamp = 4
        schema = get_zred_schema(nsamp)
        tab = Table(dtype=schema)
        self.assertIn('zred_samp', tab.colnames)
        self.assertEqual(tab['zred_samp'].shape[1], nsamp)

if __name__ == '__main__':
    unittest.main()
