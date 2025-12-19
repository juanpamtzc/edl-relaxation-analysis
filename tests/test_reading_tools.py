from analysis.reading_tools import readDatFile, readTRJFile, readTRJFile_stresses, readLAMMPSThermodynamicFile
import unittest
import numpy as np

class TestReading(unittest.TestCase):

    def test_readDatFile_returns_correct_data(self):
        filename = "./data/test_data.dat"
        data = readDatFile(filename)
        self.assertIsInstance(data, dict)


    def test_readDatFile_output_has_required_keys(self):
        filename = "./data/test_data.dat"
        data = readDatFile(filename)
        required_keys = ['Masses', 'Atoms', '# atoms',
                        '# bonds', '# angles', '# dihedrals',
                        '# impropers', '# atom types',
                        '# bond types', '# angle types',
                        '# dihedral types', '# improper types',
                        'xlo', 'xhi', 'ylo', 'yhi',
                        'zlo', 'zhi']
        for key in required_keys:
            self.assertIn(key, data, f"Missing required key: {key}")

