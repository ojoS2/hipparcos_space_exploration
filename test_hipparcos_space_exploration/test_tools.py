import unittest
import io
import contextlib
import pandas as pd 
import numpy as np
import sys
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExplorationWithPython/HiparcusStudies/hipparcos_space_exploration/src')
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExplorationWithPython/HiparcusStudies/hipparcos_space_exploration/data')
from tools import func
from tools import visual
from tools import CatalogObject
from tools import StelarObject

class TestClassMethods(unittest.TestCase):
    

    def test_import_data(self):
      df = func.load_data(number_of_records = 'all', how = 'tail')
      self.assertEqual(df.shape,(116593, 30))
      self.assertEqual(list(df.columns), ['HIP', 'Sn', 'So', 'Nc',
                         'RArad', 'DErad', 'Plx', 'pmRA', 'pmDE',
       'e_RArad', 'e_DErad', 'e_Plx', 'e_pmRA', 'e_pmDE', 'Ntr',
         'F2', 'F1', 'var', 'ic', 'Vmag', 'e_Hpmag', 'sHp', 'VA',
           'B-V', 'e_B-V', 'V-I', 'UW', 'common name',
             'spectral type', 'measured distance_ly'])
      df = func.load_data(number_of_records=1000, how='tail')
      self.assertEqual(df.shape,(1000, 30))
      self.assertTrue(abs((df.head(1))['Plx'].values - 48.5) < 0.0001)
      self.assertTrue(abs((df.tail(1))['Plx'].values - 796.92) < 0.0001)
      df = func.load_data(number_of_records=1000, how='head')
      self.assertEqual(df.shape,(1000, 30))
      self.assertTrue(abs((df.head(1))['Plx'].values + 0.01) < 0.0001)
      self.assertTrue(abs((df.tail(1))['Plx'].values - 0.14) < 0.0001)


    def test_poly(self):
       x = 0
       y_0 = func.poly([1,1,1], x)
       y_1 = func.poly([0,-1,1], x)
       y_2 = func.poly([-1,1,-1,1], x)
       y_3 = func.poly([1,-1,1,-1], x) 
       self.assertAlmostEqual(1,y_0,places=4)
       self.assertAlmostEqual(0,y_1,places=4)
       self.assertAlmostEqual(-1,y_2,places=4)
       self.assertAlmostEqual(1,y_3,places=4)
       x = 1
       y_0 = func.poly([1,1,1], x)
       y_1 = func.poly([0,-1,1], x)
       y_2 = func.poly([-1,1,-1,1], x)
       y_3 = func.poly([-1,1,-1,1], x) 
       self.assertAlmostEqual(3,y_0,places=4)
       self.assertAlmostEqual(0,y_1,places=4)
       self.assertAlmostEqual(0,y_2,places=4)
       self.assertAlmostEqual(0,y_3,places=4)


    def test_point_distance(self):
       x_0,y_0 = 1, 0
       x_1,y_1 = 1, 1
       x_2,y_2 = 0, 2
       x_3,y_3 = -1, -1
       x_4,y_4 = 0, 0
       self.assertAlmostEqual(func.point_distance((0,0),(x_0,y_0)), 1.0, places=4)
       self.assertAlmostEqual(func.point_distance((0,0),(x_1,y_1)), 1.4142, places=4)
       self.assertAlmostEqual(func.point_distance((0,0),(x_2,y_2)), 2., places=4)
       self.assertAlmostEqual(func.point_distance((0,0),(x_3,y_3)), 1.4142, places=4)
       self.assertAlmostEqual(func.point_distance((0,0),(x_4,y_4)), 0., places=4)


    def test_graph_distances(self):
        vec_0 = func.graph_distances(0.1, 15)
        vec_1 = func.graph_distances(1.7, 15)
        vec_2 = func.graph_distances(0.1, -15)
        vec_3 = func.graph_distances(1.7, -15)
        dict_0 = {'wd': 3.3051155838665682,
        'nd': 8.326285440801461, 'bd': 22.576304387689827,
         'sd': 10.405844592974965, 'ms': 13.985397236116546,
          'sG': 16.595336118964482, 'bG': 15.725616051799076,
            'rG': 16.795015191613974, 'brG': 17.739346789923573,
              'sGb': 19.38176953909926, 'sGa': 21.162959448540626,
                'hG': 23.85384715479699}
        dict_1 = {'wd': 27.117877331219653,
                   'nd': 24.605579602261756, 'bd': 6.34292817269943,
                     'sd': 2.4540690269402603, 'ms': 3.1996639291774045,
                       'sG': 12.418066091970408,
                         'bG': 17.25751905074696,
                           'rG': 15.617499428411303,
                             'brG': 18.68477324847768,
                               'sGb': 20.080809982133992,
                                 'sGa': 21.605298723167525,
                                    'hG': 24.019197700319815}
        dict_2 = {'wd': 26.694884416133434,
                   'nd': 21.673714559198537,
                     'bd': 7.597253841496819,
                       'sd': 19.594155407025035,
                         'ms': 16.014602763883456,
                           'sG': 13.407360883550206,
                             'bG': 14.274383948200924,
                               'rG': 13.238098954731903,
                                 'brG': 12.260653210076427,
                                   'sGb': 10.618230460900739,
                                     'sGa': 8.837040551459376,
                                       'hG': 6.146152845203009}
        dict_3 = {'wd': 3.1464991256542425,
                   'nd': 5.485224047122164,
                     'bd': 23.65707182730057,
                       'sd': 27.54593097305974,
                         'ms': 26.800336070822596,
                           'sG': 17.58193390802959,
                             'bG': 12.797650617984045,
                               'rG': 14.382500571588697,
                                 'brG': 11.31522675152232,
                                   'sGb': 9.919190017866006,
                                     'sGa': 8.394701276832475,
                                       'hG': 5.980802299680185}
        for key in dict_0.keys():
           self.assertAlmostEqual(vec_0[key], dict_0[key], places=4)
           self.assertAlmostEqual(vec_1[key], dict_1[key], places=4)
           self.assertAlmostEqual(vec_2[key], dict_2[key], places=4)
           self.assertAlmostEqual(vec_3[key], dict_3[key], places=4)


    def test_inverse(self):
       self.assertAlmostEqual(func.inverse(1), 1, places=4)
       self.assertAlmostEqual(func.inverse(2), 0.5, places=4)
       self.assertAlmostEqual(func.inverse(0.5), 2, places=4)
       self.assertAlmostEqual(func.inverse(-10), -0.1, places=4)
       self.assertRaises(ValueError, func.inverse, 0)


    def test_index_sorter(self):
        self.assertEqual(func.index_sorter([10,11,12,13]), [0,1,2,3])
        self.assertEqual(func.index_sorter([10,11,11,13]), [0,1,2,3])
        self.assertEqual(func.index_sorter([14,11,12,13]), [1,2,3,0])
        self.assertEqual(func.index_sorter([-14,11,12,13]), [0,1,2,3])
    

    def test_weigths(self):
       r_0={'0': 1, '1': 1, '2': 1, '3': 1, '4': 1,
      '5': 1, '6': 1, '7': 1, '8': 1, '9': 1,
        '10': 1, '11': 1}
       s_0=[0.08333, 0.08333, 0.08333, 0.08333, 0.08333, 0.08333,
           0.08333, 0.08333, 0.08333, 0.08333, 0.08333, 0.08333]
       calc = func.weigths(r_0)
       test = list(zip(func.spectral_types, s_0))
       for i in range(len(func.spectral_types)): 
            self.assertAlmostEqual(calc[i][1], test[i][1], places=4)
       r_1={'0': 0.08333, '1': 0.08333, '2': 0.08333, '3': 0.08333,
           '4': 0.08333, '5': 0.08333, '6': 0.08333, '7': 0.08333,
             '8': 0.08333, '9': 0.08333,'10': 0.08333, '11': 0.08333}
       s_1=s_0
       calc = func.weigths(r_1)
       test = list(zip(func.spectral_types, s_1))
       for i in range(len(func.spectral_types)): 
            self.assertAlmostEqual(calc[i][1], test[i][1], places=4)
       r_2={'0': 1, '1': 2, '2': 3, '3': 4,
           '4': 4, '5': 3, '6': 2, '7': 1,
             '8': 5, '9': 6,'10': 7, '11': 8}
       s_2=[0.2082816761715844, 0.2082816761715844, 0.1041408380857922,
            0.1041408380857922, 0.06942722539052813, 0.06942722539052813,
            0.0520704190428961, 0.0520704190428961, 0.04165633523431689,
            0.034713612695264066, 0.029754525167369202, 0.02603520952144805]
       calc = func.weigths(r_2)
       test = list(zip(func.spectral_types, s_2))
       for i in range(len(func.spectral_types)): 
            self.assertAlmostEqual(calc[i][1], test[i][1], places=4)     


    def test_mass_class_1(self):
        self.assertAlmostEqual(func.mass_class_1(0.3), 1.122460461, places=4)


    def test_mass_class_2(self):
        self.assertAlmostEqual(func.mass_class_2(1.5), 1.10668192, places=4)


    def test_mass_class_3(self):
        self.assertAlmostEqual(func.mass_class_3(10), 1.753732775, places=4)


    def test_mass_class_4(self):
        self.assertAlmostEqual(func.mass_class_4(100), 0.003125, places=4)


    def test_add_distance(self):
        self.assertAlmostEqual(CatalogObject.add_distance(1000), 1., places=4)
        self.assertAlmostEqual(CatalogObject.add_distance(-1000), 1., places=4)
        self.assertAlmostEqual(CatalogObject.add_distance(1), 1000., places=4)
        self.assertAlmostEqual(CatalogObject.add_distance(10000), 0.1, places=4)

    
    def test_add_cartesian_coordinates(self):
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, 1)[0], 1)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, 1)[1], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, 1)[2], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, np.pi, 1)[0], -1)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, np.pi, 1)[1], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, np.pi, 1)[2], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, -1)[0], -1)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, -1)[1], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(np.pi/2, 0, -1)[2], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(0, np.pi, 1)[0], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(0, np.pi/3, 1)[1], 0)
        self.assertAlmostEqual(CatalogObject.add_cartesian_coordinates(0, np.pi/4, 1)[2], 1)


    def test_add_abs_magnitude(self):
        self.assertAlmostEqual(CatalogObject.add_abs_magnitude(-26.74, 0.000004773), 4.866, places=4)


    def test_add_luminosity(self):
        self.assertAlmostEqual(StelarObject.add_luminosity(4.83), 1.0, places=4)


    def test_add_mass_estimative(self):
        self.assertAlmostEqual(StelarObject.add_mass_estimative(1), 1.0, places=4)


    def test_add_radius_estimative(self):
        self.assertAlmostEqual(StelarObject.add_radius_estimative(1), 1, places=4)

    def test_add_effective_temperature_estimative(self):
        self.assertTrue(abs(
        StelarObject.add_effective_temperature_estimative(1,1)-5780) < 10)

if __name__ == '__main__':
    unittest.main()