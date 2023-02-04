import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import math
from scipy.optimize import curve_fit
import os
import sys
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExploration\
                    WithPython/HiparcusStudies/\
                    hipparcos_space_exploration/src')
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExplorationWithPython/\
                    HiparcusStudies/hipparcos_space_exploration/data')


class CatalogObject():
    """Object containig no approximated features and use only data from the
    datasets """
    parsec_to_ly = 3.26156
    ly_to_meter = 94607183210293

    def __init__(self, HIP, RArad, DErad, Plx, Vmag, BV, VI,
                 pmRA, pmDE, name, measured_st, measured_distance) -> None:
        self.HIP_identifier = int(HIP)
        self.theta = DErad
        self.phi = RArad
        self.vel_theta = pmDE
        self.vel_phi = pmRA
        self.BV = BV
        self.VI = VI
        self.name = name
        self.measured_spectral_type = measured_st
        self.measured_distance = measured_distance
        self.distance = CatalogObject.add_distance(Plx)
        self.distance_ly = self.distance * CatalogObject.parsec_to_ly
        self.x, self.y, self.z = CatalogObject\
            .add_cartesian_coordinates(RArad, DErad, self.distance_ly)
        self.vel_x, self.vel_y, self.vel_z = CatalogObject\
            .add_proper_motion_cartesian(RArad, pmRA, DErad, pmDE,
                                         self.distance_ly *
                                         CatalogObject.ly_to_meter)
        self.visual_magnitude = Vmag
        self.absolute_magnitude = CatalogObject\
            .add_abs_magnitude(Vmag, self.distance)

    def __str__(self):
        return f'Hipparcos catalogue identifier:\t\
         {self.HIP_identifier}.\
        \nObject common name: \t\t {self.name} \
        \nObject spectral type: \t\t {self.measured_spectral_type} \
        \nObject angular position (radians):\t theta -> {self.theta},\
        \n\t\t\t\t\t phi -> {self.phi}\
        \nObject angular aparent velocity:\t dot_theta ->{self.vel_theta}\
        ,\n\t\t\t\t\t dot_phi -> {self.vel_phi}\
        \nObject cartesian aparent velocity:\t\
{self.vel_x, self.vel_y, self.vel_z}\
        \nObject distance (in lightyears):\t {self.distance}.\
        \nObject cartesian position:\t\
        {self.vel_x, self.vel_y, self.vel_z}\
        \nObject visual hipparcos magnitude:\t\t {self.visual_magnitude}\
        \nObject visual B-V magnitude: \t\t {self.BV} \
        \nObject visual V-I magnitude: \t\t {self.VI} \
        \nObject absolute magnitude (calculated from hipparcos magnitude):\t\
        {self.absolute_magnitude}\
        \nObject measured distance: \t\t {self.measured_distance}'

    def add_distance(Plx):
        """Calculate the distances from the object using the formula
        D = 1/plx * 1000 * 3,26156 (in ly) and include
        in the dataframe
        ____________
        Args: the dataframe (df)
        ____________
        Returns: the dataframe with the additinal column

        Parameters
        ----------
        Plx :
            

        Returns
        -------

        """
        return abs(1000.0 / Plx)

    def add_cartesian_coordinates(phi, theta, r):
        """Calculate the cartesian coordinates of the objects for
        posterior plotting
        ____________
        Args: the dataframe (df)
        ____________
        Returns: the dataframe with three additinal columns
        (the x, y, z coordinates)

        Parameters
        ----------
        phi :
            
        theta :
            
        r :
            

        Returns
        -------

        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        x = r * cos_theta * sin_phi
        y = r * sin_theta * sin_phi
        z = r * cos_phi
        return x, y, z

    def add_abs_magnitude(Vmag, r):
        """Calculate the absolute magnitude using the formula
        M = m + 5 - 5LOG_10(d(pc)) notice the distance in parsecs
        ____________
        Args: the dataframe (df)
        ____________
        Returns: the dataframe with the additinal column, the absolute
        magnitude

        Parameters
        ----------
        Vmag :
            
        r :
            

        Returns
        -------

        """
        return Vmag + 5 - 5 * np.log10(r)

    def add_proper_motion_cartesian(phi, vel_phi, theta, vel_theta, r):
        """Calculate the cartesian coordinates of the object proper motion
        ____________

        Parameters
        ----------
        phi :
            
        vel_phi :
            
        theta :
            
        vel_theta :
            
        r :
            

        Returns
        -------

        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        vel_x = r * (cos_phi * sin_theta * vel_theta +
                     sin_phi * cos_theta * vel_phi)
        vel_y = r * (sin_phi * sin_theta * vel_phi -
                     cos_phi * cos_theta * vel_theta)
        vel_z = -r * cos_phi * vel_phi
        return vel_x, vel_y, vel_z


class StelarObject(CatalogObject):
    """ """
    spectral_names = {'wd': 'white dwarf', 'nd': 'dwarf nova',
                      'bd': 'brown dwarf', 'sd': 'sub-dwarf',
                      'ms': 'main sequence star', 'sG': 'Sub giant',
                      'bG': 'blue giant', 'rG': 'red giant',
                      'brG': 'bright giant', 'sGb': 'supergiant Ib',
                      'sGa': 'supergiant Ia', 'hG': 'hypergiant'}
    R_s = 696340000
    cB = 5.670374419 * 10**-8
    cBxR_sxR_s = 27495044249
    L_s = 3.846 * 10**26

    def __init__(self, HIP, RArad, DErad, Plx, Vmag, BV, VI, pmRA,
                 pmDE, name, measured_st, measured_distance) -> None:
        super().__init__(HIP, RArad, DErad, Plx, Vmag, BV, VI, pmRA, pmDE,
                         name, measured_st, measured_distance)
        self.spectral_type_probabilities = func\
            .weigths(func.graph_distances(self.BV, self.absolute_magnitude))
        self.spectral_type = StelarObject\
            .spectral_names[self.spectral_type_probabilities[0][0]]
        self.estimated_luminosity = StelarObject\
            .add_luminosity(self.absolute_magnitude)
        self.estimated_mass = StelarObject\
            .add_mass_estimative(self.estimated_luminosity)
        self.estimated_radius = StelarObject\
            .add_radius_estimative(self.estimated_mass)
        self.estimated_effective_temperature = StelarObject\
            .add_effective_temperature_estimative(self.estimated_luminosity,
                                                  self.estimated_radius)
        self.estimated_ballesteros_temperature = StelarObject\
            .add_ballesteros_temperature(self.BV)
        self.spectral_subtype = StelarObject\
            .add_spectral_subtype(self.spectral_type,
                                  self.estimated_effective_temperature)

    def __str__(self):
        af = '\nAdditional feature'
        temp = super().__str__()
        aux = af + ' spectral type probabilities:\n'
        for i in self.spectral_type_probabilities:
            aux = aux + 'spectral type = '
            aux = aux + str(StelarObject.spectral_names[i[0]])
            aux = aux + ' with probability ' + str(i[1]) + '\n'
        aux = aux + af + ' selected spectral type: \t'
        aux = aux + str(self.spectral_type)
        aux = aux + af + ' estimated luminosity (in solar unities): \t'
        aux = aux + str(self.estimated_luminosity)
        aux = aux + af + ' estimated mass (in solar unities): \t'
        aux = aux + str(self.estimated_mass)
        aux = aux + af + ' estimated temperature (in K): \t'
        aux = aux + str(self.estimated_effective_temperature)
        aux = aux + af + ' estimated radius (in solar unities): \t'
        aux = aux + str(self.estimated_radius)
        aux = aux + af +\
            ' estimated spectral subtype (for main sequence stars only): \t'
        aux = aux + str(self.spectral_subtype)
        temp = temp + aux
        return temp

    def add_luminosity(AbsMag):
        """Estimate the luminosity using L/Ls = 2.511886432**(Ms-M),
        L1_sol = 1, M1_sol = 4.83. Notice that this approximation will
        fail to not consider the acattering of light caused by dust
        particles e gases in the interstelar mean
        ____________
        Args: the absolute magnitude, a float
        ____________
        Returns: the luminosity, a float

        Parameters
        ----------
        AbsMag :
            

        Returns
        -------

        """
        return 2.511886432**(4.83 - AbsMag)

    def add_mass_estimative(estimated_luminosity):
        """Estimate the mass using mass-luminosity estimatives like
        class 1 : M < 0.43 M_s or L < 0.033 -> L/Ls = 0.23(M/Ms)**2.3
        class 2 : 2 M_s > M > 0.43 M_s or 16 > L > 0.033
                    -> L/Ls = (M/M_s)**4
        class 3 : 55 M_s > M > 2 M_s or 1 727 418 > L > 16
                    -> L/L_s = 1.4 (M/M_s)**3.5
        class 4 : M > 55 M_s or L > 1 727 418
                    -> L/L_s = 32000 (M/M_s)
        This aproximation assumes scale relations similar to Eddigton
        relations
        ____________
            Args: the luminosity, a float
        ____________
            Returns: the mass estimative, a float

        Parameters
        ----------
        estimated_luminosity :
            

        Returns
        -------

        """
        if estimated_luminosity < 0.033:
            return func.mass_class_1(estimated_luminosity)
        elif estimated_luminosity < 16:
            return func.mass_class_2(estimated_luminosity)
        elif estimated_luminosity < 1727418:
            return func.mass_class_3(estimated_luminosity)
        else:
            return func.mass_class_4(estimated_luminosity)

    def add_radius_estimative(Mass):
        """Estimate the radius of the star. This approximation assumes that all
        stars are main sequence stars
        ____________
        Args: the object's estimated mass, a float
        ____________
        Returns: the object's estimated radius, a float

        Parameters
        ----------
        Mass :
            

        Returns
        -------

        """
        return func.main_sequence_radius(Mass)

    def add_effective_temperature_estimative(estimated_luminosity,
                                             estimated_radius):
        """Estimate the effective temperatue using the
        Stefan-Boltzman law: L = 4 pi R**2 cB T**4. Its accurancy
        depends on the accurancy of the estimated luminosity and
        radius
        ____________
        Args: the estimated luminosity and the estimated radius, two floats
        ____________
        Returns: the estimated effective temperature, a float

        Parameters
        ----------
        estimated_luminosity :
            
        estimated_radius :
            

        Returns
        -------

        """
        return ((estimated_luminosity*StelarObject.L_s) /
                (4 * np.pi * (estimated_radius*StelarObject.R_s)**2
                * StelarObject.cB))**(0.25)

    def add_spectral_subtype(spectral_type, eff_temperature):
        """Estimate the subtype of the sterlar object.
        Only main sequence type stars are categorized, all others
        will recieve None. This estimative depends on the
        effective temperature precision
        ____________
        Args: the spectral type, a string, and the estimated effective
          temperature, a float
        ____________
        Returns: the spectral subtype, a string or a None type

        Parameters
        ----------
        spectral_type :
            
        eff_temperature :
            

        Returns
        -------

        """
        if spectral_type == 'main sequence star':
            if eff_temperature >= 25000:
                return 'O type'
            elif eff_temperature >= 11000:
                return 'B type'
            elif eff_temperature >= 7500:
                return 'A type'
            elif eff_temperature >= 6000:
                return 'F type'
            elif eff_temperature >= 5000:
                return 'G type'
            elif eff_temperature >= 3500:
                return 'K type'
            else:
                return 'M type'
        else:
            return None

    def add_ballesteros_temperature(B_V):
        """

        Parameters
        ----------
        B_V :
            

        Returns
        -------

        """
        temp = 0.92*B_V
        return 4600*(1 / (temp + 1.7) + 1 / (temp + 0.62))


class func():
    """ """
    spectral_types = ['wd', 'nd', 'bd', 'sd', 'ms', 'sG', 'bG', 'rG', 'brG',
                      'sGb', 'sGa', 'hG']
    wd_inf, wd_sup, wd_inf_value, wd_sup_value = \
        -0.4, 0.5, 6.686965287899504, 12.091313570055563
    nd_inf, nd_sup, nd_inf_value, nd_sup_value = \
        -0.4, 0.8, 4.533166846071749, 9.58911441193521
    bd_inf, bd_sup, bd_inf_value, bd_sup_value = \
        1.5, 2.2, 7.532854231224746, 12.7594698572575
    sd_inf, sd_sup, sd_inf_value, sd_sup_value = \
        -0.1, 1.8, 3.709996149015936, 13.398300676350345
    ms_inf, ms_sup, ms_inf_value, ms_sup_value = \
        -0.28, 1.75, -1.8159962283262188, 12.055671853455351
    sG_inf, sG_sup, sG_inf_value, sG_sup_value = \
        0.3, 1.8, 1.594130917327584, 2.0923925615736767
    bG_inf, bG_sup, bG_inf_value, bG_sup_value = \
        -0.25, 0.8, -3.0932112806553755, 2.23403504078178
    rG_inf, rG_sup, rG_inf_value, rG_sup_value = \
        0.8, 2.0, 1.7804211891878365, -1.6868729788688874
    brG_inf, brG_sup, brG_inf_value, brG_sup_value = \
        -0.4, 2.0, -4.62668901537128, -5.04698276320433
    sGb_inf, sGb_sup, sGb_inf_value, sGb_sup_value = \
        -0.4, 2.5, -5.553638327525648, -7.972627669936597
    sGa_inf, sGa_sup, sGa_inf_value, sGa_sup_value = \
        -0.4, 2.5, -6.737672889332979, -8.130138237716658
    hG_inf, hG_sup, hG_inf_value, hG_sup_value = \
        -0.4, 2.5, -9.122577472503304, -9.687751580042956

    def distance_variations(df, step, max):
        """

        Parameters
        ----------
        df :
            
        step :
            
        max :
            

        Returns
        -------

        """
        data = pd.DataFrame({'distance_ly': [step * i for i in range(4,
                                                                     max)]})
        avg_temperature = []
        avg_luminosity = []
        avg_mass = []
        avg_radius = []
        counts = []
        density = []
        for i in range(4, step * max, step):
            aux = df[(df['distance_ly'] > i) & (df['distance_ly'] < i+step)]
            avg_temperature.append(aux['temperature'].mean())
            avg_luminosity.append(aux['luminosity'].mean())
            avg_mass.append(aux['mass'].mean())
            avg_radius.append(aux['radius'].mean())
            counts.append(aux.shape[0])
            density.append(aux.shape[0] / ((4 * np.pi / 3) *
                                           (step**3 + 3 * step**2 * i
                                           + 3 * step * (i**2))))

        data['avg_temperature'] = avg_temperature
        data['avg_luminosity'] = avg_luminosity
        data['avg_mass'] = avg_mass
        data['avg_radius'] = avg_radius
        data['star_counts'] = counts
        data['star_density'] = density
        return data

    def load_data(number_of_records='all', how='tail'):
        """Load and preprocess the data from the files
        ____________
        Args: the number of records to consider, a integer.
        The default variable 'all' means all the data is considered
              the how variable, how should return the cut dataframe.
              The default is tail meaning the larges paralax (or the
              nearest bodies) will be returned.
        ____________
        Returns: None

        Parameters
        ----------
        number_of_records :
             (Default value = 'all')
        how :
             (Default value = 'tail')

        Returns
        -------

        """
        f_data = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                               '..', 'data', 'hip2.csv'))
        f_name = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                               '..', 'data', 'names.csv'))
        df = pd.read_csv(f_data)  # load the main dataframe
        df.drop(columns=['Unnamed: 0'], inplace=True)
        names_df = (pd.read_csv(f_name))  # load the names dataframe
        names_df['HIP'] = names_df['HIP'].astype(int)
        names_df = names_df[['HIP', 'common name', 'spectral type',
                             'measured distance_ly']]
        # merge the dataframes
        df = pd.merge(df, names_df, how='left', on='HIP')
        df = df[abs(df['Plx']) != 0.0]
        # this columns presents errors, probably were inputted
        df = df[abs(df['B-V']) > 0.0001]
        df['param'] = abs(df['Plx'])
        df.sort_values(by='param', ascending=True, inplace=True)
        df.drop(columns=['param'], inplace=True)
        if number_of_records == 'all':
            return df
        else:
            clause_1 = type(number_of_records) == int
            clause_2 = number_of_records <= df.shape[0]
            clause_3 = number_of_records > 0
            if clause_1 and clause_2 and clause_3:
                if how == 'tail':
                    return df.tail(number_of_records)
                elif how == 'head':
                    return df.head(number_of_records)
                else:
                    raise ValueError(f"how = {how} is not a reconised option.\
                                     Use 'tail' or 'head' instead.")
            else:
                raise ValueError(f'number_of_records = {number_of_records}\
                                 is an invalid number or type.')

    def poly(vec, x):
        """

        Parameters
        ----------
        vec :
            
        x :
            

        Returns
        -------

        """
        sub_values = [value*x**index for index, value in enumerate(vec)]
        return np.sum(sub_values)

    def point_distance(p_1, p_2):
        """Calculate the distance between two points in a 2d environment
        ______________________
        Takes: two tuples containig (x, y) locations of each
        ______________________
        Returns: the distance between these points(a float)

        Parameters
        ----------
        p_1 :
            
        p_2 :
            

        Returns
        -------

        """
        x_1, y_1 = p_1
        x_2, y_2 = p_2
        return np.sqrt((x_1 - x_2)**2 + (y_1 + y_2)**2)

    def graph_distances(x, y):
        """

        Parameters
        ----------
        x :
            
        y :
            

        Returns
        -------

        """
        dist_dict = {}
        if x > func.wd_sup:
            dist_dict['wd'] = func\
                .point_distance((x, y), (func.wd_sup, func.wd_sup_value))
        elif x < func.wd_inf:
            dist_dict['wd'] = func\
                .point_distance((x, y), (func.wd_inf, func.wd_inf_value))
        else:
            dist_dict['wd'] = abs(y - func.wd_curve(x))
        if x > func.nd_sup:
            dist_dict['nd'] = func\
                .point_distance((x, y), (func.nd_sup, func.nd_sup_value))
        elif x < func.nd_inf:
            dist_dict['nd'] = func\
                .point_distance((x, y), (func.nd_inf, func.nd_inf_value))
        else:
            dist_dict['nd'] = abs(y - func.nd_curve(x))
        if x > func.bd_sup:
            dist_dict['bd'] = func\
                .point_distance((x, y), (func.bd_sup, func.bd_sup_value))
        elif x < func.bd_inf:
            dist_dict['bd'] = func\
                .point_distance((x, y), (func.bd_inf, func.bd_inf_value))
        else:
            dist_dict['bd'] = abs(y - func.bd_curve(x))
        if x > func.sd_sup:
            dist_dict['sd'] = func\
                .point_distance((x, y), (func.sd_sup, func.sd_sup_value))
        elif x < func.sd_inf:
            dist_dict['sd'] = func\
                .point_distance((x, y), (func.sd_inf, func.sd_inf_value))
        else:
            dist_dict['sd'] = abs(y - func.sd_curve(x))
        if x > func.ms_sup:
            dist_dict['ms'] = func\
                .point_distance((x, y), (func.ms_sup, func.ms_sup_value))
        elif x < func.ms_inf:
            dist_dict['ms'] = func\
                .point_distance((x, y), (func.ms_inf, func.ms_inf_value))
        else:
            dist_dict['ms'] = abs(y - func.ms_curve(x))
        if x > func.sG_sup:
            dist_dict['sG'] = func\
                .point_distance((x, y), (func.sG_sup, func.sG_sup_value))
        elif x < func.sG_inf:
            dist_dict['sG'] = func\
                .point_distance((x, y), (func.sG_inf, func.sG_inf_value))
        else:
            dist_dict['sG'] = abs(y - func.sG_curve(x))
        if x > func.bG_sup:
            dist_dict['bG'] = func\
                .point_distance((x, y), (func.bG_sup, func.bG_sup_value))
        elif x < func.bG_inf:
            dist_dict['bG'] = func\
                .point_distance((x, y), (func.bG_inf, func.bG_inf_value))
        else:
            dist_dict['bG'] = abs(y - func.bG_curve(x))
        if x > func.rG_sup:
            dist_dict['rG'] = func\
                .point_distance((x, y), (func.rG_sup, func.rG_sup_value))
        elif x < func.rG_inf:
            dist_dict['rG'] = func\
                .point_distance((x, y), (func.rG_inf, func.rG_inf_value))
        else:
            dist_dict['rG'] = abs(y - func.rG_curve(x))
        if x > func.brG_sup:
            dist_dict['brG'] = func\
                .point_distance((x, y), (func.brG_sup, func.brG_sup_value))
        elif x < func.brG_inf:
            dist_dict['brG'] = func\
                .point_distance((x, y), (func.brG_inf, func.brG_inf_value))
        else:
            dist_dict['brG'] = abs(y - func.brG_curve(x))
        if x > func.sGb_sup:
            dist_dict['sGb'] = func\
                .point_distance((x, y), (func.sGb_sup, func.sGb_sup_value))
        elif x < func.sGb_inf:
            dist_dict['sGb'] = func\
                .point_distance((x, y), (func.sGb_inf, func.sGb_inf_value))
        else:
            dist_dict['sGb'] = abs(y - func.sGb_curve(x))
        if x > func.sGa_sup:
            dist_dict['sGa'] = func\
                .point_distance((x, y), (func.sGa_sup, func.sGa_sup_value))
        elif x < func.sGa_inf:
            dist_dict['sGa'] = func\
                .point_distance((x, y), (func.sGa_inf, func.sGa_inf_value))
        else:
            dist_dict['sGa'] = abs(y - func.sGa_curve(x))
        if x > func.hG_sup:
            dist_dict['hG'] = func\
                .point_distance((x, y), (func.hG_sup, func.hG_sup_value))
        elif x < func.hG_inf:
            dist_dict['hG'] = func\
                .point_distance((x, y), (func.hG_inf, func.hG_inf_value))
        else:
            dist_dict['hG'] = abs(y - func.hG_curve(x))
        return dist_dict

    def inverse(x):
        """Invert a number

        Parameters
        ----------
        x :
            

        Returns
        -------

        """
        if x == 0:
            raise ValueError('Division by zero.')
        return 1/x

    def index_sorter(vec):
        """

        Parameters
        ----------
        vec :
            

        Returns
        -------
        type
            

        """
        return list(np.argsort(vec))

    def weigths(dist_dict):
        """Calculate the weigths (weigthed probability) of the star types as
        the inverse of the distanceto the star type lines in the pre-defined
        limits
        ______________________
        Takes: the coordinates of the point and the star type lines limits
        ______________________
        Returns: a normalized vector of the distance of the point to each star
        type lines (a float), which may be interpreted as probabilities

        Parameters
        ----------
        dist_dict :
            

        Returns
        -------

        """
        weights = []
        for value in dist_dict.values():
            if value < 0:
                raise ValueError('Negative distance processing')
            weights.append(func.inverse(value))
        indexes = func.index_sorter(weights)
        temp = list(zip(func.spectral_types, list(weights/np.sum(weights))))
        return [temp[i] for i in indexes[::-1]]

    def mass_class_1(L):
        """

        Parameters
        ----------
        L :
            

        Returns
        -------
        type
            

        """
        mass_estimative = (4.347826087 * L)**(0.434782609)
        return mass_estimative

    def mass_class_2(L):
        """

        Parameters
        ----------
        L :
            

        Returns
        -------
        type
            

        """
        mass_estimative = L**(0.25)
        return mass_estimative

    def mass_class_3(L):
        """

        Parameters
        ----------
        L :
            

        Returns
        -------
        type
            

        """
        mass_estimative = (0.714285714 * L)**(0.285714286)
        return mass_estimative

    def mass_class_4(L):
        """

        Parameters
        ----------
        L :
            

        Returns
        -------
        type
            

        """
        mass_estimative = 0.00003125 * L
        return mass_estimative

    def main_sequence_radius(Mass):
        """Radius estimatives from the mass

        Parameters
        ----------
        Mass :
            

        Returns
        -------

        """
        if Mass < 1:
            Radius = Mass**0.57
        else:
            Radius = Mass**0.8
        return Radius

    def wd_curve(x):
        """The white dwarf's curves
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 0.5 or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([11.094401273671647, 7.007583132580344,
                              -10.027517079625023], x)

    def nd_curve(x):
        """The dwarf novae's curves
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 0.8 or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([6.201701661209501, 4.425502170753053,
                              2.7898356464361895, 2.328670966578929,
                              -7.643465172080826], x)

    def bd_curve(x):
        """The brown dwarf's curves
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2.2 or x < 1.5:
            raise ValueError("Function out of scope")
        else:
            return func.poly([2239.106059628984, -5138.6455473835995,
                              4406.186757861455, -1667.5961454611233,
                              235.18259781148802], x)

    def sd_curve(x):
        """The sub dwarf's curves
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > func.sd_sup or x < func.sd_inf:
            raise ValueError("Function out of scope")
        else:
            return func.poly([4.127494174689552, 4.443023101387477,
                              2.6341535324833836, -2.7099531759586646,
                              -17.289141872604016, 49.01767077806707,
                              -31.506719891530715, -28.81025593579541,
                              48.85197223722716, -23.64042706824589,
                              3.9352857517801714], x)

    def ms_curve(x):
        """The principal sequence's curves
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 1.75 or x < -0.28:
            raise ValueError("Function out of scope")
        else:
            return func.poly([0.38373525720441615, 7.546926803153075,
                              -16.44983605795446, 37.61633179431189,
                              71.67399422667708, -508.0062725300726,
                              1082.3729883178735, -1199.9894588289494,
                              737.8925286377942, -237.6399595506343,
                              31.238352419420224], x)

    def sG_curve(x):
        """The sub giant's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 1.8 or x < 0.3:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-0.5218767420862986, 8.1735562705822,
                              -3.733991352897526], x)

    def bG_curve(x):
        """The blue giant's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 0.8 or x < -0.25:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-1.3185042659684205, 6.235335567109576,
                              -3.1772416018272684, 1.121313471585914,
                              0.05760005073243082], x)

    def rG_curve(x):
        """The red giant's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2.3 or x < 0.8:
            raise ValueError("Function out of scope")
        else:
            return func.poly([5.010153175567576, -10.126351359983651,
                              13.00529569535027, -8.031650990529664,
                              1.6117313567729419], x)

    def brG_curve(x):
        """The red bright giant's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2. or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-3.033661406679939, 3.1510307383965936,
                              -2.0788457083293945], x)

    def sGb_curve(x):
        """The Super giant Ib's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2.5 or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-4.56317876623693, 1.9465035977457026,
                              -1.324113263690228], x)

    def sGa_curve(x):
        """The super giant Ia's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2.5 or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-6.250742347590018, 0.9457284632666354,
                              -0.6789947277269165], x)

    def hG_curve(x):
        """The hypergiant's curve
        ----------
        Receives: x, a real number
        ----------

        Parameters
        ----------
        x :
            

        Returns
        -------
        type
            

        """
        if x > 2.5 or x < -0.4:
            raise ValueError("Function out of scope")
        else:
            return func.poly([-8.895387414026551, 0.4459171030748532,
                              -0.305145107792566], x)


class visual():
    """ """

    def curves(B_V_magnitude,  Abs_magnitude):
        """Get the coefficients of the spectral type curves
        ____________
        Args: The B_V magnitude and the Absolute magnitude
        (the same used to plot the HR diagram)
        ____________
        Returns: 7 arrays, each containing the parameters of the
        polynomial coefficients to fit the HR diagram and define
        the Star type regions

        Parameters
        ----------
        B_V_magnitude :
            
        Abs_magnitude :
            

        Returns
        -------

        """
        vec_locations = []
        # useful polynomials

        def func_2(x, a_0, a_1, a_2):
            """

            Parameters
            ----------
            x :
                
            a_0 :
                
            a_1 :
                
            a_2 :
                

            Returns
            -------

            """
            return a_0 + a_1*x + a_2*x**2

        def func_3(x, a_0, a_1, a_2, a_3):
            """

            Parameters
            ----------
            x :
                
            a_0 :
                
            a_1 :
                
            a_2 :
                
            a_3 :
                

            Returns
            -------

            """
            return a_0 + a_1*x + a_2*x**2 + a_3*x**3

        def func_4(x, a_0, a_1, a_2, a_3, a_4):
            """

            Parameters
            ----------
            x :
                
            a_0 :
                
            a_1 :
                
            a_2 :
                
            a_3 :
                
            a_4 :
                

            Returns
            -------

            """
            return a_0 + a_1*x + a_2*x**2 + a_3*x**3 + a_4*x**4

        def func_10(x, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7,
                    a_8, a_9, a_10):
            """

            Parameters
            ----------
            x :
                
            a_0 :
                
            a_1 :
                
            a_2 :
                
            a_3 :
                
            a_4 :
                
            a_5 :
                
            a_6 :
                
            a_7 :
                
            a_8 :
                
            a_9 :
                
            a_10 :
                

            Returns
            -------

            """
            return a_0 + a_1*x + a_2*x**2 + a_3*x**3 + a_4*x**4 +\
                   a_5*x**5 + a_6*x**6 + a_7*x**7 + a_8*x**8 +\
                   a_9*x**9 + a_10*x**10
        # fitting degree 2 polynomial on the white dwarfs region
        test = pd.DataFrame({'X': np.linspace(-0.4, 0.5, num=100)})
        test['Y'] = test['X'].map(lambda x: 5 * x + 8)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit'] = new_df['x_0'].map(lambda x: 5 * x + 8)
        new_df = new_df[new_df['y_0'] > new_df['limit']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['wd', list(popt.copy()), (-0.4, 0.5)])
        # fitting degree 4 polynomial on the novae dwarfs region
        test = pd.DataFrame({'X': np.linspace(-0.4, 0.8, num=100)})
        test['Y_0'] = test['X'].map(lambda x: 5 * x + 5)
        test['Y_1'] = test['X'].map(lambda x: 5 * x + 9)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0'].map(lambda x: 5 * x + 9)
        new_df['limit_1'] = new_df['x_0'].map(lambda x: 5 * x + 5)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        new_df = new_df[new_df['x_0'] > -0.4]
        new_df = new_df[new_df['x_0'] < .8]
        popt, pcov = curve_fit(func_4, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['nd', list(popt.copy()), (-.4, .8)])
        # fitting degree 4 polynomial on the brown dwarfs region
        test = pd.DataFrame({'X': np.linspace(1.5, 2.2, num=100)})
        test['Y_0'] = test['X'].map(lambda x: 17 * x - 18)
        test['Y_1'] = test['X'].map(lambda x: 4.5 * x - 0.8)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0'].map(lambda x: 17 * x - 17.5)
        new_df['limit_1'] = new_df['x_0'].map(lambda x: 4.5 * x - 0.8)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        new_df = new_df[new_df['x_0'] > 1.5]
        new_df = new_df[new_df['x_0'] < 2.2]
        popt, pcov = curve_fit(func_4, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['bd', list(popt.copy()), (1.5, 2.2)])
        # fitting the subdwarf branch with 3th degree polynomial
        test = pd.DataFrame({'X': np.linspace(-0.1, 1.8, num=50)})
        test['Y_0'] = test['X'].map(lambda x: 5 * x + 5)
        test['Y_1'] = test['X'].map(lambda x: 5 * x + 3.5)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0'].map(lambda x: 5 * x + 5)
        new_df['limit_1'] = new_df['x_0'].map(lambda x: 5 * x + 3.5)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_10, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['sd', list(popt.copy()), (-.1, 1.8)])
        # fitting the Main sequence stars to a 10-th order polynomial
        test = pd.DataFrame({'X': np.linspace(-0.28, 1.75, num=100)})
        test['Y_0'] = test['X'].map(lambda x: 6.5 * x + 2)
        test['Y_1'] = test['X'].map(lambda x: 6.5 * x - 1.5)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0'].map(lambda x: 6.5 * x + 2)
        new_df['limit_1'] = new_df['x_0'].map(lambda x: 6.5 * x - 1.5)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        new_df = new_df[new_df['x_0'] > -0.28]
        new_df = new_df[new_df['x_0'] < 1.75]
        popt, pcov = curve_fit(func_10, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['ms', list(popt.copy()), (-0.28, 1.75)])
        # fitting the subgiant branch with 2th degree polynomial
        test = pd.DataFrame({'X': np.linspace(0.3, 1.8, num=50)})
        test['Y_0'] = test['X'].map(lambda x: -5 * (x - 1.2)**2 + 5.5)
        test['Y_1'] = test['X'].map(lambda x: -2 * (x - 1.2)**2 + 3)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -5 * (x - 1.2)**2 + 5.5)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: -2 * (x - 1.2)**2 + 3)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['sG', list(popt.copy()), (0.3, 1.8)])
        # fitting the blue-giant branch with 4th degree polynomial
        test = pd.DataFrame({'X': np.linspace(-0.25, 0.8, num=50)})
        test['Y_0'] = test['X'].map(lambda x: 5 * x - 3)
        test['Y_1'] = test['X'].map(lambda x: 5 * x - 0.5)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: 5 * x - 3)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: 5 * x - 0.5)
        new_df = new_df[new_df['y_0'] > new_df['limit_0']]
        new_df = new_df[new_df['y_0'] < new_df['limit_1']]
        popt, pcov = curve_fit(func_4, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['bG', list(popt.copy()), (-0.25, 0.8)])
        # fitting the red-giant branch with 4th degree polynomial
        test = pd.DataFrame({'X': np.linspace(.8, 2.3, num=100)})
        test['Y_0'] = test['X'].map(lambda x: -5 * (x - 1.0)**2 + 3)
        test['Y_1'] = test['X'].map(lambda x: -3 * x + 3.5)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -5 * (x - 1.0)**2 + 3)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: -3 * x + 3.5)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_4, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['rG', list(popt.copy()), (.8, 2.3)])
        # fitting a 4-th polynomial to the Bright Giant branch
        test = pd.DataFrame({'X': np.linspace(-0.4, 2, num=50)})
        test['Y_0'] = test['X']\
            .map(lambda x: -2.5 * (x - 0.75)**2 - 1)
        test['Y_1'] = test['X']\
            .map(lambda x: -1.5 * (x - 0.75)**2 - 3)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -2.5 * (x - 0.75)**2 - 1)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: -1.5 * (x - 0.75)**2 - 3)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['brG', list(popt.copy()), (-0.4, 2.)])
        # fitting a 2-th polynomial to the super giant Ib branch
        test = pd.DataFrame({'X': np.linspace(-0.4, 2.5, num=50)})
        test['Y_0'] = test['X']\
            .map(lambda x: -1.5 * (x - 0.75)**2 - 3)
        test['Y_1'] = test['X']\
            .map(lambda x: -1.0 * (x - 0.75)**2 - 5)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -1.5 * (x - 0.75)**2 - 3)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: -1.0 * (x - 0.75)**2 - 5)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['sGb', list(popt.copy()), (-0.4, 2.5)])
        # fitting a 2-th polynomial to the super giant Ia branch
        test = pd.DataFrame({'X': np.linspace(-0.4, 2.5, num=50)})
        test['Y_0'] = test['X']\
            .map(lambda x: -1.0 * (x - 0.75)**2 - 5)
        test['Y_1'] = test['X']\
            .map(lambda x: -.5 * (x - 0.75)**2 - 7)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -1.0 * (x - 0.75)**2 - 5)
        new_df['limit_1'] = new_df['x_0']\
            .map(lambda x: -.5 * (x - 0.75)**2 - 7)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        new_df = new_df[new_df['y_0'] > new_df['limit_1']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['sGa', list(popt.copy()), (-0.4, 2.5)])
        # fitting a 2-th polynomial to the hyper giant branch
        test = pd.DataFrame({'X': np.linspace(-0.4, 2.5, num=50)})
        test['Y_0'] = test['X'].map(lambda x: -.5 * (x - 0.75)**2 - 7)
        new_df = pd.DataFrame({'x_0': B_V_magnitude, 'y_0': Abs_magnitude})
        new_df['limit_0'] = new_df['x_0']\
            .map(lambda x: -.5 * (x - 0.75)**2 - 7)
        new_df = new_df[new_df['y_0'] < new_df['limit_0']]
        popt, pcov = curve_fit(func_2, new_df['x_0'], new_df['y_0'])
        vec_locations.append(['hG', list(popt.copy()), (-0.4, 2.5)])
        return vec_locations

    def HR_diagram(B_V_magnitude, Abs_magnitude, names, curves=False,
                   points=False, only_named=False):
        """Plot the HR-diagram, plot the limits of the regions considered
        to each star type in the HR diagram, plot a polynomial regression
        of each subset considering the limits and return those regressions
        cefficients
        ____________
        Args: The B_V magnitude and the Absolute magnitude (the same used
        to plot the HR diagram)
        ____________
        Returns: 7 arrays, each containing the parameters of the polynomial
        coefficients to fit the HR diagram and define the Star type regions

        Parameters
        ----------
        B_V_magnitude :
            
        Abs_magnitude :
            
        names :
            
        curves :
             (Default value = False)
        points :
             (Default value = False)
        only_named :
             (Default value = False)

        Returns
        -------

        """
        if only_named:
            B_V = []
            Abs = []
            label = []
            for i, name in enumerate(names):
                if name is not np.nan:
                    B_V.append(B_V_magnitude[i])
                    Abs.append(Abs_magnitude[i])
                    label.append(name)
            data = pd.DataFrame({'X': B_V, 'Y': Abs, 'l': label})
            F1 = {"family": "serif", "weight": "bold", "color": "black",
                  "size": 18}
            F2 = {"family": "serif", "weight": "bold", "color": "black",
                  "size": 20}
            norm = mpl.colors.Normalize(vmin=np.min(B_V), vmax=np.max(B_V))
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, row in data.iterrows():
                ax.scatter(row['X'], row['Y'], c=row['X'], cmap="YlOrRd",
                           marker='*', s=20, linewidth=0.0001, norm=norm,
                           alpha=0.8)
                plt.annotate(row['l'], (row['X']+0.05, row['Y']),
                             color='white', size=8)
        else:
            data = pd.DataFrame({'X': B_V_magnitude, 'Y': Abs_magnitude})
            F1 = {"family": "serif", "weight": "bold", "color": "black",
                  "size": 18}
            F2 = {"family": "serif", "weight": "bold", "color": "black",
                  "size": 20}
            norm = mpl.colors.Normalize(vmin=np.min(B_V_magnitude),
                                        vmax=np.max(B_V_magnitude))
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(data['X'], data['Y'], c=data['X'], cmap="YlOrRd",
                       marker='.', s=5, linewidth=0.0001, norm=norm,
                       alpha=0.8)
        plt.xlabel("Color Index (B-V)", fontdict=F1)
        plt.ylabel("Absolute Magnitude (Mv)", fontdict=F1)
        plt.title("Hertzsprung-Russell diagram", fontdict=F2)
        plt.xlim(-0.45, 5.2)
        plt.ylim(15.1, -15.1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        Color_BC = plt.gca()
        Color_BC.set_facecolor("black")
        plt.tight_layout()
        if curves:
            # white dwarfs
            test = pd.DataFrame({'X': np.linspace(func.wd_inf,
                                                  func.wd_sup, num=100)})
            test['Y'] = test['X'].map(func.wd_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='darkblue', ax=ax)
            ax.annotate(text='white dwarfs', color='white', alpha=1,
                        rotation=-30, size=8, xy=(-0.1, 12))
            # dwarf novae
            test = pd.DataFrame({'X': np.linspace(func.nd_inf,
                                                  func.nd_sup, num=100)})
            test['Y'] = test['X'].map(func.nd_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='dodgerblue', ax=ax)
            ax.annotate(text='novae dwarfs', color='white', alpha=1,
                        rotation=-30, size=8, xy=(0.0, 7.8))
            # brown dwarfs
            test = pd.DataFrame({'X': np.linspace(func.bd_inf,
                                                  func.bd_sup, num=100)})
            test['Y'] = test['X'].map(func.bd_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='darkgoldenrod',
                         ax=ax)
            ax.annotate(text='brown dwarfs', color='white', alpha=1,
                        rotation=-70, size=8, xy=(2.05, 12))
            # subdwarfs
            test = pd.DataFrame({'X': np.linspace(func.sd_inf,
                                                  func.sd_sup, num=100)})
            test['Y'] = test['X'].map(func.sd_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='mediumspringgreen',
                         ax=ax)
            ax.annotate(text='subdwarfs', color='white', alpha=1,
                        rotation=-30, size=8, xy=(1., 10.2))
            # main sequence stars
            test = pd.DataFrame({'X': np.linspace(func.ms_inf,
                                                  func.ms_sup, num=100)})
            test['Y'] = test['X'].map(func.ms_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='cyan', ax=ax)
            ax.annotate(text='main sequence stars', color='white', alpha=1,
                        rotation=-50, size=8, xy=(1.4, 13.2))
            # subgiants
            test = pd.DataFrame({'X': np.linspace(func.sG_inf, func.sG_sup,
                                                  num=100)})
            test['Y'] = test['X'].map(func.sG_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='orange', ax=ax)
            ax.annotate(text='subgiants', color='white', alpha=1,
                        rotation=25, size=8, xy=(1.4, 4.15))
            # blue giants
            test = pd.DataFrame({'X': np.linspace(func.bG_inf,
                                                  func.bG_sup, num=50)})
            test['Y'] = test['X'].map(func.bG_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='blue', ax=ax)
            ax.annotate(text='blue giants', color='black', alpha=1,
                        rotation=-30, size=8, xy=(0, 1.3))
            # red Giants
            test = pd.DataFrame({'X': np.linspace(func.rG_inf,
                                                  func.rG_sup, num=100)})
            test['Y'] = test['X'].map(func.rG_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='red', ax=ax)
            ax.annotate(text='red giants', color='white', alpha=1,
                        rotation=15, size=8, xy=(1.8, -1.5))
            # bright giants
            test = pd.DataFrame({'X': np.linspace(func.brG_inf,
                                                  func.brG_sup, num=100)})
            test['Y'] = test['X'].map(func.brG_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='chocolate', ax=ax)
            ax.annotate(text='brigh giants', color='white', alpha=1,
                        rotation=25, size=8, xy=(1.6, -3.8))
            # super giants Ib
            test = pd.DataFrame({'X': np.linspace(func.sGb_inf,
                                                  func.sGb_sup, num=50)})
            test['Y'] = test['X'].map(func.sGb_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='lightyellow', ax=ax)
            ax.annotate(text='super giants Ib', color='white', alpha=1,
                        rotation=20, size=8, xy=(1.6, -5.1))
            # super giants Ia
            test = pd.DataFrame({'X': np.linspace(func.sGa_inf,
                                                  func.sGa_sup, num=50)})
            test['Y'] = test['X'].map(func.sGa_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='yellow', ax=ax)
            ax.annotate(text='super giants Ia', color='white', alpha=1,
                        rotation=10, size=8, xy=(1.6, -6.8))
            # hypergiants
            test = pd.DataFrame({'X': np.linspace(func.hG_inf,
                                                  func.hG_sup, num=50)})
            test['Y'] = test['X'].map(func.hG_curve)
            sns.lineplot(x=test['X'], y=test['Y'], color='white', ax=ax)
            ax.annotate(text='hypergiants', color='white', alpha=1,
                        rotation=5, size=8, xy=(1.6, -9.4))
        if points:
            X = [func.wd_inf, func.wd_sup, func.nd_inf, func.nd_sup,
                 func.sd_inf, func.sd_sup, func.ms_inf, func.ms_sup,
                 func.sG_inf, func.sG_sup, func.bG_inf, func.bG_sup,
                 func.rG_inf, func.rG_sup, func.brG_inf, func.brG_sup,
                 func.sGb_inf, func.sGb_sup, func.sGa_inf, func.sGa_sup,
                 func.hG_inf, func.hG_sup, func.bd_inf, func.bd_sup]
            Y = [func.wd_inf_value, func.wd_sup_value, func.nd_inf_value,
                 func.nd_sup_value, func.sd_inf_value, func.sd_sup_value,
                 func.ms_inf_value, func.ms_sup_value, func.sG_inf_value,
                 func.sG_sup_value, func.bG_inf_value, func.bG_sup_value,
                 func.rG_inf_value, func.rG_sup_value, func.brG_inf_value,
                 func.brG_sup_value, func.sGb_inf_value, func.sGb_sup_value,
                 func.sGa_inf_value, func.sGa_sup_value, func.hG_inf_value,
                 func.hG_sup_value, func.bd_inf_value, func.bd_sup_value]
            sns.scatterplot(x=X, y=Y, marker='X', color='white', size=10,
                            ax=ax)
        plt.legend([], [], frameon=False)
        plt.show()
        return None

    def plot_map(Stars_vector):
        """Plot a 3D-map (which is navigable using the right-left mouse
        buttons) with color defined by star type and temperature, size
        define by the star radius and the map centered in the sun
        ____________
        Args: the dataframe (df) and two distance limits, will be plotted
        all stars distant beteween the estipulated limits
        ____________
        Returns: None

        Parameters
        ----------
        Stars_vector :
            

        Returns
        -------

        """
        color_map = {'white dwarf': 'white', 'dwarf nova': 'deepskyblue',
                     'brown dwarf': 'peru', 'sub-dwarf': 'mediumspringgren',
                     'main sequence star': 'yellow', 'Sub giant': 'orange',
                     'blue giant': 'blue', 'red giant': 'coral',
                     'bright giant': 'pink', 'supergiant Ib': 'yellowgreen',
                     'supergiant Ia': 'lime', 'hypergiant': 'ghostwhite'}
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(0.0, 0.0, 0.0, s=10, marker='*', c='white')
        for temp in Stars_vector:
            ax.scatter(temp.x, temp.y, temp.z, s=10*temp.estimated_radius,
                       cmap=color_map[temp.spectral_type], alpha=1, marker='o')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        plt.grid(b=None)
        Color_BC = plt.gca()
        Color_BC.set_facecolor("black")
        # plt.savefig('30.png')
        plt.show()

    def plot_map_df(df):
        """Plot a 3D-map (which is navigable using the right-left mouse
        buttons) with color defined by star type and temperature, size
        define by the star radius and the map centered in the sun
        ____________
        Args: the dataframe (df) and two distance limits, will be plotted all
        stars distant beteween the estipulated limits
        ____________
        Returns: None

        Parameters
        ----------
        df :
            

        Returns
        -------

        """
        color_map = {'white dwarf': 'white', 'dwarf nova': 'deepskyblue',
                     'brown dwarf': 'peru', 'sub-dwarf': 'mediumspringgren',
                     'main sequence star': 'yellow', 'Sub giant': 'orange',
                     'blue giant': 'blue', 'red giant': 'coral',
                     'bright giant': 'pink', 'supergiant Ib': 'yellowgreen',
                     'supergiant Ia': 'lime', 'hypergiant': 'ghostwhite'}
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(0.0, 0.0, 0.0, s=10, marker='*', c='white')
        ax.text(0, 0, 0.1, 'Sun', color='white', size=6)
        for i, row in df.iterrows():
            if row['common name'] is np.nan:
                ax.scatter(row.X, row.Y, row.Z, s=10*row.radius,
                           cmap=color_map[row.spectral_type], alpha=1,
                           marker='o')
            else:
                ax.scatter(row.X, row.Y, row.Z, s=10*row.radius,
                           cmap=color_map[row.spectral_type], alpha=1,
                           marker='*')
                ax.text(row.X, row.Y, row.Z+0.1,
                        row['common name'], color='white', size=6)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        plt.grid(b=None)
        Color_BC = plt.gca()
        Color_BC.set_facecolor("black")
        plt.show()

    def local_sky(init=(2*np.pi/5, 2*np.pi/5), end=(3*np.pi/5, 3*np.pi/5),
                  mag_limit=(-16, 11), names=True):
        """Plots a square from init to end showing the position of
        stars in the magnitude range given by the argument mag_limit.
        ____________
        Args: A tuple, init. The initial values of theta and phi. A tuple, end.
        The final values of theta and phi. A tuple, mag_limit. The limits of
        visual magnitude to consider. A boolean, names. Wheter to write down
        the knwon names or not.
        Returns: None

        Parameters
        ----------
        init :
             (Default value = (2*np.pi/5)
        2*np.pi/5) :
            
        end :
             (Default value = (3*np.pi/5)
        3*np.pi/5) :
            
        mag_limit :
             (Default value = (-16)
        11) :
            
        names :
             (Default value = True)

        Returns
        -------

        """
        df = data_analysis.create_dataframe()
        temp = df[(df.Vmag > mag_limit[0]) & (df.Vmag < mag_limit[1])]
        temp = temp[(temp['DErad'] >= init[1]) & (temp['DErad'] <= end[1])]
        temp = temp[(temp['RArad'] >= init[0]) & (temp['RArad'] <= end[0])]
        color_map = {'white dwarf': 'white', 'dwarf nova': 'deepskyblue',
                     'brown dwarf': 'peru', 'sub-dwarf': 'mediumspringgren',
                     'main sequence star': 'yellow', 'Sub giant': 'orange',
                     'blue giant': 'blue', 'red giant': 'coral',
                     'bright giant': 'pink', 'supergiant Ib': 'yellowgreen',
                     'supergiant Ia': 'lime', 'hypergiant': 'ghostwhite'}
        fig, ax = plt.subplots()
        if names:
            for i, row in temp.iterrows():
                if row['common name'] is not np.nan:
                    ax.scatter(row.DErad, row.RArad,
                               cmap=color_map[row.spectral_type],
                               s=30 - row.Vmag,
                               marker='*')
                    ax.annotate(row['common name'],
                                (row.DErad - 0.05, row.RArad + 0.05),
                                color='white', size=6)
                else:
                    ax.scatter(row.DErad, row.RArad,
                               cmap=color_map[row.spectral_type],
                               s=0.1 * (15 - row.Vmag),
                               marker='o')
        else:
            for i, row in temp.iterrows():
                ax.scatter(row.RArad, row.DErad,
                           cmap=color_map[row.spectral_type],
                           s=15 - row.Vmag,
                           marker='.')
        ax.set_xlabel('Phi')
        ax.set_ylabel('Theta')
        Color_BC = plt.gca()
        Color_BC.set_facecolor("black")
        plt.show()

    def local_plane(R):
        """Plots a the projection of star in the plane phi = 0
        ____________
        Args: a float, R. The distance in ly to consider for the projection
        _____________
        Returns: None

        Parameters
        ----------
        R :
            

        Returns
        -------

        """
        df = data_analysis.create_dataframe()
        temp = df[df.distance_ly <= R]
        color_map = {'white dwarf': 'white', 'dwarf nova': 'deepskyblue',
                     'brown dwarf': 'peru', 'sub-dwarf': 'mediumspringgren',
                     'main sequence star': 'yellow', 'Sub giant': 'orange',
                     'blue giant': 'blue', 'red giant': 'coral',
                     'bright giant': 'pink',
                     'supergiant Ib': 'yellowgreen', 'supergiant Ia': 'lime',
                     'hypergiant': 'ghostwhite'}
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for i, row in temp.iterrows():
            if row['common name'] is not np.nan:
                ax.scatter(row.RArad, row.distance_ly*np.cos(row.DErad),
                           cmap=color_map[row.spectral_type],
                           s=15 - row.Vmag,
                           marker='*')
                ax.annotate(row['common name'],
                            (row.RArad + 0.05,
                            row.distance_ly*np.cos(row.DErad) + 0.05),
                            color='white', size=6)
            else:
                ax.scatter(row.RArad, row.distance_ly*np.cos(row.DErad),
                           cmap=color_map[row.spectral_type],
                           s=15 - row.Vmag,
                           marker='.')
        Color_BC = plt.gca()
        Color_BC.set_facecolor("black")
        ax.set(facecolor='white')
        plt.show()


class data_analysis():
    """Another approach to the problem is to build dataframes to evaluate
    the system's properties. Instead of building objects with star-like
    properties, here we build a dataframe with all this objects and
    properties and build a data-alalisis framework to explore further the
    data.

    Parameters
    ----------

    Returns
    -------

    """

    def create_dataframe(size='all', sort='asc', by='distance_ly'):
        """create a dataframe with all data and all approxiamtions made in
          the OOP   apporach.
        _______
            Args: The size of the data frame, a integer or string. The default
            'all' means all data is considered
            ____________
            Returns: the whole dataframe

        Parameters
        ----------
        size :
             (Default value = 'all')
        sort :
             (Default value = 'asc')
        by :
             (Default value = 'distance_ly')

        Returns
        -------

        """
        df = func.load_data(number_of_records='all')
        df['distance_pc'] = df['Plx'].apply(CatalogObject.add_distance)
        df['distance_ly'] = df['distance_pc']*CatalogObject.parsec_to_ly
        if sort == 'asc':
            df.sort_values(by=by, ascending=True, inplace=True)
        elif sort == 'desc':
            df.sort_values(by=by, ascending=False, inplace=True)
        elif sort is None:
            pass
        else:
            raise ValueError('Unreconized [sort] argument.')
        if type(size) == int:
            df = df[0:size]
        Abs = []
        spectral_type = []
        for i, row in df.iterrows():
            temp = CatalogObject.add_abs_magnitude(row.Vmag, row.distance_pc)
            Abs.append(temp)
            spectral_type.append(StelarObject.spectral_names[func.weigths(
                func.graph_distances(row['B-V'], temp))[0][0]])
        df['abs_magnitude'] = Abs
        df['spectral_type'] = spectral_type
        df['luminosity'] = df['abs_magnitude'].apply(
            StelarObject.add_luminosity)
        df['mass'] = df['luminosity'].apply(StelarObject.add_mass_estimative)
        df['radius'] = df['mass'].apply(StelarObject.add_radius_estimative)
        temperature = []
        subtype = []
        x, y, z = [], [], []
        for i, row in df.iterrows():
            aux = StelarObject.add_effective_temperature_estimative(
                row.luminosity, row.radius)
            temperature.append(aux)
            subtype.append(StelarObject.add_spectral_subtype(
                row.spectral_type, aux))
            temp_x, temp_y, temp_z = CatalogObject.add_cartesian_coordinates(
                row.RArad, row.DErad, row.distance_ly)
            x.append(temp_x)
            y.append(temp_y)
            z.append(temp_z)
        df['temperature'] = temperature
        df['sub_type'] = subtype
        df['X'] = x
        df['Y'] = y
        df['Z'] = z
        return df

    def distance_variation_features(df, step=1, max=300, where='all'):
        """Add to the dataframe the approaches made for the stelar objects
        and return a dataframe ordered by distance containing all records
        in between 0 and max counting by the step argument.
            ____________
            Args: The data frame. A integer, the distance step. a Integer,
        the max distance to consider. A string, what to consider. If
        where = 'all' then all spectral type will be considered. If
        where is some key or value of StelarObject.spectral_names
        the dataframe will further be filtered by the name of the
        star type given as argument.
            ____________
            Returns: the filtered dataframe

        Parameters
        ----------
        df :
            
        step :
             (Default value = 1)
        max :
             (Default value = 300)
        where :
             (Default value = 'all')

        Returns
        -------

        """
        if where == 'all':
            return func.distance_variations(df, step, max)
        elif where in StelarObject.spectral_names.values():
            return func.distance_variations(df[df['spectral_type'] == where],
                                            step, max)
        elif where in StelarObject.spectral_names.keys():
            return func.distance_variations(df[
                df['spectral_type'] == StelarObject.spectral_names[where]],
                  step, max)
        else:
            raise ValueError("Unreconized 'where' argument")

    def plot_bexenplots_degree_distance(df, feat_to_plot, inf_distance,
                                        sup_distance):
        """Plot boxenplots of a object feature in a distance interval
        ____________
        Args: the dataframe (df), the feature (to plot) name and two distance
        limits, will be plotted all stars distant between the estipulated
        limits
        ____________
        Returns: None

        Parameters
        ----------
        df :
            
        feat_to_plot :
            
        inf_distance :
            
        sup_distance :
            

        Returns
        -------

        """
        df.sort_values(by='Distance', ascending=True, inplace=True,
                       ignore_index=True)
        temp = df[df['Distance'] <= sup_distance]
        temp = temp[temp['Distance'] >= inf_distance]
        fig, ax = plt.subplots(3, 1, figsize=(16, 14))
        plt.subplots_adjust(wspace=.3, hspace=.3)
        g_0 = sns.boxenplot(data=temp, y=feat_to_plot, x='RArad', ax=ax[0],
                            color='yellow')
        g_1 = sns.boxenplot(data=temp, y=feat_to_plot, x='DErad', ax=ax[1],
                            color='orange')
        g_2 = sns.boxenplot(data=temp, y=feat_to_plot, x='Distance',
                            ax=ax[2], color='violet')
        ax[0].set_xticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi,
                          3*math.pi/2], ['180', '270', '0', '90', '180',
                                         '270'])
        ax[1].set_xticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi],
                         ['180', '270', '0', '90', '180'])
        ax[0].set_xlabel('Degrees')
        ax[1].set_xlabel('Degrees')
        ax[2].set_xlabel('measured distance_ly')
        plt.show()
        return None

    def plot_lineplots_degree_distance(df, feat_to_plot, inf_distance,
                                       sup_distance, hue=None):
        """Plot lin'eplots of a numerical feature in a distance
          interval
        ____________
        Args: the dataframe (df), the feature (to plot) name and two
          distance limits, will be plotted all stars distant between
            the estipulated limits
        ____________
        Returns: None

        Parameters
        ----------
        df :
            
        feat_to_plot :
            
        inf_distance :
            
        sup_distance :
            
        hue :
             (Default value = None)

        Returns
        -------

        """
        df.sort_values(by='distance_ly', ascending=True, inplace=True,
                       ignore_index=True)
        temp = df[df['distance_ly'] <= sup_distance]
        temp = temp[temp['distance_ly'] >= inf_distance]
        fig, ax = plt.subplots(3, 1, figsize=(16, 14))
        plt.subplots_adjust(wspace=.3, hspace=.3)
        if hue is None:
            g_0 = sns.lineplot(data=temp, y=feat_to_plot, x='RArad',
                               ax=ax[0], color='yellow')
            g_1 = sns.lineplot(data=temp, y=feat_to_plot, x='DErad',
                               ax=ax[1], color='orange')
            g_2 = sns.lineplot(data=temp, y=feat_to_plot, x='distance_ly',
                               ax=ax[2], color='violet')
        else:
            g_0 = sns.lineplot(data=temp, y=feat_to_plot, x='RArad',
                               ax=ax[0], color='yellow', hue=hue)
            g_1 = sns.lineplot(data=temp, y=feat_to_plot, x='DErad',
                               ax=ax[1], color='orange', hue=hue)
            g_2 = sns.lineplot(data=temp, y=feat_to_plot, x='distance_ly',
                               ax=ax[2], color='violet', hue=hue)
        ax[0].set_xticks([-math.pi, -math.pi/2, 0, math.pi/2,
                         math.pi, 3*math.pi/2], ['180', '270',
                         '0', '90', '180', '270'])
        ax[1].set_xticks([-math.pi, -math.pi/2, 0, math.pi/2, math.pi],
                         ['180', '270', '0', '90', '180'])
        ax[0].set_xlabel('Degrees')
        ax[1].set_xlabel('Degrees')
        ax[2].set_xlabel('Distance (ly)')
        plt.show()
        return None
