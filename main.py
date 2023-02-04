import sys
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExplorationWithPython/HiparcusStudies/hipparcos_space_exploration/src')  # noqa: E501
sys.path.insert(0, '/home/ricardo/Desktop/SpaceExplorationWithPython/HiparcusStudies/hipparcos_space_exploration/data')  # noqa: E501
from tools import CatalogObject
from tools import StelarObject
from tools import func
from tools import visual
from tools import data_analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pd.options.display.max_columns = None  # print all columns
pd.options.display.max_rows = None  # print all rows


def OOP_demonstration():
    """ Test the object oriented programing paradigm apllied to the
    package. You will recieve back the features of a object randomly
    chosen from the data"""
    print('The OOP approach presents a StarObject object which is inherited\
 from the parent class CatalogObject. The difference between the two is that\
 the CatalogObject presents only calculations directly derivable from the\
 dataset. For example, distances are taken directly from paralaxis\
 measurements. The StarObject inherits the CatalogObject and also presents\
 features calculated from approximations related to the position of the\
 object in the HR diagram.')
    print('To build the CatalogObject is used the hipparcos dataset\
 taken from the ESA website and an additional file containing the hipparcos\
 indexes, common name, measured spectral type, measured distance (in 1997)\
 and source from that information (generally the www.universeguide.com\
 website). Now I will ask the hiparccos identifier to instantiate those\
 objects with the data related to the inputted identifyer.')
    df = func.load_data(number_of_records='all', how='tail')
    index = input('Type the (integer) hipparcos identifier:\t')
    if len(index) == 0:
        index = np.random.randint(1, df.shape[0])
    else:
        index = int(index)
    item = df[df.HIP == index]
    test = StelarObject(HIP=item.iloc[0]['HIP'], RArad=item.iloc[0]['RArad'],
                        DErad=item.iloc[0]['DErad'], Plx=item.iloc[0]['Plx'],
                        Vmag=item.iloc[0]['Vmag'], BV=item.iloc[0]['B-V'],
                        VI=item.iloc[0]['V-I'], pmRA=item.iloc[0]['pmRA'],
                        pmDE=item.iloc[0]['pmDE'],
                        name=item.iloc[0]['common name'],
                        measured_st=item.iloc[0]['spectral type'],
                        measured_distance=item.iloc[0]['measured distance_ly'])
    print(test)
    return None


def data_analisis_demonstration():
    """Demonstration of using the code in an data analysis
    directed approach."""
    print('Loading the data to a pandas dataframe')
    df = data_analysis.create_dataframe(size=30000, sort='asc',
                                        by='distance_ly')
    print(df.head())
    input('The data loaded into the dataframe is very extensive and in this \
demosntration we shall need only the paralax (plx), the visual magnitude \
(Vmag), and the B-V magnitude (B-V). But before that, type anything to \
explore the variation of the average aproximated features with the distance \
from the solar system (first 500 light-years).')

    def avg_variations(df, step=1, max=300, where='all', kind='line'):
        """

        Parameters
        ----------
        df :
            
        step :
             (Default value = 1)
        max :
             (Default value = 300)
        where :
             (Default value = 'all')
        kind :
             (Default value = 'line')

        Returns
        -------

        """
        print('Processing the data.')
        data = data_analysis.distance_variation_features(df, step=step,
                                                         max=max,
                                                         where='all')
        data_wd = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='wd')
        data_nd = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='nd')
        data_bd = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='bd')
        data_sd = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='sd')
        data_ms = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='ms')
        data_sG = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='sG')
        data_bG = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='bG')
        data_rG = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='rG')
        data_brG = data_analysis.distance_variation_features(df, step=step,
                                                             max=max,
                                                             where='brG')
        data_sGb = data_analysis.distance_variation_features(df, step=step,
                                                             max=max,
                                                             where='sGb')
        data_sGa = data_analysis.distance_variation_features(df, step=step,
                                                             max=max,
                                                             where='sGa')
        data_hG = data_analysis.distance_variation_features(df, step=step,
                                                            max=max,
                                                            where='hG')
        data.fillna(0, inplace=True)
        data_wd.fillna(0, inplace=True)
        data_nd.fillna(0, inplace=True)
        data_bd.fillna(0, inplace=True)
        data_sd.fillna(0, inplace=True)
        data_ms.fillna(0, inplace=True)
        data_sG.fillna(0, inplace=True)
        data_bG.fillna(0, inplace=True)
        data_rG.fillna(0, inplace=True)
        data_brG.fillna(0, inplace=True)
        data_sGb.fillna(0, inplace=True)
        data_sGa.fillna(0, inplace=True)
        data_hG.fillna(0, inplace=True)
        datasets = {'all': data, 'wd': data_wd, 'nd': data_nd, 'bd': data_bd,
                    'sd': data_sd, 'ms': data_ms, 'sG': data_sG,
                    'bG': data_bG, 'rG': data_rG, 'brG': data_brG,
                    'sGb': data_sGb, 'sGa': data_sGa, 'hG': data_hG}
        colors = {'all': 'black', 'wd': 'silver', 'nd': 'red', 'bd': 'sienna',
                  'sd': 'orange', 'ms': 'gold', 'sG': 'olivedrab',
                  'bG': 'lawngreen', 'rG': 'aquamarine', 'brG': 'deepskyblue',
                  'sGb': 'royalblue', 'sGa': 'rebeccapurple', 'hG': 'magenta'}
        labels = StelarObject.spectral_names
        labels['all'] = 'all objects'

        def plot_density_variation():
            """ """
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plt.subplots_adjust(wspace=.3, hspace=.3)
            if kind == 'line':
                print(labels)
                for index, value in datasets.items():
                    sns.lineplot(data=value, x='distance_ly', y='star_density',
                                 ax=ax, color=colors[index],
                                 label=labels[index])
            elif kind == 'scatter':
                for index, value in datasets.items():
                    sns.scatterplot(data=value, x='distance_ly',
                                    y='star_density', ax=ax,
                                    marker='.', color=colors[index],
                                    label=labels[index])
            else:
                return None
            ax.set_xlabel('Distance (light years)')
            ax.set_ylabel('Stars per cubic light year')
            ax.set_title('Star density in function of distance')
            plt.show()
        print(data.head())
        data['norm_lim'] = data['distance_ly']\
            .apply(lambda x: 6.36241121e-01 + 1.68445770e-02 * x +
                   4.59928420e-05 * x**2)
        sns.scatterplot(data=data, x='distance_ly', y='avg_luminosity',
                        marker='.', color='black')
        sns.lineplot(data=data, x='distance_ly', y='norm_lim', color='red')

        def func(x, a_0, a_1, a_2):
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
            return a_0 + a_1 * x + a_2 * x**2

        fittedParameters, pcov = curve_fit(func, data['distance_ly'],
                                           data['avg_luminosity'])
        print(fittedParameters)
        plt.show()

        data['norm_temp'] = data['distance_ly']\
            .apply(lambda x: -5731.17721777 +
                   2199.97159567 * np.log(x + 34.32346184))
        sns.scatterplot(data=data, x='distance_ly', y='avg_temperature',
                        marker='.', color='black')
        sns.lineplot(data=data, x='distance_ly', y='norm_temp', color='red')
        plt.show()
        fittedParameters, pcov = curve_fit(lambda x, a, b, c: a +
                                           b * np.log(x - c),
                                           data['distance_ly'],
                                           data['avg_temperature'])
        print(fittedParameters)
        data['norm_mass'] = data['distance_ly']\
            .apply(lambda x: -2.86034425 + 0.72868525 * np.log(x + 67.299702))
        sns.scatterplot(data=data, x='distance_ly', y='avg_mass',
                        marker='.', color='black')
        sns.lineplot(data=data, x='distance_ly', y='norm_mass', color='red')
        plt.show()
        fittedParameters, pcov = curve_fit(lambda x, a, b, c:
                                           a + b * np.log(x - c),
                                           data['distance_ly'],
                                           data['avg_mass'])
        print(fittedParameters)
        data['norm_radius'] = data['distance_ly']\
            .apply(lambda x: -1.07370745 + 0.41369796 * np.log(x + 27.357384))
        sns.scatterplot(data=data, x='distance_ly', y='avg_radius',
                        marker='.', color='black')
        sns.lineplot(data=data, x='distance_ly', y='norm_radius', color='red')
        plt.show()
        data['norm_den'] = data['distance_ly']\
            .apply(lambda x: 0.00021852 + 0.01226634 / x + 0.0676009 / x**2)
        data['norm_den_e'] = data['distance_ly']\
            .apply(lambda x:  7.10467129e-05 + 2.38910759e-02 / x +
                   4.86977984e-01 / x**2 + -7.77456715e+00 / x**3 +
                   2.47987008e+01 / x**4)
        data['norm_den_l'] = data['distance_ly']\
            .apply(lambda x: - 2.01782377e-04 + 9.55847748e-02 / x
                   - 3.97273323e+00 / x**2 + 9.07468662e+01 / x**3
                   - 9.09767460e+02 / x**4 + 3.91460064e+03 / x**5
                   - 5.94190054e+03 / x**6)
        fittedParameters, pcov = curve_fit(lambda x, a, b, c, d, e,
                                           f, g: a + b / x + c / x**2 +
                                           d / x**3 + e / x**4 + f / x**5 +
                                           g / x**6, data['distance_ly'],
                                           data['star_density'])
        print(fittedParameters)
        sns.scatterplot(data=data, x='distance_ly', y='star_density',
                        marker='.', color='black')
        sns.lineplot(data=data, x='distance_ly', y='norm_den', color='blue')
        sns.lineplot(data=data, x='distance_ly', y='norm_den_l', color='red')
        plt.show()
    avg_variations(df, kind='line')


def maps_demonstration():
    """Demonstration of the use of the package code to build stelar maps."""
    print("Enter the minimum and maximum visual\
 magnitude to select the objects to plot. The limits are numbers\
 so the lower number will represent an inferior limit, although\
 it represents higher magnitudes. The default is -16 for Min\
 and 2 for Max.")
    Min = input('Min:\t')
    Max = input('Max:\t')
    if len(Min) == 0:
        Min = -16
    else:
        Min = float(Min)
    if len(Max) == 0:
        Max = 2
    else:
        Max = float(Max)
    print('Ploting sky stelar map as seen from the telescope.')
    visual.local_sky(init=(-np.pi, -2*np.pi), end=(np.pi, 2*np.pi),
                     mag_limit=(Min, Max), names=True)
    print('Enter with the number of nearest stars you want to\
 include in the HR-diagram, or type [Enter] to use\
 the default of 10000')
    size = input('How many stars?\t')
    if len(size) == 0:
        size = 1000
    else:
        size = int(size)
    df = data_analysis.create_dataframe(size=size, sort='asc',
                                        by='distance_ly')
    print(f'Ploting a HR diagram of the {size} bodies closer to the\
 telescope.')
    visual.HR_diagram(df['B-V'], df['abs_magnitude'], df['common name'],
                      curves=False, points=False, only_named=False)
    print("How much of the nearest stars to plot in a 3D map? You can\
 navigate it using the mouse (for example zoom in and out with the\
 right button). Press [Enter] to use the default of 100 stars")
    size = input('How many stars?\t')
    if len(size) == 0:
        size = 100
    else:
        size = int(size)
    df = df.head(size)
    print(f'Ploting a 3D map of the {size} closest bodies to the telescope.')
    visual.plot_map_df(df)


print('\n\nThis is just a small guide on the hipparcus_space_exploration\
project containing some examples. See the documentation for more.\n\n')
flag = input('type:\n\t 0 for and OOP designed example where we will\
 explore the object star created from the paralax, visual\
 magnitude and B-V magnitude data taken from the hipparcos\
 mission repository.\
\n\n\t 1 for a data analysis directed exploration where the same\
 properties are used to build a pandas dataframe instead of\
 independent objects. In this framework we can do an extensive\
 data directed aproach.\
\n\n\t 2 for a short demonstration of using the built in\
 maps utilities.\
\n\n\t if you type anything else, the job will be killed.\n\n')
if flag == '0':
    print('Initializing the Object Oriented Programing demonstration mode.')
    OOP_demonstration()
elif flag == '1':
    print('Initializing the Data Analysis demonstration mode.')
    data_analisis_demonstration()
elif flag == '2':
    print('Initializing the stelar maps demonstration mode.')
    maps_demonstration()
else:
    print('Exiting!')
