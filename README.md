# hipparcos_space_exploration
Is the first of a series of packages to produce a scientific analysis of the mission's observations data available on the internet. This package uses data from the Hipparcos mission (see the [ESA](https://www.cosmos.esa.int/web/hipparcos) website) and internet archives located on [this](https://www.universeguide.com/) domain, where I took stars' common names. 

I decided to do this on two different paradigms. One can use the OOP philosophy of building objects carrying star characteristics (both directly deduced from the data or approximated). It can use a data analysis approach, using the data in data frame format instead. In either approach, one can create a 3D stellar map, a projection map, or a phi *versus* theta chart showing the location of the observations. 

## data directory
This directory contains the data files the code uses and holds the produced figures.

## src directory
This directory holds the code. All functions and classes are contained in the *tools.py* file.

## test_hipparcos_space_exploration directory
Contains all the automated tests used to validate the code.

## demonstrations in the main.py file
This file contains three demonstrations of the package analysis capabilities shown in a verbose mode. Running this file, three options will be given to you: 

### option 0: Object Oriented Programming approach

Typing the 0 in the console after running the *main.py* file you are asked for an identifier consisting of the HIP code in the file. If you just type enter, the identifier will be sampled from the data. After this identifier is shown, the code will print the features of a **StelarObject**. 

|            feature       | nature | type | description | access atribute|
|:------------------|------------------|------------------|------------------|-----------------------:|
|Object common name               |          non-approximative |  str| The name the object is referred to in the community (name not associated with a specific catalog)| .name|
|Object spectral type         |          non-approximative |  str| The object spectral type as given by the data|.measured_spectral_type|
|Object angular position (radians) |          non-approximative |  (float, float)| The angular position in the axis (theta, phi) taken from the data|(.theta, .phi)|
|Object angular apparent velocity (Km/s) |          non-approximative |  (float, float)| The angular velocities in the axis (theta, phi) taken from the data|(.vel_theta, .vel_phi)|
|Object cartesian aparent velocity (Km/s) |          non-aproximative |  (float, float, float)| The angular velocities transformed to cartesian coordenates. Do not include components of the radial velocity and each component is accessed in a particular component, not a tuple.|(.vel_x, .vel_y, .vel_z)|
|Object distance (in lightyears) |          non-aproximative |  float| The distance in light-years to the object as calculated from the paralax available in the data. One can also access the distance in parsecs with the .distance attribute|.distance_ly|
|Object distance (in lightyears) |          non-aproximative |  float| The distance in light-years to the object as calculated from the paralax available in the data. One can also access the distance in parsecs with the .distance attribute|.distance_ly|
|Object cartesian position |          non-approximative |  (float, float, float)| The object position in cartesian coordinates|(.x, .y, .z)|
|Object visual Hipparcos magnitude | non-approximative |  float| The visual magnitude as presented in the Hipparcos catalog |.Vmag|
|Object visual B-V magnitude |          non-approximative |  float| The B-V magnitude as presented in the Hipparcos catalogue| .B-V|
|Object visual V-I magnitude | non-approximative |  float| The V-I magnitude as presented in the Hipparcos catalog |.VI|
|Object absolute magnitude|         non-aproximative |  float| The absolute magnitude calculated from the .Vmag and .distance atributes| .absolute_magnitude|
|Object measured distance|         non-approximative |  float| The measured distance in 1997| .measured_distance|
|spectral type|     approximative | vector of tuples (str, float) | The probabilities of an object belonging to a given group calculated by the inverse of the distance of the object to characteristic lines of each group in the HR diagram| .spectral_type_probabilities|
|selected spectral type|     approximative | str| The spectral type with highest probability as calculated in the **spectral type** feature| .spectral_type|
|estimated luminosity|     approximative | float| The luminosity estimated by the absolute magnitude calculated| .estimated_luminosity|
|estimated luminosity|     approximative | float| The luminosity estimated by the absolute magnitude (in solar unities)| .estimated_luminosity|
|estimated mass|  approximative | float| The mass estimated using different mass-luminosity scaling relations characteristic of each luminosity class (in solar unities)| .estimated_mass|
|estimated temperature|  approximative | float| The effective temperature (in Kelvin) estimated using radius estimative and luminosity estimative (you can access another estimative for main sequence stars with the attribute .estimated_ballesteros_temperature)| .estimated_effective_temperature |
|estimated radius|  approximative | float| The radius of the object estimated using different mass-radius scaling relations characteristic of main sequence stars at the moment (in solar unities)| .estimated_radius|
|estimated spectral subtype|  approximative | str| The spectral subtype of main sequence stars (in this version) using the classification of their effective temperature| .spectral_subtype|

### option 1: Data analysis approach
This option displays a demonstration of a data analysis approach. Merging the data into a data frame we can reproduce all the features displayed in OOP approximation faster and can use the data to get insights from associated statistics.

First, the data is loaded and merged into a data frame and its head is displayed on the console. Then a preliminary analysis of the average estimated properties of the stars closer than 500 light years are displayed in graphics. The averages are made by sampling concentric spherical shells summing those properties and dividing over the shell volume. The plots show a quadratic increase in the average luminosity, a logarithmic increase in the temperature, and a logarithmic increase in the average mass and radius. Finally, the density of stars presents a good fit of expression of the type *a + b/x + c/x^2* but this approximation breaks down at higher distances, so another non-linear fit of the type *a + b/x + c/x^2 + d/x^3 + e/x^4 + f/x^5 + g/x^6*  which work well at greater distances were also plotted in the figure. 

Just this simple visualization already shows there is some important feature to consider before manipulating the information at hand. First of all the fact that simple functions do not work at low distances is a consequence of the stellar distribution being approximately uniform in our close neighborhood. This is in accord with the principle of statistical homogeneity of stars' population in regions presenting similar gas distributions, age, and galactic position (the Orion Arm is approximately 10,000 light-years wide). The decline in star density that happens after some distance is related to the telescope's capabilities. As the minimum magnitude listed in the file is close to 15, we can assume that the telescope cannot measure anything dimmer. Therefore, any object with a visual magnitude higher than 15 will not appear in the measurements. For a typical Brown Dwarf, this limit is reached at 20 ly. Therefore, to be seen, the star must be brighter. This implies that their masses must increase as their radius, as seen. 


### option 2: Maps demonstration
This mode displays graphical constructions. First, you be asked to give two numbers, the minimum magnitude to consider and the maximum magnitude to consider. The defaults are -16 for Min and 2 for Max. Then, the code will generate a map of the stars with visual magnitudes in that range. This will generate a y *versus* x style plot displaying points in the theta, and phi axis representing the positions of the stars in the sky from the point of view of the telescope. The shown stars present the magnitude within the specified values. 

After that, You will be asked to give the number of stars to plot in the HR diagram. The stars are ordered in ascending distance from the telescope and tipping -1 returns the plot of the whole data frame.

Closing the window, you will be asked once again the number of nearest stars. This time, to plot a 3D map of the local neighborhood. The sizes are proportional to the calculated radius and the colors are linked to the star types according to the following table: 
|    star type       | associated color |
|:-------------------|:-----------------|
| white dwarf        | white            |
| dwarf nova         | deepskyblue      |
| brown dwarf        | peru             |
| sub-dwarf          | mediumspringgren |
| main sequence star | yellow           |
| Sub giant          | orange           |
| blue giant         | blue             |
| red giant          | coral            |
| bright giant       | pink             |
| supergiant Ib      | yellowgreen      |
| supergiant Ia      | lime             |
| hypergiant         | ghostwhite       | 

## instalation and further reading

Find out more about this work on [GitHub](https://datacamp.com). 
install it with 
```

pip install hipparcos_space_exploration
```
or 
```

pip install --index-url https://test.pypi.org/simple      --extra-index-url https://pypi.org/simple
hipparcos_space_exploration
```

To install from TestPyPI