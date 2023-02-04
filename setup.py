from setuptools import setup, find_packages
setup(
author="Ricardo S P Lopes",
description="A package to load, process and explore data from the \
             hipparcos mission vailable database",
name="hipparcos_space_exploration",
version="0.1.0",
packages=find_packages(include=["hipparcos_space_exploration",
                                "hipparcos_space_exploration.*"]),
install_requires=['pandas>=1.5','numpy>=1.23','matplotlib>=3.6','seaborn>=0.12.1','scipy>=1.9.0'],
python_requires='>=3.8',
license='MIT',
setup_requires=['pytest-runner'],
tests_require=['pytest==4.4.1'],
test_suite='tests',
)