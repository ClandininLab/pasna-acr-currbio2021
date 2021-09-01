from setuptools import setup, find_packages

setup(
    name='pasna2021',
    version='1.0',
    description='Analysis code for Carreira-Rosario et al. Current Biology 2021.',
    long_description='Original analysis code for Carreira-Rosario et al. "Mechanosensory input during circuit formation shapes Drosophila motor behavior through Patterned Spontaneous Network Activity", Current Biology 2021.',
    url='https://github.com/ClandininLab/pasna-acr-currbio2021',
    author='Minseung Choi',
    author_email='minseung@stanford.edu',
    packages=['pasna2021'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'scikit-learn',
        'statsmodels',
        'openpyxl'
    ],
    include_package_data=True
)