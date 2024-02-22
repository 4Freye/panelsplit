from setuptools import setup, find_packages

setup(
    name='panelsplit',  # Replace with the desired name of your package
    version='0.1.0',
    packages=find_packages(include=['panelsplit', 'panelsplit.*']),  # Include only packages within the "panelsplit" folder

    # Metadata
    author='Eric Frey',
    author_email='eric.frey@bse.eu',
    description='A custom toolkit for working with panel data.',
    url='https://github.com/4Freye/panelsplit',  # URL to your GitHub repository
    license='MIT',

    # Dependencies
    install_requires=[
        'scikit-learn',  # for TimeSeriesSplit and clone
        'tqdm',  # for progress bar
        'pandas',  # for DataFrame and Series
        'matplotlib',  # for plotting
        'joblib',  # for parallel computing
        'numpy'  # for numerical operations
    ],


    # Other configurations
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
