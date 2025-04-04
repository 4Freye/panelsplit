from setuptools import setup, find_packages

# Read the contents of the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='panelsplit',
    version='1.0.2',
    packages=find_packages(include=['panelsplit', 'panelsplit.*']),  # Include only packages within the "panelsplit" folder

    # Metadata
    author='panelsplit developers',
    author_email='eric.frey@bse.eu',
    description='A tool for panel data analysis.',
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

    # README file content
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Other configurations
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)