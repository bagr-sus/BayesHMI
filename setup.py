import setuptools
from setuptools import find_packages
__version__="0.1.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayeshmi",
    version=__version__,
    license='GPL 3.0',
    description='BayesHMI - Bayesian inversion for specific geophysical problems.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ondrej Simunek, Pavel Exner',
    author_email='ondrej.simnuek@tul.cz',
    url='https://github.com/bagr-sus/bayes-scripts',
    # download_url='https://github.com/flow123d/swrap/archive/v{__version__}.tar.gz',
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering',
    ],

    keywords=[
        'bayesian inversion', 'delayed acceptance',
    ],
    # include_package_data=True, # package includes all files of the package directory
    zip_safe=False,
    install_requires=['pandas', 'scipy', 'matplotlib'],
    python_requires='>=3.10',

    # according to setuptols documentation
    # the including 'endorse.flow123d_inputs' should not be neccessary,
    # however packege_data 'endorse.flow123d_inputs' doesn't work without it
    packages=['bp_simunek', 'bp_simunek.common', 'bp_simunek.plotting', 'bp_simunek.samplers', 'bp_simunek.samplers.surrogates'],
    package_dir={
        "": "src"
    },
    # package_data={
    #     "endorse" : ["*.txt"],
    #     "endorse.flow123d_inputs": ['*.yaml']
    # },
    # entry_points={
    #     'console_scripts': ['endorse_gui=endorse.gui.app:main', 'endorse_mlmc=endorse.scripts.endorse_mlmc:main']
    # }
)



        

        
