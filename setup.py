from setuptools import setup, find_packages

setup(
    name='GWPhotonCounting',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy>=1.18.0',
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here, e.g.,
            # 'gwphotoncounting=GWPhotonCounting.cli:main',
        ],
    },
    author='Ethan Payne',
    author_email='ethan.payne@ligo.org',
    description='Module to assist in understanding a photon counting readout scheme for future gravitational-wave detectors',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ethanpayne42/GWPhotonCounting',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)