from setuptools import setup, find_packages

setup(
    name="pystack3d",
    version='2023.1',
    license='GPL v3',
    include_package_data=False,
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "skimage",
        "statsmodels",
        "pystackreg",
        "PIL",
        "tiffile",
        "tomli",
        "tomlkit",
        "parse",
        "multiprocessing"
    ],
    packages=find_packages(where='.', include=['pystack3d*']),

    description="PyStack3D (A python tool to process stacks of images)",

    url="https://github.com/CEA-MetroCarac/pystack3d",
    author_email="patrick.quemere@cea.fr",
    author="Patrick Quéméré",
    keywords="PyStack3D, stack, images, FIB-SEM, processing, correction, cropping, background removal, registration, intensity rescaling, "
             "destripping, curtains, resampling, multithreading, multiprocessing",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Environment :: Console',
    ]
)