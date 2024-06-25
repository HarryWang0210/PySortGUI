import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name="pysortgui",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pysortgui=pysortgui.main:launch_app',
        ],
    },
    install_requires=install_requires,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'pysortgui': ['External/bins/*',
                      'UI/style.qss',
                      #   'plugins/*.*',
                      #   'datafun/*.*',
                      #   'icons/*.*'
                      ],
    },

    # metadata for upload to PyPI
    author="Harry Wang",
    author_email="harry.wang0210@gmail.com",
    description="This package provides a spike sorter for neural data",
    # license="PSF",
    keywords="ephys neurons spike sorting",
    # url="http://example.com/HelloWorld/",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
