from setuptools import setup, find_packages
# from pip.req import parse_requirements
import pathlib

import pkg_resources
# parse the requirements given in the requirements.txt file
# install_reqs = parse_requirements('./requirements.txt', session=False)
# reqs = [str(ir.req) for ir in install_reqs]

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

# parse the dependency links, i.e. other projects on gitlab for example
# with open('./dependency-links.txt') as dep_file:
#     dep_links = dep_file.readlines()

setup(
    name="pysortgui",
    version="1.0.0",
    packages=find_packages(),
    scripts=['bin/pysortgui',
             ],  # 'bin/spikesortercli'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine

    install_requires=install_requires,
    # dependency_links=dep_links,
    # dependency_links=[
    #    'git+https://gitlab.com/alessandro.scaglione/pyephys.git@master#egg=pyephys-0.0'],


    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'pysortgui': ['External/bins/*.*',
                      #   'plugins/*.*',
                      #   'datafun/*.*',
                      #   'icons/*.*'
                      ],
        # And include any *.msg files found in the 'hello' package, too:
        # 'hello': ['*.msg'],
    },

    # metadata for upload to PyPI
    author="Harry Wang",
    author_email="harry.wang0210 at gmail.com",
    description="This package provides a spike sorter for neural data",
    # license="PSF",
    keywords="ephys neurons spike sorting",
    # url="http://example.com/HelloWorld/",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
