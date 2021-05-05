from typing import Tuple

import setuptools


def _get_version() -> Tuple[int, int, int]:
    """
    Returns version of current PyPulseq release.

    Returns
    -------
    major, minor, revision : int
        Major, minor and revision numbers of current PyPulseq release.
    """
    with open('VERSION', 'r') as version_file:
        major, minor, revision = version_file.read().strip().split('.')
    return major, minor, revision


def _get_long_description() -> str:
    """
    Returns long description from `README.md` if possible, else 'Pulseq in Python'.

    Returns
    -------
    str
        Long description of PyPulseq project.
    """
    try:  # Unicode decode error on Windows
        with open("README.md", "r") as fh:
            long_description = fh.read()
    except:
        long_description = 'Pulseq in Python'
    return long_description


setuptools.setup(
    author="Keerthi Sravan Ravi",
    author_email="ks3621@columbia.edu",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    description="Pulseq in Python",
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.3.4',
        'numpy>=1.19.5',
        'scipy>=1.5.4'
    ],
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    name="pypulseq",
    packages=setuptools.find_packages(),
    # package_data for wheel distributions; MANIFEST.in for source distributions
    package_data={
        '': ['../VERSION'],
        'pypulseq.SAR': ['QGlobal.mat']
    },
    project_urls={
        'Documentation': 'https://pypulseq.readthedocs.io/en/latest/'
    },
    python_requires='>=3.6.3',
    url="https://github.com/imr-framework/pypulseq",
    version=".".join(_get_version()),
)
