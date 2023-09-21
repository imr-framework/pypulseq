import setuptools

from version import major, minor, revision


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
        long_description = "Pulseq in Python"
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
        "coverage>=6.2",
        "matplotlib>=3.5.2",
        "numpy>=1.19.5",
        "scipy>=1.8.1",
        "sigpy@git+https://github.com/mikgroup/sigpy",  # change before release to PyPI
    ],
    license="License :: OSI Approved :: GNU Affero General Public License v3",
    long_description=_get_long_description(),
    long_description_content_type="text/markdown",
    name="pypulseq",
    packages=setuptools.find_packages(),
    py_modules=["version"],
    # package_data for wheel distributions; MANIFEST.in for source distributions
    package_data={"pypulseq.SAR": ["QGlobal.mat"]},
    project_urls={"Documentation": "https://pypulseq.readthedocs.io/en/latest/"},
    python_requires=">=3.6.3",
    url="https://github.com/imr-framework/pypulseq",
    version=".".join((str(major), str(minor), str(revision))),
)
