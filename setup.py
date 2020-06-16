import setuptools

try:  # Unicode decode error on Windows
    with open("README.md", "r") as fh:
        long_description = fh.read()
except:
    long_description = 'Pulseq in Python'

setuptools.setup(
    name="pypulseq",
    version="1.2.0.post3",
    author="Keerthi Sravan Ravi",
    author_email="ks3621@columbia.edu",
    description="Pulseq in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imr-framework/pypulseq",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'pypulseq.utils.SAR': ['QGlobal.mat']},
    install_requires=['matplotlib>=3.0.3',
                      'numpy>=1.16.3',
                      'scipy>=1.4.1'],
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
