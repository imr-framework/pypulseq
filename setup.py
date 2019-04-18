import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypulseq",
    version="0.0.3",
    author="Keerthi Sravan Ravi",
    author_email="sravan953@gmail.com",
    description="Pulseq in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imr-framework/pypulseq",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
