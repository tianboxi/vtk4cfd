import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vtk4cfd", # Replace with your own username
    version="0.0.1",
    author="Tianbo Raye Xie",
    author_email="tianboxi@usc.edu",
    description="A library of high level examples using VTK python interface to post-process CFD results",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    #        'vtk>=8.0',
    #        'numpy>=1.16',
    #        'scipy>=1.2',

    ],
    python_requires='>=2.7',

)
