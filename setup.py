from setuptools import setup, find_packages

setup(
    name="onionnet",
    version="1.0.0",
    description="A package for creating and analysing large mulilayered networks using graph-tool.",
    author="Macabe Daley",
    author_email="***",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        #"graph-tool",  # Note graph-tool is not pip installable, so you should install it via conda before running OnionNet.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)