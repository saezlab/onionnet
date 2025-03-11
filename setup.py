from setuptools import setup, find_packages

setup(
    name="onionnet",
    version="0.1.0",
    description="A package for creating and exploring large layered networks (ie. onions) using graph-tool.",
    author="Macabe Daley",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        #"graph-tool",  # Ensure users know this is a dependency; note that graph-tool may need special installation instructions.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)