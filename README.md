# OnionNet <img src="./.assets/.onionnet_logo_v0c.png" alt="OnionNet Logo" width="120" align="right" />
---
> A graph-tool wrapper for handling large multi-layered networks

There are a number of different python packages for dealing with networks, each with their own pros and cons. `NetworkX` is probably the widest used in the python ecosystem, with a gentle learning curve and a vast array of functionality. But being a purely python implementation, it scales very poorly to large networks. For example in this [blogpost](https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages-v2), although somewhat aged now, benchmarking showed `NetworkX` performed nearly 100x slower than `graph-tool` on pagerank. While more efficient packages implemented in C++ with python wrappers are available, such as `graph-tool`, they may offer more targeted functionality or have a steeper learning curve to operate. 

OnionNet is a wrapper for `graph-tool` to make it easier to build, manipulate, analyse and visualise very large multi-layered networks. This was spurred from my own needs faced in the creation and analysis of LipiNet, but the package is abstract enough to be potentially useful for anyone working large multilayered networks of any kind.

## Installation
`OnionNet` first requires `graph-tool` to be installed. Because `graph-tool` is built around C++ for efficiency, unfortunately there is no straightforward pip installation. Nonetheless, there are a number of ways to install `graph-tool` besides pip, see [here](https://graph-tool.skewed.de/installation.html) for more details. The easiest way for most users is probably to create a new env via `conda`:

```
conda create --name gt -c conda-forge graph-tool ipython jupyter
conda activate gt
```
Then you can install `OnionNet` with:
```
git clone https://github.com/saezlab/onionnet.git
cd onionnet
pip install -e .
```
In the near future we intend to include OnionNet on PyPI.

## Quick Start

If you have pandas dataframes with nodes and edges, then getting started is quick and simple. See the `getting_started.ipynb` for a full walkthrough from scratch.

## Contributing 

`OnionNet` is in active development and subject to change. Some functions may be modified or deprecated in future releases. If you find `OnionNet` helpful or have ideas for improvement, we'd love to hear more!