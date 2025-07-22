# Introduction

> OnionNet = A graph-tool wrapper for handling large multi-layered networks

There are a number of different python packages for dealing with networks, each with their own pros and cons. `NetworkX` is probably the widest used in the python ecosystem, with a gentle learning curve and a vast array of functionality. But being a purely python implementation, it scales very poorly to large networks. For example in this [blogpost](https://www.timlrx.com/blog/benchmark-of-popular-graph-network-packages-v2), although somewhat aged now, benchmarking showed `NetworkX` performed nearly 100x slower than `graph-tool` on pagerank. While more efficient packages implemented in C++ with python wrappers are available, such as `graph-tool`, they may offer more targeted functionality or have a steeper learning curve to operate. 

OnionNet is a wrapper for `graph-tool` to make it easier to build, manipulate, analyse and visualise very large multi-layered networks. This was spurred from my own needs faced in the creation and analysis of LipiNet, but the package is abstract enough to be potentially useful for anyone working large multilayered networks of any kind.