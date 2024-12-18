# Multi-GASTON

Multi-GASTON is an unsupervised deep learning model that learns multiple spatial gradients simoutaineously from spatially resolved data such as spatial metabolomics and spatially-resolved transcriptomics (SRT). It is an extension of GASTON (https://pmc.ncbi.nlm.nih.gov/articles/PMC10592770/), which was designed for SRT data and learns a single _topographic map_ of a 2-D tissue slice in terms of a 1-D coordinate called _isodepth_, where all genes can be expressed as a function of this isodepth. Now allowing metabolites or genes to follow distinct spatial patterns, Multi-GASTON captures the metabolite abundance or gene expression topography by learning _k distinct isodepths_, that smoothly vary across a tissue slice and capture spatial organizations of different groups of spatially variable metabolites or genes.

<p align="center">
<img src="https://github.com/raphael-group/Multi-GASTON/tree/main/plots/NNarchitecture.png?raw=true" height=400/>
</p>

## Installation
TO BE CHANGED

## Software dependencies
* torch (=2.0.0)
* matplotlib (=3.8.0)
* pandas (=2.1.1)
* scikit-learn (=1.3.1)
* numpy (=1.23.4)
* jupyterlab (=4.0.6)
* seaborn (=0.12.2)
* tqdm (=4.66.1)
* scipy (=1.11.2)
* scanpy (=1.9.5)
* squidpy (=1.3.1)
