# Multi-GASTON

Multi-GASTON is a unified and interpretable deep learning framework that learns multiple spatial axes of variation shared across multiple spatially resolved samples of the same tissue, such as spatial transcriptomics (ST) and spatial metabolomics. Multi-GASTON models the latent geometry of a tissue using multiple _tissue-intrinsic coordinates_, where different subsets of genes vary along each tissue-intrinsic coordinate. Specifically, Multi-GASTON models the expression of each gene as a piecewise continuous function of a small number of tissue-intrinsic coordinates, where the pieces correspond to spatial domains and the continuous functions describe continuous variation in expression within each domain along each spatial axis. We note that Multi-GASTON only requires that the different ST samples (or other samples of the same modality) have the same latent geometry, defined by the tissue-intrinsic coordinate system, and does not require the different ST samples to have similar physical geometries (e.g. shape, size, orientation).

<p align="center">
<img src="plots/workflow.png" height=500/>
</p>

## Installation
First install the conda environment from the environment.yml file:
```
cd multi_gaston
conda env create -f environment.yml
```
Then, activate the enviroment and install multi_gaston
```
conda activate multi_gaston_env
pip install -e .
```
The installation should take less than 10 minutes.

## Tutorial
An example application of Multi-GASTON to VisiumHD data of mouse small intestine can be found in `demo/`.
Other example applications of Multi-GASTON (restricted to a single sample) to spatial metabolomics datasets of murine liver and small intestine can also be found in https://github.com/raphael-group/MET-MAP, where the method is re-named as Metabolic Topography Mapper, MET-MAP (Samarah, L.Z., Zheng, C., Xing, X. et al. Spatial metabolic gradients in the liver and small intestine. Nature 648, 182–190 (2025). https://doi.org/10.1038/s41586-025-09616-5).

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
