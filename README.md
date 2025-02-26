
[![DOI](https://zenodo.org/badge/398899394.svg)](https://doi.org/10.5281/zenodo.14933301)


# GridSeis


![Diagram](/plots/scatter_xgb.png)

![Diagram](/plots/validation.png)

![Diagram](/plots/fft_heatmap_january.png)


![Diagram](/plots/fft_heatmap_december.png)


# How to run this code

I tend to have a bare essentials miniconda base env. I then make a conda env:

`conda create -n grid_seis python=3.12`

and use pip to manage the packages

`pip install -r requirements.txt`

to create plots, run `1_investigation.py`
to train the model run `2_modelling.py`