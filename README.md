
[![DOI](https://zenodo.org/badge/398899394.svg)](https://doi.org/10.5281/zenodo.14933301)


# GridSeis

## Summary

![Diagram](/plots/scatter_xgb.png)

![Diagram](/plots/validation.png)

## Background: Fourier Transforms

![Fourier Transform visualisation](plots/fourier.png)

A Fourier transform is a mathematical technique that decomposes a signal into its constituent frequencies. It converts a signal from its original domain (often time) into a representation in the frequency domain.

In the image, you can see how a complex signal (likely in the time domain) is broken down into its frequency components, revealing the dominant frequencies that make up the original waveform.

## Methodology: GB Grid Frequency


![Diagram](/plots/fft_heatmap_january.png)


![Diagram](/plots/fft_heatmap_december.png)




<!-- ![Diagram](/plots/fourier.png) -->




# How to run this code

I tend to have a bare essentials miniconda base env. I then make a conda env:

`conda create -n grid_seis python=3.12`

and use pip to manage the packages

`pip install -r requirements.txt`

to create plots, run `1_investigation.py`
to train the model run `2_modelling.py`