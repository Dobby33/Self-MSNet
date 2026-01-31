# Self-MSNet
This is the source code for "Self-Supervised Multi-Scale Uniform Motion Deblurring via Alternating Optimization".


# Overview:
<img width="1331" height="800" alt="image" src="https://github.com/user-attachments/assets/686d3da2-2bc3-4ed7-ad27-9469e89a5c92" />


# Abstract:
Blind image deblurring is a challenging low-level vision task that involves estimating the unblurred image when the blur kernel is unknown. We focus on the removal of uniform motion blur that remains spatially invariant throughout the image. In this paper, we present a self-supervised multi-scale blind image deblurring method to jointly estimate the latent image and the blur kernel via alternating optimization. In the image estimation step, we construct a multi-scale generator network with multiple inputs and multiple outputs to collaboratively estimate latent images at various scales, supervised by an image pyramid constructed from the only blurred image. This generator places architectural constraints on the network and avoids the need for mathematical expression of image priors. In the blur kernel estimation step, the blur kernel at each scale is independently estimated with a direct solution to a quadratic regularized least-squares model for its flexible adaptation to the proposed multi-scale generator for image estimation. Thanks to the collaborative estimation across multiple scales, our method avoids the computationally intensive coarse-to-fine propagation and additional image deblurring processes used in traditional mathematical optimization-based methods. Quantitative and qualitative experimental results on simulated and realistic datasets demonstrate the superior performance of our method, especially for handling large and real-world blurs.


# Dependencies:
Pytorch: 1.8.0

Cuda: 10.1

Python: 3.7


# Preparation:
conda create -n selfmsnet python=3.7

conda activate selfmsnet


# Datasets:
Lai et al’s dataset[1] can be downloaded from: http://vllab.ucmerced.edu/˜wlai24/cvpr16_deblur_study.

Kohler et al’s dataset[2] can be downloaded from:  http://webdav.is.mpg.de/pixel/benchmark4camerashake.


# Run:
Move the input blurred images to the corresponding folder named ‘datasets/lai’ or ‘datasets/kohler’.

Run the following commands for motion deblurring. Use **--model_name** to select either **Self_MSNet** (our full model *Self-MSNet*) or **Self_MSNet_S** (the single-scale variant *Self-MSNet-S*). Deblurred results will be saved to the ‘results/’ directory.

python main.py --data_set lai --model_name Self_MSNet

python main.py --data_set kohler --model_name Self_MSNet


# If you use this code in your research, please cite:
@article{GUO2025112774,

title = {Self-Supervised Multi-Scale Uniform Motion Deblurring via Alternating Optimization},

author = {Lening Guo and Jing Yu and Ning Zhang and Chuangbai Xiao},

journal = {Pattern Recognition},

volume = {173},

pages = {112774},

year = {2026},

issn = {0031-3203},

doi = {https://doi.org/10.1016/j.patcog.2025.112774},

url = {https://www.sciencedirect.com/science/article/pii/S0031320325014372}}


# References:
[1] LAI W S, HUANG J B, HU Z, et al. A comparative study for single image blind deblurring[C]//IEEE Conf. Comput. Vis. Pattern Recog. IEEE, 2016: 1701-1709.

[2] KÖHLER R, HIRSCH M, MOHLER B, et al. Recording and playback of camera shake: Benchmarking blind deconvolution with a real-world database[C]//Eur. Conf. Comput. Vis. Berlin, Heidelberg: Springer, 2012: 27-40.

# Acknowledgements:
This work builds upon the foundational contributions of:

- REN D, ZHANG K, WANG Q, et al. Neural blind deconvolution using deep priors[C]//IEEE Conf. Comput. Vis. Pattern Recog. 2020: 3341-3350.

- BAI Y, YU J, LI Y, et al. Deep prior-based blind image deblurring[J]. Acta Electronica Sinica, 2023, 51(4): 1050-1067.

- CHO S J, JI S W, HONG J P, et al. Rethinking coarse-to-fine approach in single image deblurring[C]//Int. Conf. Comput. Vis. 2021: 4641-4650.

