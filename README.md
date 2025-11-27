# Self-MSNet
This is the source code for Self-Supervised Multi-Scale Uniform Motion Deblurring via Alternating Optimization.

# Abstract:
Blind image deblurring is a challenging low-level vision task that involves estimating the unblurred image when the blur kernel is unknown. We focus on the removal of uniform motion blur that remains spatially invariant throughout the image. In this paper, we present a self-supervised multi-scale blind image deblurring method to jointly estimate the latent image and the blur kernel via alternating optimization. In the image estimation step, we construct a multi-scale generator network with multiple inputs and multiple outputs to collaboratively estimate latent images at various scales, supervised by an image pyramid constructed from the only blurred image. This generator places architectural constraints on the network and avoids the need for mathematical expression of image priors. In the blur kernel estimation step, the blur kernel at each scale is independently estimated with a direct solution to a quadratic regularized least-squares model for its flexible adaptation to the proposed multi-scale generator for image estimation. Thanks to the collaborative estimation across multiple scales, our method avoids the computationally intensive coarse-to-fine propagation and additional image deblurring processes used in traditional mathematical optimization-based methods. Quantitative and qualitative experimental results on simulated and realistic datasets demonstrate the superior performance of our method, especially for handling large and real-world blurs.

# Dependencies:
Pytorch: 1.8.0
Cuda: 10.1
Python: 3.7
