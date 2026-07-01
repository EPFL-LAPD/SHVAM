<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="">Single-View Holographic Volumetric 3D Printing with Coupled Differentiable Wave-Optical and Photochemical Optimization</a></h1>

  <a href="https://felixwechsler.science/pages/publications/SHVAM/">
    <img src="https://raw.githubusercontent.com/EPFL-LAPD/SHVAM/refs/heads/main/banner_github.jpg" alt="Logo" width="100%">
  </a>

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH), July 2026.
    <br />
    <a href="https://www.felixwechsler.science/"><strong>Felix Wechsler</strong></a>
    ·
    <a href="https://www.linkedin.com/in/riccardo-rizzo-phd"><strong>Riccardo Rizzo</strong></a>
    ·
    <a href="https://www.epfl.ch/labs/lapd/page-67957-en-html/"><strong>Christophe Moser</strong></a>
    ·
  </p>

  <p align="center">
    <a href='https://arxiv.org/pdf/2601.16330'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper (arXiv) PDF'>
    </a>
  </p>
</p>


<br />
<br />


## Abstract
Volumetric additive manufacturing promises near-instantaneous fabrication of 3D objects, yet achieving high fidelity at the micro-scale remains challenging due to the complex interplay between optical diffraction and chemical effects. 
We present **Single-View Holographic Volumetric Additive Manufacturing** (SHVAM), a mechanically static system that shapes volumetric dose distributions using time-multiplexed, phase-only holograms projected from a single optical axis. 
To achieve high resolution with SHVAM, we formulate hologram synthesis as a coupled inverse problem, integrating a differentiable wave-optical forward model with a simplified photochemical model that explicitly captures inhibitor diffusion and non-linear dose response. 
Optimizing hologram sequences under these coupled constraints allows us to pre-compensate for chemical blur, yielding higher print fidelity than optical-only optimization. 
We demonstrate the efficacy of SHVAM by fabricating simple 2D and 3D structures with lateral feature sizes of approximately 10µm within a 0.8mmx0.8mmx3mm volume in seconds.


## Installation
This code is based on Python and JAX. So please install dependencies with:
```python
pip install numpy scipy matplotlib jax jaxopt tqdm jax[cuda12_pip] jaxlib notebook
```

## Minimal Example
Open the Jupyter notebook to see an exemplary optimization of one of the conical structures in the picture.
