# Opt_mean_field_spin_glass

This repository contains the source code for the course project of **CS581B7**.  
The project focuses on optimization algorithms for mean-field spin glass models, especially the Sherrington--Kirkpatrick (SK) model and its spherical relaxation.

## Project Description

Mean-field spin glass optimization is a high-dimensional, nonlinear, and nonconvex optimization problem. In this project, we study several methods for solving or approximating spin glass models, including:

- Spectral relaxation
- Semidefinite programming (SDP) relaxation
- Hessian ascent for spherical spin glass models
- Approximate Message Passing (AMP)

The goal of this code is to reproduce numerical experiments, compare different relaxation and optimization methods, and provide computational support for the final report of **CS581B7** at **Colorado State University (CSU)** in **Spring 2026**.

## Files

- `AMP.py`: Implementation and testing of the Approximate Message Passing algorithm.
- `HessianAscent.py`: Implementation of the Hessian ascent algorithm for spherical spin glass models.
- `Plot_relax.py`: Plotting scripts for comparing relaxation methods.
- `test_pro.py`: Testing script for numerical experiments.
- `Results/`: Folder containing generated numerical results and figures.

## Requirements

The code is written in Python. Commonly used packages include:

```bash
numpy
scipy
matplotlib
gurobipy
cvxpy

