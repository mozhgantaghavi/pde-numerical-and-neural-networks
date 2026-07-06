# Partial Differential Equations (PDEs) Solver: From Isoparametric FEM to Physics-Informed Neural Networks (PINNs)

Hi there! 👋 Welcome to my repository. 

In this project, I wanted to explore how we can solve complex Partial Differential Equations (PDEs) using two completely different worlds: **classic numerical mathematics** and **modern artificial intelligence**. 

Instead of just sticking to one method, I built a bridge from classical numerical methods to cutting-edge deep learning. Inside this repo, you will find:
* **The Traditional Way:** An Isoparametric Finite Element Method (Iso-FEM) solver that handles curved geometries using standard mesh grids and mathematical integration.
* **The AI Way:** Physics-Informed Neural Networks (PINNs) that use PyTorch and automatic differentiation to solve equations by teaching a neural network the laws of physics—completely mesh-free!

---

## Detailed Model Overview

Here is a simple, human-friendly breakdown of the three main solvers implemented in this project:

### 1. Isoparametric FEM (Laplace Equation)
* **The Problem:** Solving the steady-state Laplace equation inside a curved, donut-shaped circular ring (stretching from an inner radius of 1.0 to an outer radius of 2.0).
* **How it works:** Regular square grids don't fit well on circular boundaries. To fix this, I used an isoparametric approach. It maps curved physical shapes into simple triangles using explicit Jacobian matrices, making the integration super clean.
* **The Results:** The solver is incredibly accurate! When compared directly against the real analytical solution, it achieved excellent error metrics:
    * **RMS Error:** $8.208623 \times 10^{-4}$
    * **$L_2$ Error:** $5.687101 \times 10^{-2}$
    * **$L_{\infty}$ Error:** $1.022891 \times 10^{-3}$

### 2. 1D Burgers' Equation (PINN)
* **The Problem:** This equation is a classic benchmark for fluid dynamics and shock wave propagation. It is notoriously tricky because it develops very sharp gradients (shocks) over time.
* **How it works:** No grids, no triangles, and no mesh! I built a deep neural network in PyTorch. Instead of learning from a dataset, the network uses automatic differentiation (`torch.autograd`) to calculate its own derivatives and forces itself to satisfy the underlying physics equation.

### 3. 2D Heat Equation (PINN)
* **The Problem:** Simulating how temperature dynamically changes and spreads across a 2D surface over time.
* **How it works:** This is another PINN setup, but with a smart twist in the training logic. During long training sessions, neural networks can sometimes diverge or lose their best state. To prevent this, I wrote a custom, conditional saving loop. The code continuously monitors the loss and **only** updates and saves the model checkpoint (`heat_2d_pinn.pt`) when it hits a new historical minimum. This ensures you always keep the absolute best version of the trained network.

---

## Project Structure
To keep the code clean and modular, everything is organized into dedicated folders with their own YAML configuration files:
* `/iso_parametric_fem`: Code for the polar mesh generator, element matrices, sparse solver, and Laplace execution.
* `/burger_1D`: The PyTorch architecture and physics loss functions for the 1D Burgers' equation.
* `/Heat_2D_PINNs`: The transient heat model script, along with the smart model checkpoint saving system.
