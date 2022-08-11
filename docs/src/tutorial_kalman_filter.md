# Implementing a Kalman filter

This tutorial describes how to implement a Kalman filter for the following state-space model

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(\mu_0 ,\Sigma_0), \\
x_n \mid x_{n-1} &\sim \mathcal{N}(\Phi  x_{n-1}, Q),\\
z_n \mid x_n &\sim \mathcal{N}(Cx_n,R),
\end{aligned}
```
given a measurement sequence $z_{0:N}$.



### Kalman filter implementation

