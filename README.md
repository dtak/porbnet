# PoRB-Nets: Poisson Process Radial Basis Function Networks

We introduce Poisson Process Radial Basis Function Networks (PoRB-Nets), an interpretable family of radial basis function networks (RBFNs) that employ a Poisson process prior over the center parameters in an RBFN. The proposed formulation enables direct specification of functional amplitude variance and lengthscale as in a GP with an RBF kernel, the latter of which can vary over the input space.

Pytorch implementations of the homogeneous and inhomogeneous Poisson process cases are available in `src/porbnet` and `src/porbnet-sgcp`, respectively. For a simple example, see `notebooks/figure1.ipynb`. 