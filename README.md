# Randomized Sparse Neural Galerkin Schemes (RSNG)
Sample code for Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks

## Setup

First locally install the rsng package with

```bash
pip install --editable .
```

Then install jax with the appropriate CPU or GPU support: [here](https://github.com/google/jax#installation)

Install all additionaly required packages run:

```bash
 pip install -r requirements.txt
```

Then you should be able to run the included notebooks:

- allen_cahn.ipynb
- burgers.ipynb