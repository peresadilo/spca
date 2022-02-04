# Sparse Principal Component analysis

- [Introduction](#introduction)
- [Documentation](#documentation)
	- [Usage](#usage)
	- [Arguments](#arguments)
- [References](#references)

## Introduction

Principal Component Analysis (PCA) is a frequently used method in the field of data science for data processing and dimensionality reduction, because of its propensity to simplify complexities in high-dimensional data without disregarding relations and patterns. `spca` is a python implementation of the sparse principal component algorithm, based on the work of Zou, Hastie, and Tibshirani (2006) who propose a method to improve interpretability by introducing sparseness contraints to PCA by implementing elastic net, a convex combination of the ridge and lasso penalties.

The authors of this package are Filipp Peresadilo, Dorus van Schai, Yuting Su and Nouri Mabrouk.

## Documentation

### Dependencies

```python

import pandas as pd
import numpy as np

```

### Usage

```python
import spca
import utils

spca.sparcepca(X=array, lambda2=list, lambda1=int, k=num, max_iteration=float, threshold=float, type=string)
```

### Arguments

| Arguments     | type           | Description  |
| ------------- |-------------| -----|
| `X`     | numpy.ndarray | (standardised/normalised) matrix of input variables or covariance matrix, specify type in 'type'|
| `lambda2`      | list      |   L1-regulization |
| `lambda1` |   float    |   L2-regulization  |
| `k` |   int    |  number of principal components  |
| `max_iteration` |  int     | maximum number of iterations   |
| `threshold` |    float   |  optimization treshold  |
| `type` |    string   |  data type ("cov" for covariance or "data" for input variables)  |

## References

Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component analysis. Journal of Computational and Graphical Statistics, 15(2), 265-286 <[doi:10.1198/106186006X113430](https://doi.org/10.1198%2F106186006X113430)>.

