# Sparse Principal Component analysis

- [Introduction](#introduction)
- [Documentation](#documentation)
	- [Usage](#usage)
	- [Arguments](#arguments)
	- [Values](#values)
- [Quick Start](#quick-start)
	- [Python Quick Start](#python-quick-start)
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
