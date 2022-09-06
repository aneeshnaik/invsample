# invsample

Sample from arbitrary univariate PDFs using inverse transform sampling.

The algorithm works as follows: given some (not necessarily normalised) PDF $f(x)$, numerical integration on regular array of $x$ points gives a corresponding array of points representing the CDF $F(x)$. The inverse CDF $F^{-1}$ is known as the quantile function. Given a sample $u$ from a uniform distibution between 0 and 1, applying the quantile function $F^{-1}(u)$ gives a random sample from the pdf $f$. Even in cases where the quantile function can not be calculated exactly, it can be applied by linearly interpolating within the $x$, $F(x)$ arrays calculated earlier.

## Usage

### Drawing a single sample from a single PDF

First, define a function to compute the PDF as a function of the variable `x`, along with any additional arguments representing function parameters. In this example, we'll consider the PDF
$$p(x|a, b, c) = \begin{cases} e^{-ax} + be^{-(x-c)^2} & 0 < x < 20; \\\ 0 & \text{otherwise}.\end{cases}$$
Note that we have not taken care to ensure that the PDF is properly normalised (i.e. that it integrates to 1), we will still be able to correctly the sample the shape of the PDF.
```python
import numpy as np

def prob(x, a, b, c):
    p = np.exp(-a * x) + b * np.exp(-(x - c)**2)
    return p
```

To draw a sample from this PDF for *one* set of PDF parameters, use `sample_single`:
```python
from invsample import sample_single

a = 0.5
b = 0.5
c = 4
args = [a, b, c]

x = sample_single(prob_fn=prob, args=args, x_min=0, x_max=20)
```
The variable `x` returned here is a float, representing a single sample from the PDF described above.

### Drawing N samples from a single PDF

To draw an array of samples (still all from the same PDF) rather than a single float, simply provide the argument `N` to `sample_single`:
```python
samples = sample_single(prob_fn=prob, args=args, N=200000, x_min=0, x_max=20)
```
The returned object `samples` here is a 1D `numpy` array, length 200000. 


### Drawing samples from a sequence of PDFs

Sometimes, one wishes to draw samples not from a single PDF but from a sequence of similar PDFs of the same functional form but with different parameter values, e.g. a sequence of Gaussians, each with different means and widths. In principle, one could do this a `for` loop and `sample_single`, but a faster, vectorised implementation is given by `sample_family`.

The following example draws samples from a comb of 6 Gaussians (one sample each). 
```python
import numpy as np
from invsample import sample_single

def gaussian(x, mu, sig):
    p = np.exp(-0.5 * (x - mu)**2 / sig**2) / np.sqrt(2 * np.pi) * sig
    return p

mu = np.array([0, 2, 4, 6, 8, 10])
sig = 0.5 * np.ones_like(mu)
args = [mu, sig]

samples = sample_family(prob_fn=gaussian, N_pdfs=6, args=args, x_min=-5, x_max=15)
```
The returned object `samples` here is a 1D `numpy` array, length 6. 

As in the case of a single PDF above, one can draw multiple samples using the argument `N`:
```python
samples = sample_family(prob_fn=gaussian, N_pdfs=6, args=args, N=200000, x_min=-5, x_max=15)
```
Now, `samples` is a 2D `numpy` array, shape (6, 200000).


## Prerequisites

The only prerequisites are `numpy` and `scipy`. The code was developed and tested using versions `1.21.5` and `1.7.3` respectively, but earlier/later versions likely to work also.
