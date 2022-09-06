#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for drawing samples from arbitrary, unnormalised 1D PDFs using
inverse transform sampling. 'sample_single' samples from a single PDF, while
'sample_family' draws from a sequence of PDFs, each described by the same
function but different parameters. See README for more details.

Created: September 2022
Author: A. P. Naik
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid as ctrapz


def sample_single(prob_fn, x_min=0, x_max=1, args=[], N=1, seed=None,
                  N_pts=5000, test_convergence=True):
    """
    Draw N samples from a single PDF described by prob_fn.

    Parameters
    ----------
    prob_fn : function
        Function that computes univariate PDF p(x) that is being sampled from.
        Function should take independent variable x as first argument (where
        x can be a numpy array of values), and (optionally) PDF parameters as
        subsequent arguments (see 'args' below). The function should return the
        (not necessarily normalised) probability p(x), which should take the
        same shape as x.
    x_min : float, optional
        The minimum of the range of p(x). The default is 0.
    x_max : float, optional
        The maximum of the range of p(x). The default is 1.
    args : list, optional
        List of additional arguments to feed to prob_fn. The default is [].
    N : int, optional
        Number of samples to draw. The default is 1.
    seed : int, numpy.random.Generator, or None; optional
        If int, then numpy random number generator is created using
        np.random.default_rng(seed). Alternatively, pre-instanced Generator can
        be used. Finally, if seed is None, then Generator with seed 42 is used.
        The default is None.
    N_pts : int, optional
        Number of points for PDF->CDF integration. The default is 5000.
    test_convergence : bool, optional
        If True, a test will be performed to ensure that PDF->CDF integration
        yields virtually the same result using N_pts/2. If this test fails then
        N_pts (see above) should be increased. Note that performance will
        be improved by setting this to False. The default is True.

    Returns
    -------
    samples : float or numpy array
        Sample(s) from PDF described by prob_fn. If N is 1 (see above), then
        this is a float, otherwise it is a one-dimensional numpy array of
        length N.

    """

    # set up coordinate array, evaluate PDF along it and integrate for CDF
    q = np.linspace(x_min, x_max, N_pts)
    p = prob_fn(q, *args)
    F = ctrapz(p, q, initial=0)

    # convergence test: check integral the same with N_pts/2
    if test_convergence:
        N_test = N_pts // 2
        q_test = np.linspace(x_min, x_max, N_test)
        p_test = prob_fn(q_test, *args)
        F_test = ctrapz(p_test, q_test, initial=0)
        np.testing.assert_approx_equal(
            F_test[-1], F[-1], significant=6, err_msg='Try increasing N_pts.'
        )

    # normalise CDF
    F /= F[-1]

    # random number generator
    if seed is None:
        rng = np.random.default_rng(42)
    elif type(seed) is int:
        rng = np.random.default_rng(seed)
    elif type(seed) is np.random.Generator:
        rng = seed
    else:
        raise TypeError("Can't understand type(seed)")

    # draw uniform sample
    u = rng.uniform(size=N)

    # interpolate from inverse CDF
    m = u[:, None] < F
    i1 = np.argmax(m, axis=-1)
    i0 = i1 - 1
    F0 = F[i0]
    F1 = F[i1]
    q0 = q[i0]
    q1 = q[i1]
    samples = (q0 * (F1 - u) + q1 * (u - F0)) / (F1 - F0)

    # if only one sample requested, return float not np array
    if N == 1:
        samples = samples.item()

    return samples


def sample_family(prob_fn, N_dists, args, x_min=0, x_max=1, N=1, seed=None,
                  N_pts=5000, test_convergence=True):
    """
    Draw N samples per PDF from a sequence of N_dists PDFs. The PDFs are all
    described by the same function (prob_fn), but with different parameter
    values given via 'args'.

    Parameters
    ----------
    prob_fn : function
        Function that computes univariate PDF p(x) that is being sampled from.
        Function should take independent variable x as first argument (where
        x can be a numpy array of values), and PDF parameters as
        subsequent arguments (see 'args' below). These parameters should appear
        in the prob_fn only in basic vectorisable operations. The function
        should return the (not necessarily normalised) probability p(x), which
        should take the same shape as x.
    N_dists : int
        Number of PDFs in sequence.
    args : list
        List of additional arguments to feed to prob_fn. Each element in list
        should represent a PDF parameter, and should be a 1D numpy array length
        N_dists, giving sequence of values for the parameter in question.
    x_min : float, optional
        The minimum of the range of p(x). The default is 0.
    x_max : float, optional
        The maximum of the range of p(x). The default is 1.
    N : int, optional
        Number of samples to draw *per PDF*. The default is 1.
    seed : int, numpy.random.Generator, or None; optional
        If int, then numpy random number generator is created using
        np.random.default_rng(seed). Alternatively, pre-instanced Generator can
        be used. Finally, if seed is None, then Generator with seed 42 is used.
        The default is None.
    N_pts : int, optional
        Number of points for PDF->CDF integration. The default is 5000.
    test_convergence : bool, optional
        If True, a test will be performed to ensure that PDF->CDF integration
        yields virtually the same result using N_pts/2. If this test fails then
        N_pts (see above) should be increased. Note that performance will
        be improved by setting this to False. The default is True.

    Returns
    -------
    samples : numpy array
        Samples from PDF sequence. If N is 1, i.e. one sample per PDF, then
        samples is a 1D array length N_dists, otherwise it is a 2D array shaped
        (N_dists, N).

    """

    # check args and unsqueeze
    if len(args) == 0:
        raise ValueError("Expected non-empty list of args for prob_fn!")
    args_unsqueezed = []
    for arg in args:
        assert arg.shape == (N_dists,), "Argument has wrong shape!"
        args_unsqueezed.append(arg[:, None])

    # set up coordinate array, evaluate PDF and integrate for CDF
    q = np.linspace(x_min, x_max, N_pts)
    p = prob_fn(q[None], *args_unsqueezed)
    F = ctrapz(p, q, initial=0)

    # convergence test: check integral the same with N_pts/2
    if test_convergence:
        N_test = N_pts // 2
        q_test = np.linspace(x_min, x_max, N_test)
        p_test = prob_fn(q_test[None], *args_unsqueezed)
        F_test = ctrapz(p_test, q_test, initial=0)
        np.testing.assert_allclose(
            F_test[:, -1], F[:, -1], rtol=1e-6, err_msg='Try increasing N_pts.'
        )

    # normalise CDF
    F /= F[:, -1][:, None]

    # random number generator
    if seed is None:
        rng = np.random.default_rng(42)
    elif type(seed) is int:
        rng = np.random.default_rng(seed)
    elif type(seed) is np.random.Generator:
        rng = seed
    else:
        raise TypeError("Can't understand type(seed)")

    # draw uniform sample
    u = rng.uniform(size=(N_dists, N))

    # interpolate from inverse CDF
    m = u[..., None] < F[:, None]
    i1 = np.argmax(m, axis=-1)
    i0 = i1 - 1
    a = np.tile(np.arange(N_dists)[:, None], reps=(1, N))
    F0 = F[a, i0]
    F1 = F[a, i1]
    q0 = q[i0]
    q1 = q[i1]
    samples = (q0 * (F1 - u) + q1 * (u - F0)) / (F1 - F0)

    # if only one sample per PDF requested, return 1D array not 2D.
    if N == 1:
        samples = samples.squeeze()

    return samples
