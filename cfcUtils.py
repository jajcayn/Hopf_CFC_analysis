import numpy as np

"""
Utility functions for Cross-Frequency Coupling, measures calculated as in:
    Jajcay Nikola, Cakan Caglar, Obermayer Klaus. (2022) Cross-Frequency Slow Oscillation–Spindle 
    Coupling in a Biophysically Realistic Thalamocortical Neural Mass Model. Frontiers in 
    Computational Neuroscience, DOI:10.3389/fncom.2022.769860    
"""


def modulation_index(P):
    """
    MI = KL(P,U)/log(n), where U is a uniform distribution.
    :param P: mean amplitude for each phase bin, shape (N_BINS,)
    :return modulation index, quantitative measure of phase amplitude coupling
    """
    n_bins = P.shape[0]
    try:
        assert np.isclose(np.sum(P), 1)
    except AssertionError:
        print("Please normalize P!")

    kl = np.log(n_bins) + np.sum(P[P != 0] * np.log(P[P != 0]))
    return kl / np.log(n_bins)


def modulation_index_general(amplitude_fast, phase_slow, n_bins=18):
    """
    Given the time series of the amplitudes for the fast components, and the time series for
    the amplitudes of the slow components, we compute the Kullback-Leibler divergence
    between the distribution of fast amplitudes over the slow oscillation phase bins
    and a uniform distribution.

    MI = KL(P,U)/log(n), where U is a uniform distribution.

    :param amplitude_fast: time series of the amplitude of the fast component
    :param phase_slow: time series of the phase of the slow component
    :return modulation index, quantitative measure of phase amplitude coupling
    """

    binned_phase = np.digitize(phase_slow, bins=np.linspace(-np.pi, np.pi, n_bins + 1))
    mean_bin_amp = np.zeros(n_bins + 1)  # in theory index of bins goes from 0 to N_BINS
    for bin_idx in np.unique(binned_phase):
        mean_bin_amp[bin_idx] = np.mean(amplitude_fast[binned_phase == bin_idx])

    mean_bin_amp = mean_bin_amp[
                   1:
                   ]  # because in theory there could be stuff that is smaller than -pi, then actually the interval between -pi and the next bin has index 1.
    # normalize the mean amplitude in each bin

    mean_bin_amp = mean_bin_amp / np.sum(mean_bin_amp)

    kl = np.log(n_bins) + \
         np.sum(mean_bin_amp[mean_bin_amp != 0] * np.log(mean_bin_amp[mean_bin_amp != 0]))

    return kl / np.log(n_bins)


def mean_vector_length(amplitude_fast, phase_slow):
    """
    The length (real part) of the result represents the amount of phase-amplitude
    coupling, while the phase represents the mean phase where amplitude
    is strongest.

    :param amplitude_fast: time series of the amplitude of the fast component
    :param phase_slow: time series of the phase of the slow component
    :return: complex value for mean vector length
    """
    temp = amplitude_fast * np.exp(phase_slow * 1j)
    return np.mean(temp)


def phase_locking_value(phase_fast, phase_slow):
    """
    Return complex phase locking value between two
    phase time series. The length of the PLV indicates
    the strength of phase locking, while the angle represents a
    phase shift.

    :param phase_fast: time series of the phase for the fast component
    :param phase_slow: time series of the phase for the fast component
    :return: complex phase locking value
    """
    return np.mean(np.exp((phase_slow - phase_fast) * 1j))


def _standardize_ts(ts):
    """

    Author: Nikola Jajcay

    Returns centered time series with zero mean and unit variance.
    """
    assert np.squeeze(ts).ndim == 1, "Only 1D time series can be centered"
    ts -= np.mean(ts)
    ts /= np.std(ts, ddof=1)

    return ts

def mutual_information(
        x, y, algorithm="EQQ", bins=None, k=None, log2=True, standardize=True
):
    """
    In our case, x should be the time series of the phase of the fast component and y should be
    the time series of the phase of the slow component.


    Author: Nikola Jajcay


    Compute mutual information between two timeseries x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) )
    where p(x), p(y) and p(x, y) are probability distributions.

    :param x: first timeseries, has to be 1D
    :type x: np.ndarray
    :param y: second timeseries, has to be 1D
    :type y: np.ndarray
    :param algorithm: which algorithm to use for probability density estimation:
        - EQD: equidistant binning [1]
        - EQQ_naive: naive equiquantal binning, can happen that samples with
            same value fall into different bins [2]
        - EQQ: equiquantal binning with edge shifting, if same values happen to
            be at the bin edge, the edge is shifted so that all samples of the
            same value will fall into the same bin, can happen that not all the
            bins have necessarily the same number of samples [2]
        - knn: k-nearest neighbours search using k-dimensional tree [3]
        number of bins, at least for EQQ algorithms, should not exceed 3rd root
            of the number of the data samples, in case of I(x,y), i.e. MI of
            two variables [2]
    :param bins: number of bins for binning algorithms
    :type bins: int|None
    :param k: number of neighbours for knn algorithm
    :type k: int|None
    :param log2: whether to use log base 2 for binning algorithms, then the
        units are bits, if False, will use natural log which makes the units
        nats
    :type log2: bool
    :param standardize: whether to standardize timeseries before computing MI,
        i.e. transformation to zero mean and unit variance
    :type standardize: bool
    :return: estimate of the mutual information between x and y
    :rtype: float

    [1] Butte, A. J., & Kohane, I. S. (1999). Mutual information relevance
        networks: functional genomic clustering using pairwise entropy
        measurements. In Biocomputing 2000 (pp. 418-429).
    [2] Paluš, M. (1995). Testing for nonlinearity using redundancies:
        Quantitative and qualitative aspects. Physica D: Nonlinear Phenomena,
        80(1-2), 186-205.
    [3] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
        information. Physical review E, 69(6), 066138.
    """
    assert x.ndim == 1 and y.ndim == 1, "Only 1D timeseries supported"
    if standardize:
        x = _standardize_ts(x)
        y = _standardize_ts(y)

    if algorithm == "knn":
        assert k is not None, "For knn algorithm, `k` must be provided"
        data = np.vstack([x, y]).T
        # build k-d tree
        tree = cKDTree(data, leafsize=LEAF_SIZE)
        # find k-nearest neighbour indices for each point, use the maximum
        # (Chebyshev) norm, which is also limit p -> infinity in Minkowski
        _, ind = tree.query(data, k=k + 1, p=np.inf)
        sum_ = 0
        for n in range(data.shape[0]):
            # find x and y distances between nth point and its k-nearest
            # neighbour
            eps_x = np.abs(data[n, 0] - data[ind[n, -1], 0])
            eps_y = np.abs(data[n, 1] - data[ind[n, -1], 1])
            # use symmetric algorithm with one eps - see the paper
            eps = np.max((eps_x, eps_y))
            # find number of points within eps distance
            n_x = np.sum(np.less(np.abs(x - x[n]), eps)) - 1
            n_y = np.sum(np.less(np.abs(y - y[n]), eps)) - 1
            # add to digamma sum
            sum_ += digamma(n_x + 1) + digamma(n_y + 1)

        sum_ /= data.shape[0]

        mi = digamma(k) - sum_ + digamma(data.shape[0])

    elif algorithm.startswith("E"):
        assert (
                bins is not None
        ), "For binning algorithms, `bins` must be provided"
        log_f = np.log2 if log2 else np.log

        if algorithm == "EQD":
            # bins are simple number of bins - will be divided equally
            x_bins = bins
            y_bins = bins

        elif algorithm == "EQQ_naive":
            x_bins = _create_naive_eqq_bins(x, no_bins=bins)
            y_bins = _create_naive_eqq_bins(y, no_bins=bins)

        elif algorithm == "EQQ":
            x_bins = _create_shifted_eqq_bins(x, no_bins=bins)
            y_bins = _create_shifted_eqq_bins(y, no_bins=bins)

        else:
            raise ValueError(f"Unknown MI algorithm: {algorithm}")

        xy_bins = [x_bins, y_bins]

        # compute histogram counts
        count_x = np.histogramdd([x], bins=[x_bins])[0]
        count_y = np.histogramdd([y], bins=[y_bins])[0]
        count_xy = np.histogramdd([x, y], bins=xy_bins)[0]

        # normalise
        count_xy /= np.float(np.sum(count_xy))
        count_x /= np.float(np.sum(count_x))
        count_y /= np.float(np.sum(count_y))

        # sum
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if count_x[i] != 0 and count_y[j] != 0 and count_xy[i, j] != 0:
                    mi += count_xy[i, j] * log_f(
                        count_xy[i, j] / (count_x[i] * count_y[j])
                    )

    else:
        raise ValueError(f"Unknown MI algorithm: {algorithm}")

    return mi
