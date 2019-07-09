import logging
from collections import namedtuple
import numpy as np
# from scipy.special import erf, gammainc, gammaincc, gamma  # needed for poissonian fits
# from scipy import optimize
from sklearn import mixture

logger = logging.getLogger(__name__)

DoubleGaussianParameterSet = namedtuple('DoubleGaussianParameterSet', [
    'threshold', 'single_atom_probability',
    'background_mean', 'background_sigma',
    'single_atom_mean', 'single_atom_sigma',  # these are the distribution parameters, do not convolve with background
    'single_gaussian_max_loglikelihood', 'double_gaussian_max_loglikelihood'
])
DoubleGaussianParameterSet.__new__.__defaults__ = (None,) * len(DoubleGaussianParameterSet._fields)


def generate_parameter_guess_otsu(roi_count_vector):
    """Find threshold guess for a 1D (not-binned) count array using Otsu's method.

    :param roi_count_vector: roi_count_vector: The raw count vector to be processed
    :return: A DoubleGaussianParameterSet object containing the result of the fit.
    """
    n = len(roi_count_vector)
    # step through the counts in sorted order and find the minimum variance
    sorted_counts = np.sort(roi_count_vector)
    min_intra_class_variance = None
    threshold_idx = -1
    for i in range(1, n):
        split_counts = np.split(sorted_counts, [i])
        class_weights = np.array([i, n-i])/n
        class_variances = [np.var(x) for x in split_counts]
        intra_class_variance = np.dot(class_weights, class_variances)
        if min_intra_class_variance is None or min_intra_class_variance > intra_class_variance:
            if not np.isnan(intra_class_variance):
                min_intra_class_variance = intra_class_variance
                threshold_idx = i
    # recalculate the initial fit parameters
    split_counts = np.split(sorted_counts, [threshold_idx])
    class_weights = np.array([threshold_idx, n - threshold_idx]) / n
    class_means = [np.mean(x) for x in split_counts]
    class_sigmas = [np.std(x) for x in split_counts]
    # take the mean of the two points on either side of the threshold index
    threshold = np.mean(sorted_counts[threshold_idx:threshold_idx + 2])
    result = DoubleGaussianParameterSet(
        threshold=threshold,
        single_atom_probability=class_weights[1],
        background_mean=class_means[0],
        background_sigma=class_sigmas[0],
        single_atom_mean=class_means[1],
        single_atom_sigma=class_sigmas[1],
    )
    return result


def generate_parameter_guess_gmm(roi_count_vector, max_atoms=1):
    """Find the threshold and parameter guess using a Gaussian Mixture Matrix.

    :param roi_count_vector: The raw count vector to be processed
    :param max_atoms: The most number of atoms to consider to describe the data
    :return: A DoubleGaussianParameterSet object containing the result of the fit.
    """
    gmix = mixture.GaussianMixture(n_components=max_atoms+1, covariance_type='diag')
    gmix.fit(np.transpose([roi_count_vector]))
    # order the components by the size of the signal
    indices = np.argsort(gmix.means_.flatten())
    guess_gauss = []
    for n in range(max_atoms + 1):
        idx = indices[n]
        guess_gauss.append([
            gmix.weights_[idx],  # amplitudes
            gmix.means_.flatten()[idx],  # x0s
            np.sqrt(gmix.covariances_.flatten()[idx])  # sigmas
        ])
    # reorder the parameters, drop the 0 atom amplitude
    guess_gauss = np.transpose(guess_gauss).flatten()[1:]
    # construct the parameter set with no threshold, this is so we can pass the expect tuple to dependent functions
    dbl_g_params = DoubleGaussianParameterSet(
        single_atom_probability=guess_gauss[0],
        background_mean=guess_gauss[1],
        background_sigma=guess_gauss[3],
        single_atom_mean=guess_gauss[2],
        single_atom_sigma=guess_gauss[4],
    )
    # now we want to update the tuple, so convert to a dictionary, update the values, and return a new tuple
    dbl_g_params_dict = dbl_g_params._asdict()
    dbl_g_params_dict.update(
        threshold=intersection(dbl_g_params),  # find the threshold and rebuild the parameter set
        # find the maximum log-likelihood for the data with a single Gaussian distribution
        single_gaussian_max_loglikelihood=gaussian_max_loglikelihood(roi_count_vector),
        # find the maximum log-likelihood for the data with a double Gaussian distribution
        double_gaussian_max_loglikelihood=double_gaussian_max_loglikelihood(roi_count_vector, dbl_g_params)
    )
    return DoubleGaussianParameterSet(**dbl_g_params_dict)


def gaussian(x, mean, stddev):
    """A simple normalized 1D Gaussian function.

    :param x: The points to evaluate at
    :param mean: The distribution mean
    :param stddev: The distribution standard deviation
    :return: The guassian function evaluated at x
    """
    return np.exp(-0.5*((x - mean) / stddev)**2) / np.sqrt(2*np.pi*stddev**2)


def double_gaussian(x, dbl_gaussian_parameters):
    """Evaluate the double guassian function given by dbl_gaussian_parameters at the point or numpy array x

    :param x: The points at which to evaluate the the function
    :param dbl_gaussian_parameters: The function parameters expected to be a DoubleGaussianParameterSet
    :return: A single value or numpy array of the function evaluated at x
    """
    a = dbl_gaussian_parameters.single_atom_probability
    return a * gaussian(x, dbl_gaussian_parameters.single_atom_mean, dbl_gaussian_parameters.single_atom_sigma) + \
        (1 - a) * gaussian(x, dbl_gaussian_parameters.background_mean, dbl_gaussian_parameters.background_sigma)


def intersection(dbl_gauss_parameters):
    """Returns the x-value where the two gaussian distributions intersect.

    :param dbl_gauss_parameters: The function parameters expected to be a DoubleGaussianParameterSet
    :return: The x-position where the two Gaussian distributions intersect
    """
    _, a1, m0, s0, m1, s1 = dbl_gauss_parameters[:6]
    if a1 < 0 or a1 > 1:
        logger.debug('bad fit params: {}'.format(dbl_gauss_parameters))
        return np.nan
    # find x where (1-a1)*G(x, m0, s0) == a1*G(x, m1, s1), where G is a normalized Gaussian distribution
    temp = s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*np.log((1-a1)/a1)*(s1**2-s0**2))
    if temp < 0:  # check that we won't get an imaginary intersection point
        logger.debug('bad fit params: {}'.format(dbl_gauss_parameters))
        return np.nan
    temp = m1*s0**2-m0*s1**2-np.sqrt(temp)
    return temp/(s0**2-s1**2)


def gaussian_max_loglikelihood(roi_count_vector):
    """Calculate the maximum log-likelihood for a single gaussian distribution describing the data.

    :param roi_count_vector: The raw counts vector.
    :return: The Maximum log-likelihood of the data if described by a single-modal Gaussian distribution.
    """
    # For a single gaussian distribution the mean and std of the dataset corresponds to the max log-likelihood
    # parameters
    mean = np.nanmean(roi_count_vector)  # data
    var = np.nanvar(roi_count_vector)  # standard dev, ignore nans
    n = len(roi_count_vector)
    # MFE: why isn't the second term equivalent to n * var? Am I dumb?
    mll = n * np.log(1 / np.sqrt(2 * np.pi * var)) - (0.5/var) * np.sum((roi_count_vector - mean)**2)
    return mll


def double_gaussian_max_loglikelihood(roi_count_vector, dbl_gaussian_params):
    """Calculate the maximum log-likelihood for a double Gaussian distribution given by the input parameters.

    :param roi_count_vector: The raw counts vector
    :param dbl_gaussian_params: The function parameters expected to be a DoubleGaussianParameterSet
    :return: The Maximum log-likelihood of the data if described by the provided double gaussian parameters.
    """
    return np.sum(np.log(double_gaussian(roi_count_vector, dbl_gaussian_params)))


def calculate_new_thresholds(roi_count_vector):
    """Calculate the thresholds for binarization of a 1D vector of counts.

    :param roi_count_vector: The raw counts vector
    :return: A threshold object (maybe this should be the whole parameters object?)
    """
    double_gausssian_parameters = generate_parameter_guess_gmm(roi_count_vector)
    return double_gausssian_parameters.threshold


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer

    # plot the waveforms and the difference
    runs = 5
    sa_sig = 0
    bg = 100
    fig, ax = plt.subplots(runs, 1, figsize=(5, 14))
    for r in range(runs):

        start = timer()
        for _ in range(100):
            d = np.array([[np.sqrt(bg)], [np.sqrt(bg+sa_sig)]] * np.random.randn(2, 100) + [[bg], [bg+sa_sig]]).flatten()
            dbl_gaussian_params, vars, sc = generate_parameter_guess_otsu(d)
        end = timer()
        print('time per loop, otsu:\t{:.5f}'.format((end - start)/100))

        diffs = []  # keep track of the single peak vs dbl peak likelihoods for calculation of a prior at some point
        start = timer()
        for _ in range(100):
            d = np.array([[np.sqrt(bg)], [np.sqrt(bg+sa_sig)]] * np.random.randn(2, 100) + [[bg], [bg+sa_sig]]).flatten()
            try:
                dbl_gaussian_params2 = generate_parameter_guess_gmm(d)
                diffs.append(dbl_gaussian_params2.double_gaussian_max_loglikelihood - dbl_gaussian_params2.single_gaussian_max_loglikelihood)
            except ValueError as e:
                print("not enough points")
        end = timer()
        print('time per loop, gmm:\t{:.5f}'.format((end - start) / 100))
        print('[{}] log mll mean diff: {:.5f}'.format(r, np.mean(diffs)))
        print('='*40)

        # d has changed here so the plotting comparison is not really fair...
        ax[r].hist(d, normed=True, bins=30)
        xs = np.linspace(
            dbl_gaussian_params.background_mean - 3 * dbl_gaussian_params.background_sigma,
            dbl_gaussian_params.single_atom_mean + 3 * dbl_gaussian_params.single_atom_sigma,
            100
        )
        ax[r].plot(xs, double_gaussian(xs, dbl_gaussian_params))
        try:
            ax[r].plot(xs, double_gaussian(xs, dbl_gaussian_params2))
            ax[r].axvline(x=dbl_gaussian_params2.threshold, color='blue')
        except NameError:
            pass
        ax[r].plot(sc, vars*np.nanmax(double_gaussian(xs, dbl_gaussian_params))/np.nanmax(vars))
        ax[r].axvline(x=dbl_gaussian_params.threshold, color='red')

        sa_sig += 5
    plt.savefig('test.png', format='png')
