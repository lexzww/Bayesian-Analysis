import random
import time
import zmq
import json
import numpy
import matplotlib.pyplot as plt
import pprint
from scipy.stats import poisson
import warnings
from multiprocessing import Pool
from functools import partial
from scipy.optimize import curve_fit
from threshold_analysis_example import generate_parameter_guess_gmm
import itertools

"""a function that returns the prior for choosing single/double
    poisson fit function to fix the problem of wrong seperation when there only background and no loading rate.
    """
class Bayesian():
    @classmethod

    def fit_func_gmm(self, data_set):
        """a function that return maximum likelihood of single and double fit
            args:
            data_set: a collection of 5000 trials of random generated data with certain length. """
        difference = numpy.array([])
        for data in data_set:
            """args:
                data: a single measurement/ random poisson data with a certain length"""
            single_maxlikelihood = generate_parameter_guess_gmm(data)[6]
            double_maxlikelihood = generate_parameter_guess_gmm(data)[7]
            difference = numpy.append(difference, numpy.subtract(double_maxlikelihood, single_maxlikelihood))
        return difference

    def prior_plot(self, difference, size, fn):
        """
            calculate prior by fitting input double poission distribution data (positive condition, difference - prior >0);
            plot cumulative distribution of difference in max likelihood array,
            and find prior value by given false negative rate"""
        (n, bins, patches) = plt.hist(difference, bins = len(difference), density = True, histtype='step', cumulative=True)
        plt.title('Cumulative Distribution Plot of Maximum Loglikelihood of Double Poisson Distribution')
        plt.show()
        idx = int(size*fn)
        return bins[idx]

# Generating raw_data set function
    def generate_data(self, size, a, b, c, d, n, l):
        """function that generates a set of random data with double posiion distribution
            size: number of data set in a collection
            mu: average in poisson distribution
            l: length of measurement"""

        l = int(0.5*l)
        data_set = numpy.array([numpy.concatenate((numpy.array([ numpy.random.poisson(a+k*b) for k in numpy.random.poisson(n,l) ]), numpy.array([ numpy.random.poisson(c+k*d) for k in numpy.random.poisson(n,l)]))) for i in range(size)])
        return data_set

    def generate_data_poisson(self, size, mu, l):
        """function that generates random data set that fit poisson distribution
            size: number of data set in a collection
            mu: average in poisson distribution
            l: length of measurement"""
        data_set = numpy.array([numpy.random.poisson(mu, l) for i in numpy.arange(size)])
        return data_set

#unpack function from class so can be used for pool.map
def generate_data_unpack(args):
    test = Bayesian()
    return test.generate_data(*args)

def generate_data_poisson_unpack(args):
    test = Bayesian()
    return test.generate_data_poisson(*args)

def prior_plot_unpack(args):
    """unpack arguments in prior_plot
    args: list of argument passed in prior_plot"""
    test = Bayesian()
    return test.prior_plot(*args)

def fit_func_gmm_unpack(args):
    test = Bayesian()
    return test.fit_func_gmm(*args)
