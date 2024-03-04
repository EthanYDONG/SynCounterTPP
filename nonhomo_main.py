import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import json
from counterfactual_tpp import sample_counterfactual, combine, check_monotonicity
from sampling_utils import thinning_T
from util_funcs import normal


# Global constants
LAMBDA_MAX = 100
T = 5
NUMBER_OF_GAUSSIANS = 5


def original_intensity(x, means, sds, amps):
    """Original intensity function of inhomogeneous Poisson process."""
    res = 0
    for i in range(NUMBER_OF_GAUSSIANS):
        res += normal(x, means[i], sds[i], amps[i])
    return res


def intervention_intensity(x, means, sds, new_amps):
    """Intervention intensity function."""
    res = 0
    for i in range(NUMBER_OF_GAUSSIANS):
        res += normal(x, means[i], sds[i], new_amps[i])
    return res


def load_sample_data(sample_dir,indicator_dir):
    """Load sample data and indicators."""
    samples_load = np.load(sample_dir, allow_pickle=True)
    with open(indicator_dir, "r") as read_file:
        indicators_load = json.load(read_file)
    return samples_load, indicators_load


def run_counterfactual_experiment(sample_data, means, sds, amps):
    """Run counterfactual experiment."""
    sample, indicators, samples_load, indicators_load = sample_data
    h_observed = sample[indicators]
    lambda_observed = [original_intensity(i, means, sds, amps) for i in h_observed]
    lambda_bar = lambda x: LAMBDA_MAX - original_intensity(x, means, sds, amps)
    h_rejected, _ = thinning_T(0, intensity=lambda_bar, lambda_max=LAMBDA_MAX, T=T)
    sample, _, indicators = combine(h_observed, lambda_observed, h_rejected, original_intensity)
    lambdas = original_intensity(sample, means, sds, amps)
    counters = []
    for counter in range(100):
        counterfactuals, counterfactual_indicators = sample_counterfactual(sample, lambdas, LAMBDA_MAX, indicators, intervention_intensity)
        if check_monotonicity(sample, counterfactuals, original_intensity, intervention_intensity, sample[indicators]) != 'MONOTONIC':
            print('Not monotonic')
        counters.append(counterfactuals)
    return sample[indicators], counters


if __name__ == "__main__":
    # Define means, sds, amps
    means = np.arange(1, T, step=(T - 1) / NUMBER_OF_GAUSSIANS)
    means[0] = 0.55
    sds = np.random.uniform(low=0, high=0.5, size=NUMBER_OF_GAUSSIANS)
    amps = 10 * np.random.uniform(low=1.0, high=3.0, size=NUMBER_OF_GAUSSIANS)

    parser = argparse.ArgumentParser(description="Run counterfactual experiment.")
    parser.add_argument("--n_workers", type=int, default=48, help="Number of workers for parallel processing")
    parser.add_argument("--sample_dir",type = str, default='data_inhomogeneous/allsamples.npy')
    parser.add_argument("--indicator_dir",type = str, default='data_inhomogeneous/allindicators.json')
    args = parser.parse_args()

    # Load sample data and indicators
    samples_load, indicators_load = load_sample_data(args.sample_dir,args.indicator_dir)

    # Prepare arguments for the counterfactual experiment
    args_list = [(samples_load[_], indicators_load[_], samples_load, indicators_load) for _ in range(1000)]

    # Run the counterfactual experiment in parallel
    with Pool(args.n_workers) as pool:
        result = list(tqdm(pool.imap(run_counterfactual_experiment, args_list), total=1000))

    
