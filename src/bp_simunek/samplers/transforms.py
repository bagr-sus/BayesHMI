import numpy as np
import scipy.stats as sps


def new_to_old_model(is_x, is_y, k0, eps, delta, gamma, is_z=60e6, sigma_c=55e6):
    kr = k0 / eps
    sigma_0 = -(is_x + is_y + is_z) / 3
    gamma_old = gamma / sigma_c
    km = delta
    beta = np.log((1 - 1 / eps) * k0 / delta) / sigma_0

    return kr, km, beta, gamma_old


def transform_params(params, priors):
    trans_params = []
    for param, prior in zip(params, priors):
        match prior["type"]:
            case "lognorm":
                trans_param = np.exp(param)
            case "truncnorm":
                a, b, mu, sigma = prior["params"]
                lower_bound = (a - mu) / sigma
                upper_bound = (b - mu) / sigma
                phi_a = sps.norm.cdf(lower_bound)
                phi_b = sps.norm.cdf(upper_bound)
                phi_param = sps.norm.cdf(param, loc=mu, scale=sigma)
                trans_param = sps.norm.ppf((phi_b - phi_a)*phi_param + phi_a)*sigma + mu

        trans_params.append(trans_param)
    return trans_params