import os
import logging
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpt
import matplotlib.colors as mcolors
import arviz as az
import numpy as np
import scipy.stats as sps

from ..samplers.tinyda_flow import TinyDAFlowWrapper
from ..samplers.surrogates.flow_torch_wrapper import Wrapper as NNwrapper
from ..simulation.flow_wrapper import Wrapper
from ..samplers.idata_tools import read_idata_from_file, idata_from_observe_times
from ..plotting.plotting_tools import save_plot, save_plots_pdf_pages
from ..definitions import ROOT_DIR

def plot_pressures(idata: az.InferenceData, exp):
    plt.figure()
    plt.xlabel("Čas [den]")
    plt.ylabel("Tlaková výška [m]")
    plt.title("Změna tlakové výšky vrtu H1 v čase")
    obs_keys = [f"obs_{idx}" for idx in np.arange(0, 26)]

    if "times" not in idata["sample_stats"].attrs:
        logging.warning("Missing time intervals in idata, cannot plot pressures")
        return
    times = idata["sample_stats"].attrs["times"]
    plt.xlim(times[0] - 5, times[-1] + 5)

    exp_plot, = plt.plot(times, exp, color="black", linewidth=1, linestyle="dotted")
    areas = 100
    quantiles_95 = []
    quantiles_75 = []
    quantiles_25 = []
    quantiles_5 = []
    means = []
    medians = []

    norm_constant = 7000

    prev_time = 0
    prev_linspace = np.ones(areas) * 275
    prev_normhist = np.ones(areas) / areas
    for time_idx, key in enumerate(obs_keys):
        #observed_unfiltered = idata["posterior_predictive"][key]
        #observed = observed_unfiltered.where((observed_unfiltered <= 500) & (observed_unfiltered > 0))
        observed = idata["posterior_predictive"][key]
        mean = observed.mean()
        std = observed.std()
        minimum = observed.min()
        maximum = observed.max()
        means.append(mean)
        medians.append(np.median(observed))
        quantiles_95.append(np.quantile(observed, 0.95))
        quantiles_75.append(np.quantile(observed, 0.75))
        quantiles_25.append(np.quantile(observed, 0.25))
        quantiles_5.append(np.quantile(observed, 0.05))


        """  linspace = np.linspace(minimum, maximum, areas)
        
        interp_linspace = np.divide(np.add(prev_linspace, linspace), 2)
        interp_time = (prev_time + times[time_idx]) / 2

        hist, bins = np.histogram(observed, linspace)

        norm_hist = np.subtract(hist, np.min(hist))
        #norm_hist = np.divide(norm_hist, np.max(norm_hist))
        norm_hist = np.divide(norm_hist, norm_constant)

        cmap = mpl.colormaps["Oranges"].resampled(areas)

        for area in np.arange(1, areas):
            pdf_a = prev_normhist[area-1] * 0.7 + norm_hist[area-1] * 0.3
            pdf_b = prev_normhist[area-1] * 0.3 + norm_hist[area-1] * 0.7
            plt.fill_between([prev_time, interp_time], [prev_linspace[area-1], interp_linspace[area-1]], [prev_linspace[area], interp_linspace[area]], color=cmap(pdf_a))
            plt.fill_between([interp_time, times[time_idx]], [interp_linspace[area-1], linspace[area-1]], [interp_linspace[area], linspace[area]], color=cmap(pdf_b))
            #plt.fill_between([prev_time, times[time_idx]], [prev_linspace[area-1], linspace[area-1]], [prev_linspace[area], linspace[area]], color=cmap(norm_hist[area-1]))

        prev_time = times[time_idx]
        prev_linspace = linspace
        prev_normhist = norm_hist """

    quantiles_95_plot, = plt.plot(times, quantiles_95, color="blue", linewidth=1)
    #quantiles_75_plot, = plt.plot(times, quantiles_75)
    #quantiles_25_plot, = plt.plot(times, quantiles_25)
    quantiles_5_plot, = plt.plot(times, quantiles_5, color="darkblue", linewidth=1)
    median_plot, = plt.plot(times, medians, color="indigo", linewidth=1, linestyle="dashed")

    plt.ylim(np.min(quantiles_5), np.max(quantiles_95))

    #filled_patch = mpt.Patch(color="orange", label="Pravděpodobnostní hustota inverze")

    plt.legend(
        [
            exp_plot, 
            median_plot, 
            quantiles_95_plot, 
            #quantiles_75_plot, 
            #quantiles_25_plot, 
            quantiles_5_plot,
            #filled_patch
        ], 
        [
            "Naměřená data",
            "Medián z inverze", 
            "95. kvantil inverze",
            #"75. kvantil inverze",
            #"25. kvantil inverze",
            "5. kvantil inverze",
            #"Pravděpodobnostní hustota inverze"
        ])
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles.extend([filled_patch])
    #plt.legend(handles, labels)
    return

def data_window_plots(idata: az.InferenceData, window_size):
    draws = idata.posterior.sizes["draw"]
    starts = np.arange(0, draws - window_size)
    ess_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    r_hat_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    mean_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    std_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}

    # process all metrics for all data windows
    for start in starts:
        subset = idata.isel(draw=slice(start, start + window_size))
        ess = az.ess(subset)
        r_hat = az.rhat(subset)
        posterior = subset.posterior
        for param in ess.data_vars:
            ess_list[param] = np.append(ess_list[param], ess[param].values.tolist())
            r_hat_list[param] = np.append(r_hat_list[param], r_hat[param].values.tolist())
            values = posterior[param]
            mean_list[param] = np.append(mean_list[param], values.mean())
            std_list[param] = np.append(std_list[param], values.std())

    # first 2 plots - ESS and r-hat
    fig_corr, axes_corr = plt.subplots(2, 2, width_ratios=[0.75, 0.25])
    fig_corr.set_figwidth(16)
    fig_corr.set_figheight(9)
    axes_corr[0, 0].set_xlabel("Začátek okna [iterace]")
    axes_corr[0, 0].set_ylabel("Effective Sample Size []")
    axes_corr[0, 0].set_title(f"Vývoj ESS s oknem {window_size}")
    refs = []
    for param, ess in ess_list.items():
        refs += axes_corr[0, 0].plot(starts, ess, linewidth=0.5)

    axes_corr[0, 1].legend(refs, list(ess_list.keys()))
    axes_corr[0, 1].axis("off")

    axes_corr[1, 0].set_xlabel("Začátek okna [iterace]")
    axes_corr[1, 0].set_ylabel("r-hat []")
    axes_corr[1, 0].set_title(f"Vývoj r-hat s oknem {window_size}")
    refs = []
    for param, r_hat in r_hat_list.items():
        refs += axes_corr[1, 0].plot(starts, r_hat, linewidth=0.5)

    axes_corr[1, 1].legend(refs, list(r_hat_list.keys()))
    axes_corr[1, 1].axis("off")

    # second pair - mean and std

    figs_stats = []
    for param in mean_list:
        means = mean_list[param]
        stds = std_list[param]

        fig_param, axes_param = plt.subplots(figsize=(16, 9))
        axes_param.set_title(f"Vývoj střední a rozpylu parametru {param} s oknem {window_size}")
        axes_param.set_xlabel("Začátek okna [iterace]")
        axes_param.set_ylabel("Hodnota parametru")
        axes_param.grid(True)

        axes_param.plot(starts, means, label="střední hodnota")
        for i in range(1, 100):
            alpha = (1 - (i / 100))  ** 2 * 0.5
            axes_param.fill_between(
                starts,
                means - stds * (i / 100),
                means + stds * (i / 100),
                alpha = alpha,
                color = "red"
            )
        figs_stats += [fig_param]

    return fig_corr, figs_stats

def load_csv(folder_path, file):
    path = os.path.join(folder_path, file)
    with open(path, "r") as file:
        lines = file.readlines()
        n_elements = len(lines[0].split(","))
        cols_arr = np.empty((0, n_elements))
        for line in lines:
            #cols = [float(i) for i in line.split(",")]
            cols_arr = np.vstack((cols_arr, np.array(line.split(","), dtype=float)))

    return cols_arr


def chain_delay_plot(data):
    cols_avg = np.mean(data, axis=1)
    cols_mean = np.median(data, axis=1)
    cols_max = np.max(data, axis=1)

    x_axis = np.arange(0, data.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    axes[0].plot(x_axis, cols_avg, label="průměrné zpoždění")
    axes[0].plot(x_axis, cols_mean, label="medián zpoždění")
    axes[0].plot(x_axis, cols_max, label="maximum zpoždění")
    axes[0].set_xlabel("n-tý přístup k archivu")
    axes[0].set_ylabel("zpoždění vůči nejrychlejšímu chainu")
    axes[0].legend()
    axes[0].grid(True)

    for i in range(data.shape[1]):
        axes[1].plot(x_axis, data[:, i], label=f"Chain {i}")
    axes[1].set_xlabel("n-tý přístup k archivu")
    axes[1].set_ylabel("zpoždění vůči nejrychlejšímu chainu")
    axes[1].legend(ncol=2, loc="upper left")
    axes[1].grid(True)


def plot_likelihood(idata: az.InferenceData, cutoff=-100):
    draws = idata.posterior.sizes["draw"]
    chains = idata.posterior.sizes["chain"]
    likelihoods = np.clip(idata["sample_stats"]["likelihood"], cutoff, None)
    prior = np.clip(idata["sample_stats"]["prior"], cutoff, None)
    posterior = np.clip(idata["sample_stats"]["posterior"], cutoff, None)
    datasets = [likelihoods, prior, posterior]
    labels = ["log-likelihood", "log-prior", "log-posterior"]
    x_axis = np.arange(0, draws)

    figs = []

    for dataset, label in zip(datasets, labels):
        fig_progression, axes_progression = plt.subplots(2, 1, figsize=(16, 9))
        fig_progression.suptitle(f"Vývoj {label} v čase (hodnoty pod {cutoff} oříznuty)")
        axes_progression[0].set_xlabel("Iterace v chainu")
        axes_progression[0].set_ylabel(f"{label}")
        for chain in np.arange(0, chains):
            axes_progression[0].plot(x_axis, dataset[chain, :], label=f"Chain {chain}")

        mean = np.mean(dataset, axis=0)
        median = np.median(dataset, axis=0)
        min = np.min(dataset, axis=0)
        axes_progression[0].legend(ncol=2, loc="lower right")
        axes_progression[0].grid(True)

        axes_progression[1].set_xlabel("Iterace v chainu")
        axes_progression[1].set_ylabel(f"")
        axes_progression[1].plot(x_axis, mean, label=f"Průměrná {label}")
        axes_progression[1].plot(x_axis, median, label=f"Medián {label}")
        axes_progression[1].plot(x_axis, min, label=f"Minimum {label}")
        axes_progression[1].legend(ncol=2, loc="lower right")
        axes_progression[1].grid(True)

        figs += [fig_progression]

        fig_hist, axes_hist = plt.subplots(figsize=(16, 9))
        fig_hist.suptitle(f"Histogram {label} (hodnoty pod {cutoff} oříznuty)")
        axes_hist.set_xlabel(f"{label}")
        axes_hist.set_ylabel("Počet")
        axes_hist.hist(dataset.values.flatten(), bins=100)

        figs += [fig_hist]

    return figs

def plot_pair_against_nn(idata: az.InferenceData, config_path=None, nn_path=None):
    # setup observed data
    observed_data = idata["posterior_predictive"]

    flattened_observed = [observed_data[var].values.reshape(-1, 1)
                  for var in observed_data.data_vars]
    
    merged_observed = np.concatenate(flattened_observed, axis=1)

    time_points = idata["sample_stats"].attrs["times"]

    # setup posterior data
    posterior_data = idata["posterior"]

    flattened_posterior = [posterior_data[var].values.reshape(-1, 1)
                  for var in posterior_data.data_vars]
    
    merged_posterior = np.concatenate(flattened_posterior, axis=1)

    # setup likelihood data
    likelihood_data = idata["sample_stats"]["likelihood"].values.reshape(-1, 1)

    # setup flow and tinyda wrappers for parameter conversion
    if config_path is None:
        config_path = os.path.join(ROOT_DIR, "tests", "simulation", "templates", "test_workdir11")

    observe_path = os.path.join(ROOT_DIR, "tests", "measured_data")
    wrapper_new = Wrapper(config_path)
    wrapper_new.set_observe_path(observe_path)
    wrapper = TinyDAFlowWrapper(wrapper_new)
    target_observe = idata["sample_stats"].attrs["observed"]

    # setup NN wrapper

    if nn_path is None:
        nn_path = os.path.join(ROOT_DIR, "src", "bp_simunek", "samplers", "surrogates", "model_TSX_large.pth")

    assert os.path.exists(nn_path), f"NN model not found at {nn_path}"

    nn_wrapper = NNwrapper(nn_path, boreholes=["H1"])

    nn_error_data = np.empty((0, 1))
    flow_error_data = np.empty((0, 1))
    nn_error_data_piecewise = np.empty((0, 1))

    for posterior, observed in zip(merged_posterior, merged_observed):

        # transform params for the NN
        new_params = wrapper.transform_params(posterior)
        old_perms = wrapper.new_to_old_model(*new_params[2:])
        old_params = np.concatenate([new_params[0:4], old_perms])

        # get observe from NN
        print(old_params)
        nn_wrapper.set_parameters(old_params)
        nn_observed = nn_wrapper.get_observations()

        # compute norm between NN observe and real observe
        norm = np.linalg.norm(nn_observed - observed)
        nn_error_data = np.vstack((nn_error_data, norm))

        # compute norm between flow observe and real observe
        norm = np.linalg.norm(observed - target_observe)
        flow_error_data = np.vstack((flow_error_data, norm))

        # compute norm between NN observe and real in individual time points
        norm = nn_observed - observed
        nn_error_data_piecewise = np.concatenate((nn_error_data_piecewise.flatten(), norm))


    fig_pair, ax_pair = plt.subplots(len(new_params) - 1, len(new_params) - 1, figsize=(16, 9))
    
    cmap = plt.cm.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=nn_error_data.min(), vmax=nn_error_data.max())
    
    # clip out observe fails
    nn_error_data = nn_error_data.clip(0, 1000)

    az.plot_pair(
        idata,
        kind="scatter",
        scatter_kwargs={"c": nn_error_data, "cmap": cmap},
        colorbar=True,
        ax=ax_pair)

    fig_hist, ax_hist = plt.subplots(figsize=(16, 9))
    _, bins, patches = ax_hist.hist(nn_error_data, bins=100)

    for patch, binn in zip(patches, bins[:-1]):
        patch.set_facecolor(cmap(norm(binn)))  # Color each bin based on its value

    # data for best match between NN and full model
    best_match_idx = np.argmin(nn_error_data)

    best_match_observe = merged_observed[best_match_idx]

    best_match_posterior = merged_posterior[best_match_idx]
    new_params = wrapper.transform_params(best_match_posterior)
    old_perms = wrapper.new_to_old_model(*new_params[2:])
    old_params = np.concatenate([new_params[0:4], old_perms])
    nn_wrapper.set_parameters(old_params)
    best_match_nn = nn_wrapper.get_observations()

    fig_pressure, ax_pressure = plt.subplots(figsize=(16, 9))

    ax_pressure.set_title("Porovnání NN a naměřených dat")
    ax_pressure.set_xlabel("Čas [den]")
    ax_pressure.set_ylabel("Tlaková výška [m]")
    ax_pressure.grid(True)

    ax_pressure.plot(target_observe, color="black", label="Naměřená data")
    ax_pressure.plot(best_match_observe, color="red", label=f"Flow výstup pro nejlepší shodu s NN (l^2 = {nn_error_data[best_match_idx]})")
    ax_pressure.plot(best_match_nn, color="blue", label=f"NN výstup pro nejlepší shodu s flowem (l^2 = {nn_error_data[best_match_idx]})")

    # data for best match between full model and real data, with corresponding nn data
    best_fit_idx = np.argmin(flow_error_data)

    best_fit_observe = merged_observed[best_fit_idx]

    best_fit_posterior = merged_posterior[best_fit_idx]
    new_params = wrapper.transform_params(best_fit_posterior)
    old_perms = wrapper.new_to_old_model(*new_params[2:])
    old_params = np.concatenate([new_params[0:4], old_perms])
    nn_wrapper.set_parameters(old_params)
    best_fit_nn = nn_wrapper.get_observations()

    ax_pressure.plot(best_fit_observe, color="green", label=f"Flow výstup pro nejlepší shodu s naměřenými daty (l^2 = {flow_error_data[best_fit_idx]})")
    ax_pressure.plot(best_fit_nn, color="orange", label=f"NN výstup pro nejlepší shodu s naměřenými daty (l^2 = {np.linalg.norm(best_fit_nn - target_observe)})")

    ax_pressure.legend()

    # 2d hist to plot occurences of NN error at every time point
    piecewise_clip = 400
    nn_error_data_piecewise = np.clip(nn_error_data_piecewise, -piecewise_clip, piecewise_clip)
    hist2d_x = np.tile(time_points, idata["posterior"].sizes["draw"] * idata["posterior"].sizes["chain"])
    fig_hist2d, ax_hist2d = plt.subplots(figsize=(16, 9))
    hist2d_handle = ax_hist2d.hist2d(hist2d_x, nn_error_data_piecewise, bins=[time_points, 80], cmap="viridis_r", cmin=1e-10)
    fig_hist2d.colorbar(hist2d_handle[3])
    ax_hist2d.set_title(f"Histogram 2D - rozložení normy mezi NN a naměřenými daty v jednolivých časových bodech (hodnoty nad {piecewise_clip})")
    ax_hist2d.set_xlabel("Čas [den]")
    ax_hist2d.set_ylabel("Norma mezi NN a naměřenými daty (L^2)")
    ax_hist2d.grid(True)


    # scater plot for relation of NN error to model error
    fig_scatter, ax_scatter = plt.subplots(figsize=(16, 9))
    # flow error data is loglike
    # refactor according to norm = sqrt(-2 * loglike * noise_cov)
    likelihood_clip = 1000
    likelihood_data = np.sqrt(-2 * likelihood_data * 50)
    likelihood_data = np.clip(likelihood_data, a_min=0, a_max=likelihood_clip)
    ax_scatter.scatter(nn_error_data, likelihood_data)
    ax_scatter.set_title(f"Závislost chyby NN vůči plnému modelu na chybě plného modelu vůči naměřeným datům, hodnoty nad {likelihood_clip} oříznuty")
    ax_scatter.set_xlabel("Norma mezi NN a naměřenými daty (L^2)")
    ax_scatter.set_ylabel("Norma mezi plným modelem a naměřenými daty (L^2)")
    ax_scatter.grid(True)

    return fig_pair, fig_hist, fig_pressure, fig_hist2d, fig_scatter


def generate_all_flow_plots(idata: az.InferenceData, folder):
    az.plot_pair(idata, kind="kde")
    save_plot("pair_plot.pdf", folder_path=folder)
    az.plot_trace(idata)
    plt.tight_layout()
    save_plot("trace_plot.pdf", folder_path=folder)
    likelihood_plots = plot_likelihood(idata, cutoff=-1000)
    save_plots_pdf_pages("likelihood_plot.pdf", folder_path=folder, figs=likelihood_plots)

    corr_plot, stats_plots = data_window_plots(idata, 100)
    save_plot("corr_progression_plot.pdf", folder_path=folder, fig=corr_plot)
    save_plots_pdf_pages("stats_progression_plot.pdf", folder_path=folder, figs=stats_plots)

    prior_means = []
    prior_stds = []
    for param in idata["posterior"].data_vars:
        mean = idata["posterior"][param].attrs["prior_mean"]
        std = idata["posterior"][param].attrs["prior_std"]
        prior_means += [mean]
        prior_stds += [std]

    exp = idata["sample_stats"].attrs["observed"]

    axes = az.plot_posterior(idata, grid=[4, 2])
    for x, axrow in enumerate(axes):
        axrow_len = len(axrow)
        for y, ax in enumerate(axrow):
            idx = x * axrow_len + y
            if idx >= len(idata["posterior"]):
                continue
            if idx % axrow_len == 0:
                ax.set_ylabel("Hustota pravděpodobnosti", fontsize=15)
            if idx // axrow_len == 3:
                ax.set_xlabel("Hodnota parametru", fontsize=15)
            mean = prior_means[idx]
            std = prior_stds[idx]
            xvals = np.linspace(mean - 3 * std, mean + 3 * std, 100)
            yvals = sps.norm.pdf(xvals, mean, std)
            posterior = ax.lines[0]
            prior, = ax.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")
            if idx == 0:
                ax.legend([prior, posterior], ["Původní odhad", "Výsledek inverze"], fontsize=15, loc="upper left")
            plt.tight_layout()
    save_plot("posterior_plot.pdf", folder_path=folder)

    with open(os.path.join(folder, "summary.txt"), "w+") as file:
        accepted, rejected = compute_accepted(idata)
        summary = str(az.summary(idata))
        summary += f"\n\n{accepted} accepted\n{rejected} rejected\n{accepted / (accepted + rejected)} acceptance rate"
        file.writelines(summary)

    plot_pressures(idata, exp)
    save_plot("pressure_plot.pdf", folder_path=folder)


def compute_accepted(idata):
    variables =  list(idata["posterior"])
    accepted = 0
    rejected = 0
    for chain in idata["posterior"][variables[0]]:
        last_sample = chain[0]
        for sample in chain:
            if sample != last_sample:
                last_sample = sample
                accepted += 1
            else:
                rejected += 1
        
    return accepted, rejected

if __name__ == "__main__":
    #idata_name = "20x300.idata"
    #folder_path = os.path.join(ROOT_DIR, "output", "test11")
    folder_path = os.path.join(ROOT_DIR, "data", "dataset16", "bruh")
    idata_name = "20x300.idata"
    idata = read_idata_from_file(idata_name, folder_path)
    print(idata["sample_stats"])
    print(idata["sample_stats"].attrs)
    generate_all_flow_plots(idata, folder_path)
    #figs = plot_pair_against_nn(idata, None)
    #save_plots_pdf_pages("pair_plot_nn.pdf", figs, folder_path=folder_path)
    #likelihood_plots = plot_likelihood(idata, -4000)
    #save_plots_pdf_pages("likelihood_plot.pdf", folder_path=folder_path, figs=likelihood_plots)
    #csv_folder = os.path.join(ROOT_DIR, "data", "dataset10")
    #csv_name = "long_model_evaluations.txt"
    #csv_data = load_csv(csv_folder, csv_name)
    #csv_idata = idata_from_observe_times(csv_data, idata)
    #az.plot_pair(csv_idata)
    #save_plot("long_evaluations_pair_plot.pdf", folder_path=folder_path)
    #az.plot_posterior(csv_idata)
    #save_plot("long_evaluations_posterior_plot.pdf", folder_path=folder_path)