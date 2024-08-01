import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib as mpl
mpl.style.use("default")
mpl.style.use("seaborn")

def plot_results(data_list, x_label, y_label, title, filename, minY=None, maxY=None, time_mode=False, mini_mode=True):
    plt.figure(figsize=(8, 6), dpi=100)
    fontsize = 18

    colors = ["tab:purple", "tab:orange", "tab:green", "tab:red", "tab:blue", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    xmax = 0
    for i, data in enumerate(data_list):
        # add horizontal line for the maximum value
        if mini_mode:
            th = np.min(data['y_mean'])
            info = f" (min={th:.2f})"
        else:
            th = np.max(data['y_mean'])
            info = f" (max={th:.2f})"
        plt.axhline(y=th, color=colors[i], linestyle='--')

        if time_mode:
            (_, caps, _) = plt.errorbar(data['x_time'], data['y_mean'], yerr=data['y_std'], label=data['label'] + info, capsize=5, fmt='o', color=colors[i], ls='--')
        else:
            (_, caps, _) = plt.errorbar(data['x'], data['y_mean'], yerr=data['y_std'], label=data['label'] + info, capsize=5, fmt='o', color=colors[i], ls='--')
        for cap in caps:
            cap.set_markeredgewidth(1)
    
        xmax = data['x'][-1] if data['x'][-1] > xmax else xmax
    
    if not time_mode:
        xmin = 5
        step = (xmax - xmin) // 5
        xticks_values = np.arange(xmin, xmax + 1, step)
        plt.xticks(xticks_values, fontsize=fontsize+2)
    else:
        plt.xticks(fontsize=fontsize+2)

    plt.xlabel(x_label, fontsize=fontsize+2)
    plt.ylabel(y_label, fontsize=fontsize+2)
    plt.yticks(fontsize=fontsize+2)
    plt.title(title, fontsize=fontsize + 4)
    plt.legend(fontsize=fontsize-2)
    plt.ylim(minY, maxY)
    plt.grid(True)

    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.show()
    return None

def plot_single_results(data, x_label, y_label, title, filename):
    plt.figure(figsize=(8, 5), dpi=100)
    fontsize = 15

    plt.errorbar(data['x'], data['y_mean'], yerr=data['y_std'], label=data['label'], capsize=5)
    plt.xlabel(x_label, fontsize=fontsize); plt.xticks(fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize); plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize + 2)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return None


if __name__ == '__main__':
    '''
    Example of usage:

    python plot_results.py --datasets "SAMVIQ Expert Audio" "SAMVIQ Expert Video" "SAMVIQ Expert AV" --filenames ./discriminability_npy/discriminability_AV_audio-wilcoxon.npy ./discriminability_npy/discriminability_AV_video-wilcoxon.npy ./discriminability_npy/discriminability_AV_AV-wilcoxon.npy --type_discri Wilcoxon --analysis_name "AV datasets"

    '''

    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('--datasets', nargs='+', help='List of dataset names')
    parser.add_argument('--filenames', nargs='+', help='List of filenames')
    parser.add_argument('--type_discri', default=None, help='Type of discriminability', choices=["Variance", "T-test", "Wilcoxon"])
    parser.add_argument('--analysis_name', default="", help='Name of the analysis')
    parser.add_argument('--type_of_particitants', default="observers", help='Type of participants: observers, subjects, assessors, naives assessors, experts assessors, etc.')
    parser.add_argument("--type_of_scores", default="MOS", help="Type of scores: MOS, DMOS, PD, JND, JOD")
    # value for type of discriminability: Variance, T-test, Wilcoxon, None
    args = parser.parse_args()

    dataset_names = args.datasets
    filenames = args.filenames
    type_discri = args.type_discri.lower().capitalize()
    analysis_name = args.analysis_name
    type_of_scores = args.type_of_scores
    type_of_particitants = args.type_of_particitants

    if type_discri is None or type_discri == "None" or type_discri == "":
        type_discri = None
    elif not type_discri in ["Variance", "T-test", "Wilcoxon"]:
        print("Invalid value for type of discriminability, it should be one of the following: Variance, T-test, Wilcoxon")
        quit()
    

    # load the results
    data_list = {"discriminability": [], "mean_ci": [], "rmse": []}
    for f in range(len(filenames)):
        filename = filenames[f]
        dataset_name = dataset_names[f]

        data = np.load(filename, allow_pickle=True).item()
        nb_obs = data['nb_obs']
        m_ratio = data['m_ratio']
        m_ci = data['m_ci']
        m_rmse = data['m_rmse']
        m_sos = data['m_sos']
        print(f"Dataset: {dataset_name}, m_ratio: {len(m_ratio[0])}, m_ci: {len(m_ci[0])}, m_rmse: {len(m_rmse[0])}, m_sos: {len(m_sos)}")
        try:
            time_count = data['time_count']
        except:
            time_count = None

        ci = 63
        low_ci = np.percentile(m_ratio, (100-ci)/2, axis=1)
        high_ci = np.percentile(m_ratio, 100-(100-ci)/2, axis=1)
        ci_discri = (high_ci - low_ci) / 2
        data_list["discriminability"].append({'x_time': time_count, 'x': nb_obs, 'y_mean': np.mean(m_ratio, axis=1), 'y_std': ci_discri, 'label': f"{dataset_name}"})

        low_ci = np.percentile(m_ci, (100-ci)/2, axis=1)
        high_ci = np.percentile(m_ci, 100-(100-ci)/2, axis=1)
        ci_mean = (high_ci - low_ci) / 2
        data_list["mean_ci"].append({'x_time': time_count, 'x': nb_obs, 'y_mean': np.mean(m_ci, axis=1), 'y_std': ci_mean, 'label': f"{dataset_name}"})

        low_ci = np.percentile(m_rmse, (100-ci)/2, axis=1)
        high_ci = np.percentile(m_rmse, 100-(100-ci)/2, axis=1)
        ci_rmse = (high_ci - low_ci) / 2
        data_list["rmse"].append({'x_time': time_count, 'x': nb_obs, 'y_mean': np.mean(m_rmse, axis=1), 'y_std': ci_rmse, 'label': f"{dataset_name}"})


    if len(filenames) == 1:
        tag_save = dataset_names[0].replace(" ", "_")
    else:
        tag_save = "multi"
        tag_save = analysis_name.replace(" ", "_") if analysis_name != "" else tag_save
    # replace special characters
    for char in "[](){}<>?/\\|;:.,!@#$%^&*~`":
        tag_save = tag_save.replace(char, "_")

    # plot the results discriminability in function of number of observers
    ylabel = f"{type_of_scores} Discriminability - {type_discri}" if type_discri is not None else f"{type_of_scores} Discriminability"
    filename_out = f"./discriminability_plots/discriminability_{tag_save}-{type_discri}.png" if type_discri is not None else f"./discriminability_plots/discriminability_{tag_save}.png"
    plot_results(data_list["discriminability"], f"Number of {type_of_particitants}", ylabel, f"Discriminability: {analysis_name}", filename_out, maxY=1, minY=0.35, mini_mode=False)

    # plot the results mean CI in function of number of observers
    plot_results(data_list["mean_ci"], f"Number of {type_of_particitants}", f"{type_of_scores} Mean CI", f"Mean CI: {analysis_name}", f"./discriminability_plots/mean_ci_{tag_save}.png", minY=0)

    # plot the results RMSE in function of number of observers
    plot_results(data_list["rmse"], f"Number of {type_of_particitants}", f"{type_of_scores} RMSE", f"RMSE: {analysis_name}", f"./discriminability_plots/rmse_{tag_save}.png", minY=0)

    if time_count is None:
        quit()
    
    # plot the results discriminability in function of subjective test duration
    ylabel = f"{type_of_scores} Discriminability - {type_discri}" if type_discri is not None else f"{type_of_scores} Discriminability"
    filename_out = f"./discriminability_plots/discriminability_{tag_save}-time-{type_discri}.png" if type_discri is not None else f"./discriminability_plots/discriminability_{tag_save}-time.png"
    plot_results(data_list["discriminability"], "Time in hours", ylabel, f"Discriminability: {analysis_name}", filename_out, maxY=1, minY=0.35, time_mode=True, mini_mode=False)

    # plot the results mean CI in function of subjective test duration
    plot_results(data_list["mean_ci"], "Time in hours", f"{type_of_scores} Mean CI", f"Mean CI: {analysis_name}", f"./discriminability_plots/mean_ci_{tag_save}-time.png", minY=0, time_mode=True)

    # plot the results RMSE in function of subjective test duration
    plot_results(data_list["rmse"], "Time in hours", f"{type_of_scores} RMSE", f"RMSE: {analysis_name}", f"./discriminability_plots/rmse_{tag_save}-time.png", minY=0, time_mode=True)


    quit()

