import numpy as np
import json
import matplotlib.pyplot as plt
import random
from scipy import stats
import multiprocessing
from tqdm import tqdm
from scipy.optimize import curve_fit
from time import time, sleep
import argparse

def get_sampled_scores(scores, choice):
    '''
    Function to sample the scores of a subset of observers
    :param scores: list of all the scores for a stimulus/PVS
    :param choice: list of the observers to consider
    '''
    #return np.array(scores)[choice]
    length = len(scores)
    return np.array([scores[i%length] for i in choice]) # sample and ensure no out of range

###############################################################
###############################################################
"""
Functions to get the discriminability in a dataset
Discriminability is the proportion of pairs of stimuli for which the difference in the scores is significant.
The significance is influenced by the distribution of the scores (mean and variance)

get_discriminability_ttest: use a t-test to compare the scores of two stimuli
get_discriminability_wilcoxon: use a wilcoxon test to compare the scores of two stimuli
get_discriminability_variance: use the variance of the scores to compare the scores of two stimuli
"""

def get_discriminability_ttest(all_scores, choice, escape_thres=1, p_value=0.05):
    '''
    Function to compute the discriminability in a dataset using a t-test
    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    :param escape_thres: threshold to escape some pairs of stimulus/PVS and reduce the computation time
    :param p_value: p-value for the t-test
    '''
    cpt_total = 0 # total number of pairs of stimuli/PVS
    cpt_diff = 0 # number of pairs of stimuli/PVS considered as significantly different
    for i in tqdm(range(len(all_scores))): # for each pair of stimuli/PVS
        for j in range(i + 1, len(all_scores)):
            if random.random() > escape_thres: # random choice to reduce the computation time (in action only if escape_thres < 1)
                continue
            
            # get the scores for the two stimulus/PVS
            sc_i = get_sampled_scores(all_scores[i], choice) # scores of stimulus i
            sc_j = get_sampled_scores(all_scores[j], choice) # scores of stimulus j

            # compute the t-test
            w, p = stats.ttest_ind(sc_i, sc_j)

            # if the p-value is below the threshold, we consider the two stimulus/PVS as significantly different
            if p < p_value:
                cpt_diff += 1
            cpt_total += 1
    
    # compute the ratio of significantly different pairs
    ratio = cpt_diff / cpt_total
    return ratio

def get_discriminability_wilcoxon(all_scores, choice, escape_thres=1, p_value=0.05):
    '''
    Function to compute the discriminability in a dataset using a Wilcoxon test
    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    :param escape_thres: threshold to escape some pairs of stimulus/PVS and reduce the computation time
    :param p_value: p-value for the Wilcoxon test
    '''
    cpt_total = 0
    cpt_diff = 0
    for i in tqdm(range(len(all_scores))): # for each pair of stimulus/PVS
        for j in range(i + 1, len(all_scores)):
            if random.random() > escape_thres: # random choice to reduce the computation time (in action only if escape_thres < 1)
                continue
            
            # get the scores for the two stimulus/PVS
            sc_i = get_sampled_scores(all_scores[i], choice)
            sc_j = get_sampled_scores(all_scores[j], choice)
            diff = sc_i - sc_j
            
            # if the two stimulus/PVS have the same scores, we skip the pair (no need to compute the Wilcoxon test)
            if np.sum(diff) == 0:
                cpt_total += 1
                continue
            
            # compute the Wilcoxon test
            w, p = stats.wilcoxon(diff)

            # if the p-value is below the threshold, we consider the two stimulus/PVS as significantly different
            if p < p_value:
                cpt_diff += 1
            cpt_total += 1
    
    # compute the ratio of significantly different pairs
    ratio = cpt_diff / cpt_total
    return ratio

def get_discriminability_variance(all_scores, choice, escape_thres=1, z=1.96):
    '''
    Function to compute the discriminability in a dataset using the variance
    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    :param escape_thres: threshold to escape some pairs of stimulus/PVS and reduce the computation time
    :param z: z-score for the confidence interval computation
    '''
    cpt_total = 0
    cpt_diff = 0
    for i in tqdm(range(len(all_scores))): # for each pair of stimulus/PVS
        for j in range(i + 1, len(all_scores)):
            if random.random() > escape_thres: # random choice to reduce the computation time (in action only if escape_thres < 1)
                continue
            
            # get the scores for the two stimulus/PVS
            sc_i = get_sampled_scores(all_scores[i], choice)
            sc_j = get_sampled_scores(all_scores[j], choice)
            
            # compute the mean and variance of the two stimulus/PVS
            m_i, m_j = np.mean(sc_i), np.mean(sc_j)
            std_i, std_j = np.std(sc_i), np.std(sc_j)

            # compute the variance of the difference
            li, lj = len(sc_i), len(sc_j)
            sig_ij = (std_i**2/li + std_j**2/lj) ** 0.5

            # if the difference is above the threshold, we consider the two stimulus/PVS as significantly different
            if z * sig_ij < abs(m_i - m_j):
                cpt_diff += 1
            cpt_total += 1
    
    # compute the ratio of significantly different pairs
    ratio = cpt_diff / cpt_total
    return ratio

###############################################################
###############################################################



###############################################################
###############################################################
'''
Functions to get the mean confidence interval, the RMSE to the ground truth and the SOS score

get_mean_ci: get the mean confidence interval of the scores
get_sos_score: get the SOS score of the dataset
get_RMSE_to_GT: get the RMSE to the ground truth of the dataset
'''

def get_mean_ci(all_scores, choice, escape_thres=1, z=1.96):
    '''
    Function to compute the mean confidence interval around the MOS of a dataset
    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    :param z: z-score for the confidence interval
    '''
    if type(all_scores[0]) == list: 
        # case where we have different number of observers for each video
        # this can happen when we have remove some outlier's scores
        mos_cis = []
        for i in range(len(all_scores)): # for each stimulus/PVS
            # get the scores for the stimulus/PVS
            all_scores_i = get_sampled_scores(all_scores[i], choice)
            # compute the confidence interval
            mos_cis.append(z * np.std(all_scores_i) / np.sqrt(len(all_scores_i)))
    else:
        # case where we have the same number of observers for each stimuli (we can use numpy)
        # get the scores for all the stimulus/PVS
        all_scores = all_scores[:, choice]
        # compute the confidence interval
        mos_cis = z * np.std(all_scores, axis=1) / np.sqrt(len(all_scores[0]))
    
    # compute the mean confidence interval pf all the stimulus/PVS in the dataset
    return np.mean(mos_cis)

def func(x, a):
    '''
    Function to fit the variance of the scores with the MOS using a polynomial function in SOS method on a scale of 1 to 5 (ACR/DSIS scale for example)
    :param x: MOS
    :param a: parameter of the polynomial function to optimize
    '''
    return a * (x - 1) * (5 - x) # quadratic function to fit the SOS score

def get_sos_score(all_scores, choice, escape_thres=1):
    '''
    Function to compute the SOS score of a dataset
    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    '''
    if type(all_scores[0]) == list: # case where we have different number of observers for each video
        variance = []; mos = []
        for i in range(len(all_scores)): # for each stimulus/PVS
            # get the scores for the stimulus/PVS
            all_scores_i = get_sampled_scores(all_scores[i], choice)
            # compute the variance and the MOS
            variance.append(np.std(all_scores_i) ** 2)
            mos.append(np.mean(all_scores_i))
        variance = np.array(variance)
        mos = np.array(mos)
    else:
        # get the scores for all the stimulus/PVS
        all_scores = all_scores[:, choice]

        # compute the variance and the MOS
        variance = np.std(all_scores, axis=1) ** 2
        mos = np.mean(all_scores, axis=1) 

    # fit the variance with the MOS using a polynomial function
    popt, pcov = curve_fit(func, mos, variance)
    perr = np.sqrt(np.diag(pcov))
    pred = func(mos, *popt)

    # compute the mean squared error
    mse = np.mean((pred - variance)**2)

    # return the parameter of the polynomial function, the error and the mean squared error
    return [popt[0], perr[0], mse]

def get_RMSE_to_GT(all_scores, choice, escape_thres=1):
    '''
    Function to compute the RMSE between the estimated MOS from all the unique observers 
    of the subjective test and a subset of these observers (potential sampled with replacement)

    :param all_scores: list of all the scores (MOS, PD, etc.) for each stimulus/PVS
    :param choice: list of the observers to consider
    '''

    if type(all_scores[0]) == list: # case where we have different number of observers for each video
        mos_sub = []
        mos_gt = []
        for i in range(len(all_scores)): # for each stimulus/PVS
            # get the scores for the stimulus/PVS
            all_scores_i = get_sampled_scores(all_scores[i], choice)
            # compute the MOS for the subset of observers and the MOS for all the observers
            mos_sub.append(np.mean(all_scores_i))
            mos_gt.append(np.mean(all_scores[i]))
        mos_sub = np.array(mos_sub)
        mos_gt = np.array(mos_gt)
        rmse = np.sqrt(np.mean((mos_sub - mos_gt)**2))
    else:
        # get the scores for all the stimulus/PVS
        sub_scores = all_scores[:, choice]

        # compute the MOS for the subset of observers and the MOS for all the observers
        mos_sub = np.mean(sub_scores, axis=1)
        mos_gt = np.mean(all_scores, axis=1)
        rmse = np.sqrt(np.mean((mos_sub - mos_gt)**2))
    
    # return the RMSE
    return rmse

###############################################################
###############################################################


###############################################################
###############################################################
'''
Function to convert a dataset into discriminability, mean confidence interval, RMSE to the ground truth and SOS score
'''
def analysis_core(filename, dataset_name, get_discriminability, escape_thres=1, minimum_obs_count=6, \
                        maximum_obs_count=-1, n_sim=25, with_replacement=True, step_size=3, scaling=None):
    '''
    Function to load a dataset and compute the discriminability for different number of observers
    :param filename: path to the dataset
    :param dataset_name: name of the dataset
    :param get_discriminability: function to compute the discriminability
    :param minimum_obs_count: minimum number of observers to consider
    :param maximum_obs_count: maximal number of observers to consider
    :param n_sim: number of simulations to bootstrap the results
    :param with_replacement: boolean to sample the observers with replacement or not
    :param step_size: step size to increase the number of observers
    :param scaling: scaling method to apply to the scores to compare the rating scale of different datasets
    '''
    global CPU_CORE_COUNT

    # Load the SUREAL format dataset
    with open(filename) as f:
        dataj = json.load(f)

    # try to get the total time of the subjective test if in dataj
    if "total_duration" in dataj:
        base_time = dataj["total_duration"] / 3600
    else:
        base_time = 0
        print('\n\nNo "total_duration" in the dataset (prefered for analysis of discriminability in function of subjective test duration).\n\n')
        sleep(2)
    
    mini_obs_count = -1; maxi_obs_count = -1
    all_scores = [] # list to store the scores for each stimulus/PVS
    DEBUG_NB_OBS = []
    for i in range(len(dataj['dis_videos'])): # for each stimulus/PVS
        # get the scores for the stimulus/PVS
        
        # if dict
        if type(dataj['dis_videos'][i]['os']) == dict:
            scores_i = np.array(list(dataj['dis_videos'][i]['os'].values()))
        # if list
        elif type(dataj['dis_videos'][i]['os']) == list:
            scores_i = np.array(dataj['dis_videos'][i]['os'])

        # scaling the scores if needed
        if scaling == "SAMVIQ_0-100_to_DSIS_1-5":
            scores_i = scores_i / 25 + 1
        elif scaling == "DCR_0-10_TO_DSIS_1-5":
            scores_i = 4 * (scores_i - 0) / (10 - 0) + 1
        all_scores.append(list(scores_i))

        # get the minimum number of observers in the dataset
        if mini_obs_count == -1 or len(scores_i) < mini_obs_count:
            mini_obs_count = len(scores_i)
        if len(scores_i) > maxi_obs_count:
            maxi_obs_count = len(scores_i)
        DEBUG_NB_OBS.append(len(scores_i))

    #print(f"DEBUG NUMBER OBS: {np.unique(DEBUG_NB_OBS)} {maxi_obs_count}")
    
    # convert the scores to a numpy array
    all_scores = np.array(all_scores)
    print(f"Dataset {dataset_name} loaded with {len(all_scores)} stimuli and at least {mini_obs_count} observers per stimulus/PVS.")
    
    if maximum_obs_count != -1: # if specific maximum number of observers is given in argument
        mini_obs_count = maximum_obs_count
    
    # create the range of number of observers to consider during computation [minimum_obs_count, minimum_obs_count + step_size, ..., maximum_obs_count]
    nb_obs = [w for w in range(minimum_obs_count, mini_obs_count + 1, step_size)]
    # create the range of observers to sample from [0, 1, ..., maximum number of observers] (to sample randomly the observers)
    obs_range = [x for x in range(maxi_obs_count)] 

    if base_time == 0:
        time_count = None
    else:
        time_count = [base_time * w/nb_obs[-1] for w in nb_obs]
    
    m_ratio = [] # list to store the discriminability ratio per number of observers
    m_ci = [] # list to store the mean confidence interval per number of observers
    m_rmse = [] # list to store the RMSE to the ground truth per number of observers
    m_sos = [] # list to store the SOS score per number of observers
    for n in tqdm(nb_obs): # for each number of observers to consider in the discriminability computation

        # create the values to pass to the function to compute the discriminability
        if with_replacement:
            values = [(all_scores, random.choices(obs_range, k=n), escape_thres) for _ in range(n_sim)]
        else:
            values = [(all_scores, random.sample(obs_range, n), escape_thres) for _ in range(n_sim)]

        t = time()
        '''
        print("SOS")
        with multiprocessing.Pool(CPU_CORE_COUNT) as pool:
            sos_scores = pool.starmap(get_sos_score, values)
        m_sos.append(sos_scores)
        print("SOS Run in", time() - t, "secondes")
        t = time()
        '''

        with multiprocessing.Pool(CPU_CORE_COUNT) as pool:
            rmse_scores = pool.starmap(get_RMSE_to_GT, values)
        m_rmse.append(rmse_scores)
        #print("RMSE_GT Run in", time() - t, "secondes")
        t = time()

        with multiprocessing.Pool(CPU_CORE_COUNT) as pool:
            mean_ci_scores = pool.starmap(get_mean_ci, values)
        m_ci.append(mean_ci_scores)
        #print("MEAN CI Run in", time() - t, "secondes")
        t = time()

        with multiprocessing.Pool(CPU_CORE_COUNT) as pool:
            discri_scores = pool.starmap(get_discriminability, values)
        m_ratio.append(discri_scores)
        #print("Discri Run in", time() - t, "secondes")
    
    m_ratio = np.array(m_ratio)
    return nb_obs, m_ratio, m_ci, m_rmse, n_sim, m_sos, time_count

# find number of cpu cores
def get_cpu_count():
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 2

if __name__ == "__main__":

    CPU_CORE_COUNT = get_cpu_count()

    print(f"Running the analysis with {CPU_CORE_COUNT} CPU cores.")

    # argparser for n_sim, step_size, escape_thres, type of discriminability, and dataset filename
    parser = argparse.ArgumentParser(description='Compute the discriminability of a dataset.')
    parser.add_argument('--n_sim', type=int, default=5, help='Number of simulations to bootstrap the results.') 
    # 5-10 to get first estimation, 100 to 1000 to get precise estimation (100 can be enough for datasets with a relatively large set of stimuli)
    parser.add_argument('--step_size', type=int, default=3, help='Step size to increase the number of observers.') 
    # 1 to get a dense plot, 3-5 generally enough to report in a plot
    parser.add_argument('--escape_thres', type=float, default=1, help='Threshold to escape some pairs of stimulus/PVS and reduce the computation time.')
    # 1 to consider all the pairs, 0.1 to consider 10% of the pairs (faster computation for first estimation)
    parser.add_argument('--type', type=str, default="wilcoxon", help='Type of discriminability to compute (variance, ttest, wilcoxon).', choices=["variance", "ttest", "wilcoxon"])
    parser.add_argument('--filename', type=str, required=True, help='Path to the dataset in SUREAL format.')
    parser.add_argument('--out_filename', type=str, required=True, help='Path to save the results.')
    parser.add_argument('--scaling', required=False, type=str, default=None, help='Scaling method to apply to the scores to compare the rating scale of different datasets.', choices=["SAMVIQ_0-100_to_DSIS_1-5", "DCR_0-10_TO_DSIS_1-5"])

    parser.add_argument('--cpu_count', type=int, default=-1, help='Number of CPU cores to use for the computation.')
    parser.add_argument('--minimum_obs_count', type=int, default=-1, help='Minimum number of observers to consider.')
    parser.add_argument('--maximum_obs_count', type=int, default=-1, help='Maximal number of observers to consider.')
    args = parser.parse_args()

    '''
    Example of command to run the script:
    
    # Run on the 3 subsets of the 260 AV quality dataset (video, audio, and audio-video) from "Perceptual Evaluation on Audio-visual Dataset of 360 Content" DOI: 10.1109/ICMEW56448.2022.9859426

    python analyzer.py --n_sim 5 --step_size 3 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_trained/AV_video.json --out_filename ./discriminability_npy/discriminability_AV_video-wilcoxon.npy --scaling SAMVIQ_0-100_to_DSIS_1-5 --cpu_count -1

    python analyzer.py --n_sim 5 --step_size 3 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_trained/AV_audio.json --out_filename ./discriminability_npy/discriminability_AV_audio-wilcoxon.npy --scaling SAMVIQ_0-100_to_DSIS_1-5 --cpu_count -1
    
    python analyzer.py --n_sim 5 --step_size 3 --escape_thres 0.1 --type wilcoxon --filename ./datasets_json/360_AV_trained/AV_AV.json --out_filename ./discriminability_npy/discriminability_AV_AV-wilcoxon.npy --scaling SAMVIQ_0-100_to_DSIS_1-5 --cpu_count -1

    python plot_results.py --datasets "SAMVIQ Expert Audio" "SAMVIQ Expert Video" "SAMVIQ Expert AV" --filenames ./discriminability_npy/discriminability_AV_audio-wilcoxon.npy ./discriminability_npy/discriminability_AV_video-wilcoxon.npy ./discriminability_npy/discriminability_AV_AV-wilcoxon.npy --type_discri Wilcoxon --analysis_name "AV datasets" --type_of_scores MOS --type_of_particitants assessors

    

    # Run on the 3 subsets of the 360 AV quality dataset with naive and expert assessors from "Comparison of conditions for omnidirectional video with spatial audio in terms of subjective quality and impacts on objective metrics resolving power" DOI: 10.1109/ICASSP48485.2024.10448123

    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/DSIS_naives_video.json --out_filename ./discriminability_npy/discriminability_DSIS_naives_video-wilcoxon.npy --cpu_count -1

    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/ACR_naives_video_dmos_subset.json --out_filename ./discriminability_npy/discriminability_AV_ACR_naives_video_dmos-wilcoxon.npy --cpu_count -1
    
    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/SAMVIQ_experts_video_subset.json --out_filename ./discriminability_npy/discriminability_AV_SAMVIQ_experts_video_sub-wilcoxon.npy --scaling SAMVIQ_0-100_to_DSIS_1-5 --cpu_count -1


    python plot_results.py --datasets "DSIS Naives Video" "ACR--HR Naives Video" "SAMVIQ Expert Video" --filenames ./discriminability_npy/discriminability_DSIS_naives_video-wilcoxon.npy ./discriminability_npy/discriminability_AV_ACR_naives_video_dmos-wilcoxon.npy ./discriminability_npy/discriminability_AV_SAMVIQ_experts_video_sub-wilcoxon.npy --type_discri Wilcoxon --analysis_name "AV Video (Expert+Naive) datasets" --type_of_scores MOS --type_of_particitants assessors

    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/DSIS_naives_av_subset.json --out_filename ./discriminability_npy/discriminability_DSIS_naives_av-wilcoxon.npy --cpu_count -1

    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/ACR_naives_av_dmos_subset.json --out_filename ./discriminability_npy/discriminability_AV_ACR_naives_av_dmos-wilcoxon.npy --cpu_count -1
    
    python analyzer.py --n_sim 100 --step_size 1 --escape_thres 1 --type wilcoxon --filename ./datasets_json/360_AV_naives/SAMVIQ_experts_av_subset.json --out_filename ./discriminability_npy/discriminability_AV_SAMVIQ_experts_av_sub-wilcoxon.npy --scaling SAMVIQ_0-100_to_DSIS_1-5 --cpu_count -1


    python plot_results.py --datasets "DSIS Naives AV" "ACR--HR Naives AV" "SAMVIQ Expert AV" --filenames ./discriminability_npy/discriminability_DSIS_naives_av-wilcoxon.npy ./discriminability_npy/discriminability_AV_ACR_naives_av_dmos-wilcoxon.npy ./discriminability_npy/discriminability_AV_SAMVIQ_experts_av_sub-wilcoxon.npy --type_discri Wilcoxon --analysis_name "AV (Expert+Naive) datasets" --type_of_scores MOS --type_of_particitants assessors

    '''

    n_sim = args.n_sim
    step_size = args.step_size
    escape_thres = args.escape_thres
    filename = args.filename
    out_filename = args.out_filename
    type_discri = args.type
    scaling = args.scaling
    cpu_count = args.cpu_count
    minimum_obs_count = args.minimum_obs_count
    maximum_obs_count = args.maximum_obs_count

    if cpu_count != -1:
        CPU_CORE_COUNT = cpu_count

    if not scaling in ["SAMVIQ_0-100_to_DSIS_1-5", "DCR_0-10_TO_DSIS_1-5", None]:
        print("Unknown scaling method.")
        exit(1)

    dataset_name = filename.split("/")[-1].split(".")[0]
    
    if out_filename == "":
        out_filename = f"./discriminability_npy/discriminability_{dataset_name}-{type_discri}.npy"
    print(out_filename)


    # get the discriminability function to use and the minimum number of observers to consider
    if type_discri == "variance":
        get_discriminability = get_discriminability_variance
        mini = 3 if minimum_obs_count == -1 else minimum_obs_count # minimum number of observers to consider
    elif type_discri == "ttest":
        get_discriminability = get_discriminability_ttest
        mini = 6 if minimum_obs_count == -1 else minimum_obs_count
    elif type_discri == "wilcoxon":
        get_discriminability = get_discriminability_wilcoxon
        mini = 6 if minimum_obs_count == -1 else minimum_obs_count
    else:
        print("Unknown discriminability type.")
        exit(1)
    
    
    # compute the discriminability for the dataset
    nb_obs, m_ratio, m_ci, m_rmse, n_sim, m_sos, time_count = analysis_core(filename, dataset_name, get_discriminability, \
                            escape_thres=escape_thres, minimum_obs_count=mini, maximum_obs_count=maximum_obs_count, n_sim=n_sim, \
                            step_size=step_size, scaling=scaling)

    # save the results in a numpy file
    np.save(out_filename, {"time_count": time_count, "nb_obs": nb_obs, "m_ratio": m_ratio, "n_sim": n_sim, 
                            "m_ci": m_ci, "m_rmse": m_rmse, "m_sos": m_sos, "type_discriminability": type_discri, \
                            "scaling": scaling, "escape_thres": escape_thres, "minimum_obs_count": mini, \
                            "maximum_obs_count": maximum_obs_count})



