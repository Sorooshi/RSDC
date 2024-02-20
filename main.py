import os
import time
import argparse
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from types import SimpleNamespace
from gdcm.common.utils import load_a_dict, save_a_dict, print_the_evaluated_results, print_per_repeat_results
from gdcm.data.preprocess import preprocess_features, preprocess_adjacency

jnp.set_printoptions(suppress=True, precision=3, linewidth=120)

import warnings
warnings.filterwarnings('ignore')

configs = {
    "results_path": Path("/home/sshalileh/GDCM/Results"),
    "figures_path": Path("/home/sshalileh/GDCM/Figures"),
    "params_path": Path("/home/sshalileh/GDCM/Params"),
    "data_path": Path("/home/sshalileh/GDCM/Datasets"),
}
configs = SimpleNamespace(**configs)

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()


def args_parser(arguments):
    _run = arguments.run
    _tau = arguments.tau
    _mu_1 = arguments.mu_1
    _mu_2 = arguments.mu_2
    _mu_1_a = arguments.mu_1_a
    _mu_2_a = arguments.mu_2_a
    _n_init = arguments.n_init
    _pp = arguments.pp.lower()
    _verbose = arguments.verbose
    _max_iter = arguments.max_iter
    _init = arguments.init.lower()
    _step_size = arguments.step_size
    _batch_size = arguments.batch_size
    _range_len = arguments.range_len
    _p_value = arguments.p_value
    _n_repeats = arguments.n_repeats
    _n_clusters = arguments.n_clusters
    _update_rule = arguments.update_rule.lower()
    _data_name = arguments.data_name.lower()
    _centroids_idx = arguments.centroids_idx
    _algorithm_name = arguments.algorithm_name.lower()
    _rho = arguments.rho
    _xi = arguments.xi

    return _run, _tau, _mu_1, _mu_2, _mu_1_a, _mu_2_a, _n_init, _pp, _verbose, _max_iter, \
               _init, _step_size, _range_len, _p_value, _n_repeats, _n_clusters, \
               _update_rule, _data_name, _centroids_idx, _algorithm_name, _batch_size, _rho, _xi


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_name", type=str, default="IRIS",
        help="Dataset's name, e.g., IRIS, or Lawyers, or dd_fix_demo (in CSV format)."
    )

    parser.add_argument(
        "--algorithm_name", type=str, default="GDCM_F",
        help="None case sensitive name of an algorithm developed for "
             "a specific data structure."
             "  The three following  data structures are supported: "
             "      1) Features only (*_F), "
             "      2) Feature-rich Network (*_FN), and"
             "      3) Multi-Layer Feature-rich Networks (*_MLFN)."
             " Note: later a third section after the second underscore was added, that is, *_*_this: "
             " where this can be:"
             "  e := represent using embedding"
             "  r := represents the refine mechanism"
             "  er: refining plus embedding"
    )

    parser.add_argument(
        "--update_rule", type=str, default="ngdc",
        help="GDC update rule, at the moment, "
             "the three following up update methods are supported"
             "   1) vgdc: applies vanilla gdc algorithm (VGDC);"
             "   2) ngdc: applies gdc with Nestrov momentum algorithm (NGDC);"
             "   3) agdc: applies gdc with Adam algorithm (Adam GDC)."

    )

    parser.add_argument(
        "--run", type=int, default=0,
        help="Run the algorithm or load the saved"
             " parameters and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm-ms",
        help="Data preprocessing method:"
             " For features data: MinMax (mm), Z-Scoring(zsc), Range-standardization(rng), "
             " Robust Standardizer (rs), quantile transformation with Normal distribution as output (qsn),"
             "  quantile transformation with uniform distribution as output (qsu), are supported."
             " For networks data: uniform_shift (us), modularity_shift(ms) are supported"
             "In the case of feature-rich networks, the pre-processing methods should of the two data sources, "
             "should be separated by \"-\", and the method for features pre-processing should be always mentioned "
             "first; e.g., to apply min-max and uniform shift, one should pass \"mm-us.\"."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=5,
        help="Number of clusters."
    )

    parser.add_argument(
        "--verbose", type=int, default=1,
        help="An integer showing the level of verbosity, "
             "the higher the more verbose."
    )

    parser.add_argument(
        "--max_iter", type=int, default=10,
        help="An integer showing the maximum number of iterations "
             "(epochs in ANN terminology)"
    )

    parser.add_argument(
        "--step_size", type=float, default=1e-2,
        help="A float showing the step size or learning rate."
    )

    parser.add_argument(
        "--range_len", type=int, default=10,
        help="NOT USED!"
             "An Integer to form a two-element list, window, where the first and second elements, in respect,"
             " are the iteration numbers of the beginning and the end of the window to compute the average and"
             "the standard deviation of the gradient values and semi-positive definiteness of hessian matrix."
             "Since in our experiments, we could not find any meaningful pattern or stopping value for this parameter,"
             "thus we excluded from the methods' description in our paper."
             "However, for consistency issues we preserve it in our implementation and later version we will remove it."
    )

    parser.add_argument(
        "--init", type=str, default="user",
        help="One of the three possible type of seed initialization:"
             "1) random, 2)K-means++, 3)user. If it is set to user, "
             "the centroids_idx argument should be provided"
    )

    parser.add_argument(
        "--centroids_idx", type=list, default=None,
        help="If init argument is set to user, this item should be provided to determine"
             " the index of seeds for centroids initialization."
    )

    parser.add_argument(
        "--p_value", type=float, default=2.,
        help="A float showing the p_value in Minkowski distance metric."
             "If it is set to None, cosine distance metric will be applied."
    )

    parser.add_argument(
        "--tau", type=float, default=1e-1,
        help="A float, tolerance threshold of the ||gradients||, mu ± tau*sigma of bootstrap means."
    )

    parser.add_argument(
        "--n_repeats", type=int, default=10,
        help="Number of repeats of a data set or a specific distribution"
    )

    parser.add_argument(
        "--n_init", type=int, default=1,
        help="Number of repeats with different seed initialization to select "
             "the best results on the data set."
    )

    parser.add_argument(
        "--mu_1", type=float, default=45e-2,
        help="Exponential decay rate for the first moment estimates in Adam or"
             " decay rate for Nestrov GDC. "
             "Note: default for x-only data"
    )

    parser.add_argument(
        "--mu_2", type=float, default=95e-2,
        help="Exponential decay rate for the second moment estimates "
             "(squared gradients estimates) in Adam.  "
             "Note: default for x-only data"
    )

    parser.add_argument(
        "--mu_1_a", type=float, default=None,
        help="Exponential decay rate for the first moment estimates in Adam or"
             " decay rate for Nestrov GDC devoted to the network data."
    )

    parser.add_argument(
        "--mu_2_a", type=float, default=None,
        help="Exponential decay rate for the second moment estimates "
             "(squared gradients estimates) in Adam devoted to the network data."
    )

    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size."
    )

    parser.add_argument('--rho', type=float, default=1,
                        help='Feature coefficient during the clustering')

    parser.add_argument('--xi', type=float, default=1,
                        help='Networks coefficient during the clustering')

    # add an arguments with_noise data
    args = parser.parse_args()

    run, tau, mu_1, mu_2, mu_1_a, mu_2_a, n_init, pp, verbose, max_iter, init, \
        step_size, range_len, p_value, n_repeats, n_clusters, update_rule, \
        data_name, centroids_idx, algorithm_name, batch_size, rho, xi = args_parser(arguments=args)

    print(
        "Configuration: \n"
        f"  run: {run} \n"
        f"  algorithm: {algorithm_name} \n"
        f"  data name: {data_name} \n"
        f"  pre-processing: {pp} \n"
        f"  step_size: {step_size} \n"
        f"  tau: {tau} \n"
        f"  mu_1: {mu_1} \n"
        f"  mu_2: {mu_2} \n"
        f"  mu_1_a: {mu_1_a} \n"
        f"  mu_2_x: {mu_2_a} \n"
        f"  p-value: {p_value} \n"
        f"  update_rule: {update_rule} \n"
        f"  range_len: {range_len} \n"
        f"  init: {init} \n"
        f"  max_iter: {max_iter} \n"
        f"  batch_size: {batch_size} \n"
    )

    configs.run = run
    old_specifier= 1

    if "gdcm_fn" in algorithm_name:
        specifier = algorithm_name + \
                    ":" + data_name + \
                    "-rule:" + str(update_rule) + \
                    "-alpha:" + str(step_size) + \
                    "-rho:" + str(rho) + \
                    "-xi:" + str(xi) + \
                    "-p:" + str(p_value) + \
                    "-tau:" + str(tau) + \
                    "-n_iter:" + str(max_iter) + \
                    "-mu1:" + str(mu_1) + \
                    "-mu2:" + str(mu_2) + \
                    "-mu1a:" + str(mu_1_a) + \
                    "-mu2a:" + str(mu_2_a) + \
                    "-Bsize:" + str(batch_size) + init
    else:
        assert False, "Unsupported algorithm or data type."

    assert len(specifier) <= 250, f"File name's length equal to {len(specifier)}, is too long."

    configs.specifier = specifier
    configs.data_name = data_name
    configs.n_repeats = n_repeats

    # to add the repeat numbers to the data_name variable for synthetic data
    if "k=" in data_name or "v=" in data_name:
        synthetic_data = True
    else:
        synthetic_data = False

    if run == 1:
        from gdcm.algorithms.distances import Distances

        results = {}
        for repeat in range(1, configs.n_repeats + 1):

            repeat = str(repeat)
            results[repeat] = {}

            if "fn" in algorithm_name.lower().split("_"):
                from gdcm.data.load_data import FeaturesRichData

                print(
                    "clustering features-rich network data: " + data_name + " repeat=" + repeat, "\n"
                )
                if not mu_1_a:
                    print(
                        "Using identical exponential decay rate for the first moment estimates of both data sources"
                    )
                    mu_1_a = mu_1

                if not mu_2_a:
                    print(
                        "Using identical exponential decay rate for the second moment estimates of both data sources"
                    )
                    mu_2_a = mu_2

                if synthetic_data is True:
                    if "nc=" in data_name:
                        dire = "FN/synthetic/categorical"
                    elif "nq=" in data_name:
                        dire = "FN/synthetic/quantitative"
                    elif "nm=" in data_name:
                        dire = "FN/synthetic/mixed"
                    else:
                        assert False, "unsupported synthetic data type!"
                    dn = data_name + "_" + repeat
                else:
                    dire = "FN"
                    dn = data_name

                data_path = os.path.join(configs.data_path, dire)
                fnd = FeaturesRichData(name=dn, path=data_path)

                x, xn, a, y_true = fnd.get_dataset()
                results[repeat]['y_true'] = y_true

                pp_f = pp.split("-")[0]
                pp_n = pp.split("-")[-1]

                x = preprocess_features(x=x, pp=pp_f)
                a = preprocess_adjacency(p=a, pp=pp_n)
                if xn.shape[0] != 0:
                    xn = preprocess_features(x=xn, pp=pp_f)
                n_clusters = len(np.unique(y_true))

                # GDC without embedding or refinement (vanilla):
                if algorithm_name.split("_")[-1].lower() == "v":
                    from gdcm.algorithms.gradient_descent_clustering_methods_features_networks import GDCMfn

                    # instantiate
                    start = time.process_time()
                    gdcm = GDCMfn(
                        p=p_value,
                        tau=tau,
                        rho=rho,
                        xi=xi,
                        mu_1_x=mu_1,
                        mu_2_x=mu_2,
                        mu_1_a=mu_1_a,
                        mu_2_a=mu_2_a,
                        init=init,
                        n_init=n_init,
                        verbose=verbose,
                        batch_size=batch_size,
                        update_rule=update_rule,
                        max_iter=max_iter,
                        n_clusters=n_clusters,
                        step_size=step_size,
                        centroids_idx=centroids_idx,
                    )

                # Refinement only:
                elif algorithm_name.split("_")[-1].lower() == "r":
                    from gdcm.algorithms.bootstrapped_gradient_descent_clustering_methods_features_networks \
                        import BGDCMfn

                    # instantiate
                    start = time.process_time()
                    gdcm = BGDCMfn(
                        p=p_value,
                        tau=tau,
                        rho=rho,
                        xi=xi,
                        mu_1_x=mu_1,
                        mu_2_x=mu_2,
                        mu_1_a=mu_1_a,
                        mu_2_a=mu_2_a,
                        init=init,
                        n_init=n_init,
                        verbose=verbose,
                        batch_size=batch_size,
                        update_rule=update_rule,
                        max_iter=max_iter,
                        n_clusters=n_clusters,
                        step_size=step_size,
                        centroids_idx=centroids_idx,
                    )

                else:
                    print("Unsupported case!")
                    assert False, "Unsupported case!"

                if p_value == 0.:
                    distance_fn = Distances.cosine_fn
                elif p_value == -1.:
                    distance_fn = Distances.canberra_fn
                elif p_value == -2.:
                    distance_fn = Distances.scaled_dot_product_fn
                elif p_value == -3.:
                    distance_fn = Distances.sigmoid_of_minkowski_fn
                elif p_value == -4.:
                    distance_fn = Distances.sigmoid_of_normalized_minkowski_fn
                elif 1. <= p_value:
                    distance_fn = Distances.minkowski_fn
                else:
                    print("ill-defined distance function")

                print(
                    f"distance_fu: {distance_fn}"
                )

                y_pred = gdcm.fit(x=x, a=a, distance_fn=distance_fn, y=y_true)
                end = time.process_time()

                # save results and logs
                results[repeat]['y_pred'] = y_pred
                results[repeat]['time'] = end - start
                results[repeat]['inertia'] = gdcm.best_inertia
                results[repeat]['data_scatter'] = gdcm.data_scatter
                results[repeat]['centroids_idx'] = gdcm.best_centroids_idx
                results[repeat]['batches'] = gdcm.best_batches
                results[repeat]['history_ari'] = gdcm.best_history_ari
                results[repeat]['history_nmi'] = gdcm.best_history_nmi
                results[repeat]['history_inertia'] = gdcm.best_history_inertia
                results[repeat]['history_grads_x'] = gdcm.best_history_grads_x
                results[repeat]['history_grads_a'] = gdcm.best_history_grads_a
                results[repeat]['history_centroid_x'] = gdcm.best_history_centroid_x
                results[repeat]['history_centroid_a'] = gdcm.best_history_centroid_a
                results[repeat]['history_grads_x_magnitude'] = gdcm.best_history_grads_x_magnitude
                results[repeat]['history_grads_a_magnitude'] = gdcm.best_history_grads_a_magnitude
                configs.stop_type = gdcm.stop_type

            elif algorithm_name.split("_")[0].lower() == "mlfn":
                print(
                    "clustering multi_layer feature_rich networks data"
                )
                assert False, "to be completed ..."
                from gdcm.algorithms.relaxed_gradient_descent_clustering import RGDCmlfn
                data_path = os.path.join(configs.data_path, "MLFN")

        # save results dict and configs
        save_a_dict(
            a_dict=results, name=configs.specifier, save_path=configs.results_path
        )

        save_a_dict(
            a_dict=configs, name=configs.specifier, save_path=configs.params_path
        )

        print("configs \n", configs.specifier, "\n")

        print("stop type:", configs.stop_type, "\n")

        print_the_evaluated_results(results=results)

    elif run != 1:

        # load results dict and configs
        results = load_a_dict(
            name=configs.specifier, save_path=configs.results_path
        )

        configs = load_a_dict(
            name=configs.specifier, save_path=configs.params_path
        )

        print("configs \n", configs.specifier, "\n")
        print_per_repeat_results(results=results)
        print_the_evaluated_results(results=results)
