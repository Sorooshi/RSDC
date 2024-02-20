import os
import wandb
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


np.set_printoptions(suppress=True, precision=3)


def save_a_dict(a_dict, name, save_path, ):
    with open(os.path.join(save_path, name+".pickle"), "wb") as fp:
        pickle.dump(a_dict, fp)
    return None


def load_a_dict(name, save_path, ):
    with open(os.path.join(save_path, name + ".pickle"), "rb") as fp:
        a_dict = pickle.load(fp)
    return a_dict


def init_a_wandb(name, project, notes, group, tag, config):

    """ name := the within the project name, e.g., RF-reg-1
        project := the project name, e.g., Non-sequential Regressions
        notes := Description, e.g., Non-sequential Regressions Comparison for SuperOX
        group := name of experiment or the algorithm under consideration, e.g., RF-1
        config := model and training configuration
        tag := tag of an experiment, e.g. run number of same experiments to compute ave. and std.
    """

    run = wandb.init(name=name,
                     project=project,
                     notes=notes,
                     entity='sorooshi',
                     group=group,
                     tags=tag,
                     config=config,
                     )

    return run


def wandb_metrics(run, y_true, y_pred, y_pred_prob, learning_method):

    meape_errors = mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100)

    # to compute ROC_AUC
    try:
        y_true.shape[1]
        y_true_ = y_true
    except:
        enc = OneHotEncoder(sparse=False)
        y_true_ = y_true.reshape(-1, 1)
        y_true_ = enc.fit_transform(y_true_)

    if learning_method == "regression":

        run.log({
            "MAE": mae(y_true=y_true, y_pred=y_pred),
            "MRAE": mrae(y_true=y_true, y_pred=y_pred),
            "R^2-Score": metrics.r2_score(y_true, y_pred),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0),
            "RMSE": rmse(y_true=y_true, y_pred=y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
        })

    elif learning_method == "classification":

        run.log({
            "ARI": metrics.adjusted_rand_score(y_true, y_pred),
            "NMI": metrics.normalized_mutual_info_score(y_true, y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
            "Precision": metrics.precision_score(y_true, y_pred, average='weighted'),
            "Recall": metrics.recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": metrics.f1_score(y_true, y_pred, average='weighted'),
            "ROC AUC": metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),
            "Accuracy": metrics.accuracy_score(y_true, y_pred, ),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0)
        })

    # for future applications I separate cls and clu
    elif learning_method == "clustering":

        run.log({
            "ARI": metrics.adjusted_rand_score(y_true, y_pred),
            "NMI": metrics.normalized_mutual_info_score(y_true, y_pred),
            "JSD": jsd(y_true=y_true, y_pred=y_pred).mean(),
            "Precision": metrics.precision_score(y_true, y_pred, average='weighted'),
            "Recall": metrics.recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": metrics.f1_score(y_true, y_pred, average='weighted'),
            # "ROC AUC": metrics.roc_auc_score(y_true_, y_pred_prob, average='weighted', multi_class="ovr"),
            "Accuracy": metrics.accuracy_score(y_true, y_pred, ),
            "MEAPE-mu": meape_errors.mean(axis=0),
            "MEAPE-std": meape_errors.std(axis=0)
        })

    return run


def wandb_features_importance(run, values_features_importance,
                              name_important_features,
                              indices_important_features,
                              importance_method):
    counter = 0
    for i in range(len(indices_important_features)):
        if counter < 5:
            run.log({
                importance_method +
                "-" + name_important_features[indices_important_features[i]] +
                "-" + str(i + 1): values_features_importance[0][indices_important_features[i]],
            })
            counter += 1

    return run


def wandb_true_pred_plots(run, y_true, y_pred, path, specifier):

    t = np.arange(len(y_true))
    fig, ax = plt.subplots(1, figsize=(14, 7))
    ax.plot(t, y_true, lw=1.5, c='g', label="y_true", alpha=1.)
    ax.plot(t, y_pred, lw=2., c='m', label="y_pred", alpha=1.)

    ax.fill_between(t, y_pred + y_pred.std(),
                    y_pred - y_pred.std(),
                    facecolor='yellow',
                    alpha=.5,
                    label="Std",
                    )

    ax.legend(loc="best")
    r2 = metrics.r2_score(y_true, y_pred)

    plt.xlabel("Index")
    plt.ylabel("True/Pred Values")
    plt.legend(loc="best")

    plt.title(
        "Plots: target vs predicted value of " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Plots"))
    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(
            p, specifier + ".png"
        )
    )

    run.log(
        {"Plots: target vs predicted value of " + specifier + str(r2): ax}
    )

    return run


def wandb_true_pred_scatters(run, y_test, y_pred, path, specifier,):

    _ = plt.figure(figsize=(14, 7))

    plt.scatter(np.arange(len(y_test)), y_test,
                alpha=0.7, marker='+', label='True')

    plt.scatter(np.arange(len(y_pred)), y_pred,
                alpha=0.8, marker='o', label='Prediction')

    plt.xlabel("Index")
    plt.ylabel("True/Pred Values ")
    plt.legend(loc="best")

    plt.title(
        "Scatters: target vs predicted values of " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Scatters"))
    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(p, specifier + ".png")
    )

    run.log(
        {"Scatters: target vs predicted values of " + specifier: plt}
    )

    return run


def wandb_true_pred_histograms(run, y_test, y_pred, path, specifier):

    plt.figure(figsize=(15, 7))
    plt.subplot(131)
    n_bins = np.linspace(y_test.min()-5, y_test.max()+5, 50)

    plt.hist(y_test, color="g",
             bins=n_bins, label="y_true",
             histtype='step', alpha=.7,
             linewidth=2,
             )

    # n_bins = np.linspace(y_pred.min()-20, y_pred.max()+20, 50)

    plt.hist(y_pred, color="m",
             bins=n_bins, label="y_pred",
             histtype='step',
             alpha=1.,
             )

    _max = max(y_test.max(), y_pred.max()) + 20

    plt.xlim([-_max, _max])
    plt.xlabel("True and Pred. values ")
    plt.ylabel('Count')
    plt.legend(loc="best")

    plt.title(
        "Histograms: " + specifier
    )

    # subdirectory
    p = Path(os.path.join(path, "Histograms"))

    if not p.exists():
        p.mkdir()

    plt.savefig(
        os.path.join(
            p, specifier + ".png"
        )
    )

    run.log(
        {"Histograms: target vs predicted of " + specifier : plt}
    )

    plt.show()

    return run


def print_the_evaluated_results(results, ):

    """results: dict, containing result of each repeat."""

    ari, nmi, precision, recall, f1_score, accuracy, time, inertia, data_scatter = [], [], [], [], [], [], [], [], []

    for k, v in results.items():
        # for kk, vv in v.items():
        y_true = v["y_true"]
        y_pred = v["y_pred"]

        ari.append(metrics.adjusted_rand_score(y_true, y_pred))
        nmi.append(metrics.normalized_mutual_info_score(y_true, y_pred))
        precision.append(metrics.precision_score(y_true, y_pred, average='weighted'))
        recall.append(metrics.recall_score(y_true, y_pred, labels=np.unique(y_true), average='weighted'))
        f1_score.append(metrics.f1_score(y_true, y_pred, average='weighted'))
        accuracy.append(metrics.accuracy_score(y_true, y_pred, ))
        time.append(v['time'])
        inertia.append(v['inertia'])
        data_scatter.append(v['data_scatter'])

    ari = np.nan_to_num(np.asarray(ari))
    ari_ = np.sort(ari)[::-1][:9]
    nmi = np.nan_to_num(np.asarray(nmi))
    precision = np.nan_to_num(np.asarray(precision))
    recall = np.nan_to_num(np.asarray(recall))
    f1_score = np.nan_to_num(np.asarray(f1_score))
    accuracy = np.nan_to_num((np.asarray(accuracy)))
    time = np.nan_to_num((np.asarray(time)))
    inertia = np.nan_to_num((np.asarray(inertia)))
    data_scatter = np.nan_to_num((np.asarray(data_scatter)))

    ari_ave = np.mean(ari, axis=0)
    ari_std = np.std(ari, axis=0)

    ari_ave_ = np.mean(ari_, axis=0)
    ari_std_ = np.std(ari_, axis=0)


    nmi_ave = np.mean(nmi, axis=0)
    nmi_std = np.std(nmi, axis=0)

    precision_ave = np.mean(precision, axis=0)
    precision_std = np.std(precision, axis=0)

    recall_ave = np.mean(recall, axis=0)
    recall_std = np.std(recall, axis=0)

    f1_score_ave = np.mean(f1_score, axis=0)
    f1_score_std = np.std(f1_score, axis=0)

    acc_ave = np.mean(accuracy, axis=0)
    acc_std = np.std(accuracy, axis=0)

    time_ave = np.mean(time, axis=0)
    time_std = np.std(time, axis=0)

    inertia_ave = np.mean(inertia, axis=0)
    inertia_std = np.std(inertia, axis=0)

    data_scatter_ave = np.mean(data_scatter, axis=0)
    data_scatter_std = np.std(data_scatter, axis=0)

    print(
        f" ari  \t "
        f"  nmi \t \t "
        f" inertia \t "
        f" time  \t "
        f" precision \t "
        f" recall \t "
        f" f1 score \t"
        f" accuracy \t "
        f" data scatter \t"
        f" top10 ari  \n "
        f" Ave ± std \t "
        f" Ave ± std \t "
        f" Ave ± std \t "
        f" Ave ± std \t "
        f" Ave ± std \t "
        f" Ave ± std \t "
        f" Ave ± std \t"
        f" Ave ± std \t "
        f" Ave ± std \t ",
        f" Ave ± std \t"
    )

    print(
        f"{ari_ave:.3f} ± {ari_std:.3f} \t"
        f"{nmi_ave:.3f} ± {nmi_std:.3f} \t"
        f"{inertia_ave:.3f} ± {inertia_std:.3f} \t"
        f"{time_ave:.3f} ± {time_std:.3f} \t"
        f"{precision_ave:.3f} ± {precision_std:.3f} \t"
        f"{recall_ave:.3f} ± {recall_std:.3f} \t"
        f"{f1_score_ave:.3f} ± {f1_score_std:.3f} \t"
        f"{acc_ave:.3f} ± {acc_std:.3f} \t"
        f"{data_scatter_ave:.3f} ± {data_scatter_std:.3f} \t"
        f"{ari_ave_:.3f} ± {ari_std_:.3f} \t"
        f"\n"
    )

    return None

def print_per_repeat_results(results, ):

    """results: dict, containing result of each repeat."""

    for k, v in results.items():
        # for kk, vv in v.items():
        y_true = v["y_true"]
        y_pred = v["y_pred"]
        print(
            f"repeat: {k} \t "
            f"ARI {metrics.adjusted_rand_score(y_true, y_pred):.3f} \t"
            f"NMI {metrics.normalized_mutual_info_score(y_true, y_pred):.3f} \t"
            f"ACC{metrics.accuracy_score(y_true, y_pred):.3f} \n"
        )

    return None





