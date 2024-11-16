import copy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import recall_score, brier_score_loss, roc_auc_score, precision_score
import shap

# HOW TO USE: from helper_functions import model_eval, bootstrap_632

# Source: https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def roc_auc_ci(y_true, y_score, conf_level=0.95):
    """
    Calculate ROC-AUC confidence interval.
    Source: https://stackoverflow.com/a/53180614
    """
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_score)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - conf_level) / 2)

    ci = norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    ci[ci < 0] = 0

    return np.array(ci)


def sensitivity_ci(y_true, y_pred, conf_level=0.95):
    """
    Calculate sensitivity (recall) score confidence interval using normal approximation of a binomial distribution.
    """
    n = np.sum(y_true == 1)
    score = recall_score(y_true, y_pred)
    se = np.sqrt(score * (1 - score) / n)
    z = norm.ppf((1 + conf_level)/2)
    ci_lower = score - z * se
    ci_upper = score + z * se
    return np.array([ci_lower, ci_upper])


def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)


def specificity_ci(y_true, y_pred, conf_level=0.95):
    """
    Calculate specificity score confidence interval using normal approximation of a binomial distribution.
    """
    n = np.sum(y_true == 0)
    score = recall_score(y_true, y_pred, pos_label=0)
    se = np.sqrt(score * (1 - score) / n)
    z = norm.ppf((1 + conf_level)/2)
    ci_lower = score - z * se
    ci_upper = score + z * se
    return np.array([ci_lower, ci_upper])

def ppv_ci(y_true, y_pred, conf_level=0.95):
    """
    Calculate positive predictive value confidence interval using normal approximation of a binomial distribution.
    """
    n = np.sum(y_true == 0)
    score = precision_score(y_true, y_pred)
    se = np.sqrt(score * (1 - score) / n)
    z = norm.ppf((1 + conf_level)/2)
    ci_lower = score - z * se
    ci_upper = score + z * se
    return np.array([ci_lower, ci_upper])

def npv_ci(y_true, y_pred, conf_level=0.95):
    """
    Calculate negative predictive value confidence interval using normal approximation of a binomial distribution.
    """
    n = np.sum(y_true == 0)
    score = precision_score(y_true, y_pred, pos_label=0)
    se = np.sqrt(score * (1 - score) / n)
    z = norm.ppf((1 + conf_level)/2)
    ci_lower = score - z * se
    ci_upper = score + z * se
    return np.array([ci_lower, ci_upper])


def brier_ci(y_true, y_prob, conf_level=0.95, repeats=1000, seed=100):
    """
    Calculate the confidence interval of the Brier score by bootstrapping the test set probabilities.
    """
    rng = np.random.default_rng(seed)

    size = np.shape(y_true)[0]
    true = y_true.reset_index(drop=True)
    
    scores = np.empty(repeats)
    for n in range(repeats):
        idx = rng.integers(size, size=size)
        scores[n] = brier_score_loss(true[idx], y_prob[idx])

    ci = np.percentile(scores, [100*(1-conf_level)/2, 100*(1+conf_level)/2])

    return ci


def model_eval(model, X_test, y_true, ci=True):
    """
    Calculate roc_auc, Brier score, sensitivity, specificity, and confidence intervals for a fitted sklearn model.

    ## Parameters
        model: sklearn model

        X_test: array-like

        y_true: array-like
    
    ## Returns
        scores: dict
            Dictionary of the metrics and their confidence intervals.
    """
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    if ci:
        result = {
            'roc_auc':          roc_auc_score(y_true, prob),
            'roc_auc_ci':       roc_auc_ci(y_true, prob),

            'brier':            brier_score_loss(y_true, prob),
            'brier_ci':         brier_ci(y_true, prob),

            'sensitivity':      recall_score(y_true, pred),
            'sensitivity_ci':   sensitivity_ci(y_true, pred),
            
            'specificity':      specificity_score(y_true, pred),
            'specificity_ci':   specificity_ci(y_true, pred),

            'ppv':              precision_score(y_true, pred),
            'ppv_ci':           ppv_ci(y_true, pred),

            'npv':              precision_score(y_true, pred, pos_label=0),
            'npv_ci':           npv_ci(y_true, pred)
            }
    else:
        result = {
            'roc_auc':          roc_auc_score(y_true, prob),
            'brier':            brier_score_loss(y_true, prob),
            'sensitivity':      recall_score(y_true, pred),
            'specificity':      specificity_score(y_true, pred),
            'ppv':              precision_score(y_true, pred),
            'npv':              precision_score(y_true, pred, pos_label=0)
            }
    return result


def bootstrap_632(model, X, y, repeats=100, conf_level=0.95, seed=100):
    """
    Evaluates a classification algorithm using Efron's 0.632 bootstrapping method.
    Calculates roc_auc, Brier score, sensitivity, specificity, and confidence intervals.
    
    Example: `bootstrap_632(LogisticRegression(random_state=123), X, y)`

    ## Parameters
        model: sklearn model
            This does not need to be fitted. The bootstrapping process will make a copy of the model and will not alter the original.

        X: array-like

        y: array-like

        repeats: int, default=100
            Number of times we generate bootstrap samples and train the model.
        
        conf_level: float, default=0.95
            Confidence level for the confidence interval. Should be a float between 0 and 1.
        
        seed: int, default=100
            Random seed used to generate bootstrap samples. Note: this does not affect the model's random_state. 
            For reproducibility, the model's random_state must be specified separately.
    
    ## Returns
        scores: dict
            Dictionary of the metrics and their confidence intervals.
    """
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(model)

    valid_perf = {
        'roc_auc': np.empty(repeats),
        'brier': np.empty(repeats),
        'sensitivity': np.empty(repeats),
        'specificity': np.empty(repeats)
    }
    
    original_perf = {
        'roc_auc': np.empty(repeats),
        'brier': np.empty(repeats),
        'sensitivity': np.empty(repeats),
        'specificity': np.empty(repeats)
    }

    idx = np.arange(X.shape[0])
    for n in range(repeats):
        # Create bootstrap sample
        train_idx = rng.integers(X.shape[0], size=X.shape[0])
        valid_idx = np.setdiff1d(idx, train_idx)
        
        # Fit model on bootstrap training sample
        m.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Model performance on bootstrap out-of-bag validation sample
        prob_boot = m.predict_proba(X.iloc[valid_idx])[:, 1]
        pred_boot = m.predict(X.iloc[valid_idx])
        valid_perf['roc_auc'][n] = roc_auc_score(y.iloc[valid_idx], prob_boot)
        valid_perf['brier'][n] = brier_score_loss(y.iloc[valid_idx], prob_boot)
        valid_perf['sensitivity'][n] = recall_score(y.iloc[valid_idx], pred_boot)
        valid_perf['specificity'][n] = recall_score(y.iloc[valid_idx], pred_boot, pos_label=0)

        # Model performance on original data
        prob_orig = m.predict_proba(X)[:, 1]
        pred_orig = m.predict(X)
        original_perf['roc_auc'][n] = roc_auc_score(y, prob_orig)
        original_perf['brier'][n] = brier_score_loss(y, prob_orig)
        original_perf['sensitivity'][n] = recall_score(y, pred_orig)
        original_perf['specificity'][n] = recall_score(y, pred_orig, pos_label=0)

    estimate = {
        'roc_auc': 0.368 * original_perf['roc_auc'] + 0.632 * valid_perf['roc_auc'],
        'brier': 0.368 * original_perf['brier'] + 0.632 * valid_perf['brier'],
        'sensitivity': 0.368 * original_perf['sensitivity'] + 0.632 * valid_perf['sensitivity'],
        'specificity': 0.368 * original_perf['specificity'] + 0.632 * valid_perf['specificity']
    }

    percentiles = [100*(1-conf_level)/2, 100*(1+conf_level)/2]
    results = {
        'roc_auc': np.mean(estimate['roc_auc']),
        'roc_auc_ci': np.percentile(estimate['roc_auc'], percentiles),
        'brier': np.mean(estimate['brier']),
        'brier_ci': np.percentile(estimate['brier'], percentiles),
        'sensitivity': np.mean(estimate['sensitivity']),
        'sensitivity_ci': np.percentile(estimate['sensitivity'], percentiles),
        'specificity': np.mean(estimate['specificity']),
        'specificity_ci': np.percentile(estimate['specificity'], percentiles)
    }

    return results


def feature_rank(shap_vals: shap.Explanation, name: str = None) -> pd.Series:
    """
    Extract the feature importance rankings from SHAP output.

    ## Parameters
        shap_vals: shap.Explanation
            Output from a SHAP explainer.
        
        name: str, optional
            Name for the output Series.
    
    ## Returns
        ranks: pd.Series
    """
    if shap_vals.values.ndim == 3:
        # Get the output for the last dimension, which should correspond to the shap values for the output 1
        mean_abs = pd.Series(np.absolute(shap_vals.values[:,:,-1]).mean(axis=0), index=shap_vals.feature_names, name=name)
    else:
        mean_abs = pd.Series(np.absolute(shap_vals.values).mean(axis=0), index=shap_vals.feature_names, name=name)
    ranks = mean_abs.sort_values(ascending=False).rank(ascending=False)
    return ranks
