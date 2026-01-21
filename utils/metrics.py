import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve
from scipy import stats


metric_rules = {"auroc": "max",
                "auprc": "max",
                "f1": "max",
                "precision": "max",
                "recall": "max",
                "specificity": "max",}


def get_optimal_err_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # eliminate infinite values
    valid_idx = np.isfinite(thresholds)
    fpr = fpr[valid_idx]
    tpr = tpr[valid_idx]
    thresholds = thresholds[valid_idx]

    # calculate EER distance
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def get_optimal_f1_cutoff(y_true, y_score):
    # Get precision, recall, and thresholds for PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Calculate F1 score for each threshold to find the optimal balance
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero

    # Get the index of the maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

    return optimal_threshold


def calculate_p_value(mean1, ci1, n1, mean2, ci2, n2):
    """
    Calculate the p-value for the two-tailed test for two independent samples with the given means,
    confidence intervals, and sample sizes.

    Args:
    - mean1, mean2: Means of the two samples.
    - ci1, ci2: Confidence intervals of the two samples (assuming 95% confidence level).
    - n1, n2: Sample sizes of the two samples.

    Returns:
    - p-value for the two-tailed test.
    """
    # Assuming 95% confidence level, t value for 2-tailed test
    df = n1 + n2 - 2
    t_value = stats.t.ppf(1 - 0.025, df)

    # Calculating Standard Error from Confidence Interval
    se1 = ci1 / t_value
    se2 = ci2 / t_value

    # Calculating the combined standard error
    sed = np.sqrt(se1 ** 2.0 + se2 ** 2.0)

    # Calculating the t-statistic
    t_stat = (mean1 - mean2) / sed

    # Calculating the p-value
    p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
    return p


def calculate_improvement(x0, x1, negative=False):
    if negative:
        return (x0 - x1) / x0 * 100
    else:
        return (x1 - x0) / x0 * 100
