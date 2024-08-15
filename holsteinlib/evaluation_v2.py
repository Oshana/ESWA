import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, class_labels):
    """
    Evaluates a model's performance by calculating class-wise metrics and overall metrics.

    Parameters:
    y_true (list or np.array): Ground truth labels.
    y_pred (list or np.array): Predicted labels by the model.
    class_labels (list): List of class labels.

    Returns:
    class_wise_metrics (dict): A dictionary containing accuracy, precision, recall, sensitivity, and specificity for each class.
    overall_report (str): A string report of overall precision, recall, f1-score for each class.
    additional_metrics (dict): A dictionary containing model accuracy, balanced accuracy, macro precision, and macro recall.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    class_wise_metrics = {cls: {} for cls in class_labels}
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate accuracy, precision, recall, sensitivity, specificity for each class
    for idx, cls in enumerate(class_labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        sensitivity = recall  # Sensitivity is the same as recall
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        class_wise_metrics[cls]['accuracy'] = accuracy
        class_wise_metrics[cls]['precision'] = precision
        class_wise_metrics[cls]['recall'] = recall
        class_wise_metrics[cls]['sensitivity'] = sensitivity
        class_wise_metrics[cls]['specificity'] = specificity
        class_wise_metrics[cls]['f1-score'] = f1

    # Generate overall classification report
    overall_report = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0)

    # Calculate additional metrics
    model_accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro',zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    additional_metrics = {
        'model_accuracy': model_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1-score': macro_f1
    }

    return class_wise_metrics, overall_report, additional_metrics, cm