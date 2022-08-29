from sklearn import metrics
import numpy as np

class performance_measures:
    def __init__(self, n):
        self.n_classes = n
        self.confusion_matrix = np.empty([])
        self.Recall = np.empty(n)
        self.Precision = np.empty(n)
        self.Specificity = np.empty(n)
        self.Acc = np.empty(n)
        self.F_measure = np.empty(n)

        self.Overall_Acc = 0.0


# Compute the performance measures
# Using sensivity (recall), specificity (precision) and accuracy 
# for each class: (normal, pneumonia, covid-19)
def compute_performance_measures(predictions, gt_labels):
    n_classes = 3
    pf_ms = performance_measures(n_classes)

    # Confussion matrix
    conf_mat = metrics.confusion_matrix(gt_labels, predictions, labels=[0,1,2])
    conf_mat = conf_mat.astype(float)
    pf_ms.confusion_matrix = conf_mat

    # Overall Acc
    pf_ms.Overall_Acc = metrics.accuracy_score(gt_labels, predictions)

    # normal: 0, pneumonia: 1, covid-19: 2
    for i in range(0, n_classes):
        TP = conf_mat[i,i]
        FP = sum(conf_mat[:,i]) - conf_mat[i,i]
        TN = sum(sum(conf_mat)) - sum(conf_mat[i,:]) - sum(conf_mat[:,i]) + conf_mat[i,i]
        FN = sum(conf_mat[i,:]) - conf_mat[i,i]
        
        pf_ms.Recall[i]       = TP / (TP + FN)
        pf_ms.Precision[i]    = TP / (TP + FP)
        pf_ms.Specificity[i]  = TN / (TN + FP); # 1-FPR
        pf_ms.Acc[i]       = (TP + TN) / (TP + TN + FP + FN)

        if TP == 0:
            pf_ms.F_measure[i] = 0.0
        else:
            pf_ms.F_measure[i] = 2 * (pf_ms.Precision[i] * pf_ms.Recall[i] )/ (pf_ms.Precision[i] + pf_ms.Recall[i])

    return pf_ms