import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'ABIDE': self.__eval_abide_regression,
            }
        else:
            self.metrics_dict = {
                'ABIDE': self.__eval_abide_classification,
            }

    def __eval_abide_classification(self, y_pred, y_true,y_pred_probabilities):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2   
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_probabilities = y_pred_probabilities[:, 1]
        y_pred_3 =  y_pred 
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_3).ravel()
        Sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        Specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_value = roc_auc_score(y_true, y_pred_probabilities)

        eval_results = {
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4),
            "Sensitivity": round(Sensitivity, 4),
            "Specificity": round(Specificity, 4),
            "auc_value": round(auc_value, 4)
        }
        return eval_results

    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]