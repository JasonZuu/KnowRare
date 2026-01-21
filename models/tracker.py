import numpy as np


class PerformanceTracker:
    def __init__(self, early_stop_epochs:int, metric="auprc",
                 direction="maximize"):
        self.metric = metric
        self.direction = direction
        if direction == "maximize":
            self.best_metrics = {metric: -np.inf}
        elif direction == "minimize":
            self.best_metrics = {metric: np.inf}
        else:
            raise ValueError("Invalid direction")
        self.best_model_state_dict = None
        self.early_stop_epochs = early_stop_epochs
        self.no_update_epochs = 0
        self.early_stop = False

    def update(self, metric_dict, model_state_dict)->bool:
        if self.direction == "maximize" and metric_dict[self.metric] > self.best_metrics[self.metric]:
            self.best_metrics = metric_dict
            self.best_model_state_dict = model_state_dict
            self.no_update_epochs = 0
        elif self.direction == "minimize" and metric_dict[self.metric] < self.best_metrics[self.metric]:
            self.best_metrics = metric_dict
            self.best_model_state_dict = model_state_dict
            self.no_update_epochs = 0
        else:
            self.no_update_epochs += 1

        # check if early stop
        if self.no_update_epochs >= self.early_stop_epochs:
            self.early_stop = True
        else:
            self.early_stop = False
        return self.early_stop

    def export_best_model_state_dict(self):
        return self.best_model_state_dict

    def export_best_metric_dict(self):
        return self.best_metrics
