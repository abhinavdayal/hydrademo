# cross entropy loss
def cross_entropy_loss(y_true: int, y_pred: int, num_classes=3):
    import numpy as np
    from sklearn.metrics import log_loss
    y_true = np.eye(num_classes)[y_true] # on-hot encoding
    y_pred = np.eye(num_classes)[y_pred] if len(y_pred.shape)==1 else y_pred# on-hot encoding
    return log_loss(y_true, y_pred)