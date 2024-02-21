import numpy as np

# basic invference engine
def infer(model: any , data: np.array):
    # for each item in data, predict the label
    return model.model.predict(data)