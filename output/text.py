import numpy as np
def text_output(loss: float, filename: str):
    # save the loss to a file
    np.savetxt(filename, np.array([loss]))