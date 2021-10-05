import numpy as np

import pandas as pd
import datetime
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    args = sys.argv

    now = datetime.datetime.now()
    df = pd.DataFrame(np.array([4, 5, 6]).T, columns=[1, 2, 3])
    df.to_csv("remote_save_result/test.csv", index=False)
