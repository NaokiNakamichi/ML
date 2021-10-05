import numpy as np

import pandas as pd
import datetime
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    args = sys.argv

    now = datetime.datetime.now()
    hoge = np.arange(9).reshape(3,3)
    df = pd.DataFrame(hoge, columns=[1, 2, 3])
    df.to_csv("GD/exp/simulation/remote_exp/remote_save_result", index=False)
