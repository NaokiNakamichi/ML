import numpy as np

import pandas as pd
import datetime
import os, sys

if __name__ == "__main__":
    args = sys.argv

    now = datetime.datetime.now()
    hoge = np.arange(9).reshape(3,3)
    df = pd.DataFrame(hoge, columns=[1, 2, 3])
    df.to_csv("remote_save_result/test.csv", index=False)
