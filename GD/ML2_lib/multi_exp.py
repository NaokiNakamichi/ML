import multiprocessing as mproc
import numpy as np

import model_opt
from GD.ML2_lib import algo_GD, helper


def noise_multi(noise=None):
    w_init = np.array([3, 3])
    _t_max = 3000
    var = np.random.randint(1, 300, 1)[0]
    noise = helper.gauss
    f = model_opt.RosenBrock(noise=noise, var=var)
    algo = algo_GD.SGD(w_init=w_init, t_max=_t_max, a=0.0007)
    w_star = f.w_star

    last_w_store = []
    iqr_store = []
    for i in range(10000):
        for i in algo:
            algo.update(model=f)
        return algo.wstore

        iqr_store.append(helper.iqr(algo.noise_store))
        last_w_store.append(algo.w)


if __name__ == "__main__":
    # 自分のマシンでコアが４(macbook)or6(imac)or8(ssh server)
    cpu_count = mproc.cpu_count()
    mypool = mproc.Pool(cpu_count)

    mypool.map(func=noise_multi)

    # Memory management.
    mypool.close()  # important for stopping memory leaks.
    mypool.join()  # wait for all workers to exit.
