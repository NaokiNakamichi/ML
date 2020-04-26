'''Driver script with mproc workers handling distinct algos.'''

## External modules.
import multiprocessing as mproc
import numpy as np
import os

## Internal modules.
from algo_setup import parse_algo
from itin_setup import parse_itin
from mml.data import DataArray
from mml.models.linreg import LinReg_Ridge
from mml.models.quadratic import Quadratic
from mml.utils import makedir_safe, write_expinfo
from mml.utils.linalg import array_meanstd, trim_2d
from perf_setup import parse_perf
from task_setup import parse_task


###############################################################################


## Task-specific clerical prep.

task_name = "POC"

print("Starting task:", task_name)

task = parse_task(task_name)
subtask_names = task.paras.keys()

for subtask_name in subtask_names:
    
    print("Starting sub-task:", subtask_name)
    
    ## Basic sub-task details.
    subtask = task.paras[subtask_name]
    num_trials = subtask["num_trials"]
    n = subtask["n"]["etc"]
    d = subtask["d"]["etc"]
    init_range = subtask["init_range"]["etc"]
    dist_level = subtask["dist_level"]["etc"]
    num_trials = subtask["num_trials"]
    num_conds = subtask["num_conds"]
    rg = subtask["rg"]

    ## All methods are given a common starting point.
    w_star = np.ones(d).reshape((d,1))
    w_init = np.copy(w_star)
    w_init += rg.uniform(low=-init_range,
                         high=init_range, size=d).reshape((d,1))
    
    ## Data setup is somewhat specialized here.
    cov_X = np.eye(d) # covariance matrix of the inputs.
    _var_noise, _mean_noise, _gen = subtask["get_gen"](level=dist_level,
                                                       nval=n)
    var_noise = _var_noise[subtask_name]
    mean_noise = _mean_noise[subtask_name]
    gen_epsilon = _gen[subtask_name]
    
    ## Prepare a directory to write the results.
    towrite = os.path.join("results", task_name, subtask_name)
    makedir_safe(towrite)
        
    ## Prepare the method itinerary.
    itin = parse_itin(task_name)
    method_keys = itin.methods.keys()
        
    ## Print the experiment information, and write it to disk.
    write_expinfo(task_name=task_name,
                  subtask_name=subtask_name,
                  details=subtask,
                  itin=itin)
    
    ## Prepare for performance evaluation.
    perf_fn = parse_perf(task_name)
    perf_names = perf_fn(model=None, algo=None, data=None,
                         model_oracle=None, w_star=None)["names"]
    num_metrics = len(perf_names)

    ## Define a worker for this task.
    def worker(mth_name):
        '''
        This routine does all the work, after being passed an algorithm name.
        '''
        
        data = DataArray()
        
        mth_paras = itin.methods[mth_name]
        algo_name = mth_paras["algo_name"]
        model_name = mth_paras["model_name"]
        mth_paras.update(subtask)
        max_records = subtask["max_records"]

        print("Running method:", mth_name)
            
        ## Performance metric preparation.
        perf_shape = (max_records, num_metrics, num_trials)
        perf_array = np.zeros(perf_shape, dtype=np.float32)
        perf_array.fill(np.nan)
        
        for tri in range(num_trials):
            
            ## Initialize models (fixed, for now).
            model_oracle = Quadratic(w_star=w_star, A=cov_X,
                                     b=np.sqrt(var_noise))
            model = LinReg_Ridge()
            
            ## Generate new data (with *centered* noise).
            X = rg.normal(loc=0.0, scale=1.0, size=(n,d))
            epsilon = gen_epsilon() - mean_noise
            y = model(X=X, w=w_star) + epsilon
            data.init_tr(X=X, y=y)
            
            algo = parse_algo(algo_name=algo_name,
                              w_init=w_init,
                              paras=mth_paras)
            
            ## Run the algorithm to completion, recording performance.
            cntr = 0
            for onestep in algo:
                
                if cntr > max_records:
                    break
                
                ## Special update condition for oracle-type procedure.
                if algo_name == "GD_Optim":
                    algo.update(model=model_oracle, data=data)
                else:
                    algo.update(model=model, data=data)
                
                ## Performance recording as needed.
                if algo.torecord:
                    
                    perf_vals = perf_fn(model=model,
                                        algo=algo,
                                        data=data,
                                        model_oracle=model_oracle,
                                        w_star=w_star)["perf"]
                    perf_array[cntr,:,tri] = perf_vals
                    cntr += 1
                    algo.torecord = False
                    print(
                        "Status ({}): tri = {}, cntr = {}".format(mth_name,
                                                                  tri,
                                                                  cntr)
                    )
            
            ## Finally, if cntr isn't already maxed out, be sure
            ## to record performance at the final step.
            if cntr < max_records:
                perf_vals = perf_fn(model=model,
                                    algo=algo,
                                    data=data,
                                    model_oracle=model_oracle,
                                    w_star=w_star)["perf"]
                perf_array[cntr,:,tri] = perf_vals
            
        ## Having run over all trials, compute statistics.
        ##  note: ndarray shape is (max_records, num_metrics).
        perf_ave, perf_sd = array_meanstd(array=perf_array,
                                          axis=2,
                                          dtype=np.float64)
        
        ## Trim the unfilled results (if needed).
        perf_ave = trim_2d(array=perf_ave)
        perf_sd = trim_2d(array=perf_sd)

        ## Write to disk.
        fname = mth_name
        np.savetxt(fname=os.path.join(towrite, fname+".ave"),
                   X=perf_ave, fmt="%.7e", delimiter=",")
        np.savetxt(fname=os.path.join(towrite, fname+".sd"),
                   X=perf_sd, fmt="%.7e", delimiter=",")
        
        return None
                
    ## END of worker definition.

    ## Start mproc procedure.

    if __name__ == "__main__":
    
        cpu_count = mproc.cpu_count()
        print("Our machine has", cpu_count, "CPUs.")
        print("Of these,", len(os.sched_getaffinity(0)), "are available.")
        
        # Put all processors to work (at an upper limit).
        mypool = mproc.Pool(cpu_count)
        
        # Pass the "worker" the appropriate info.
        mypool.map(func=worker, iterable=method_keys)
        
        # Memory management.
        mypool.close() # important for stopping memory leaks.
        mypool.join() # wait for all workers to exit.
        
    ## End mproc procedure.


###############################################################################


