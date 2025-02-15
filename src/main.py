from simulated_annealing import SimulatedAnnealing
from cvrp import CVRP
import pandas as pd
from tqdm import tqdm
import os

def search_best_parameters(
        root: str = "./instances",
        instance_paths: list[str] = ["A/A-n32-k5", "A/A-n46-k7", "A/A-n80-k10","B/B-n50-k8", "B/B-n78-k10"]
    ):
    
    param_temps_initial = [100, 1000, 5000]
    params_cooling_func = ["exp", "lin", "log"]
    params_cooling_rate = [0.5555, 0.7777, 0.9999]
    df = pd.DataFrame()
    for instance in instance_paths:
        for init_temp in param_temps_initial:
            for cooling_func in params_cooling_func:
                for cooling_rate in params_cooling_rate:
                    instance_obj = CVRP(os.path.join(root, instance + '.vrp'))
                    sa = SimulatedAnnealing(
                        initial_temp=init_temp,
                        cooling_func=cooling_func,
                        cooling_rate=cooling_rate,
                        time_limit=300
                        )
                    print(f"instance: {instance}, initial_temp: {init_temp}," 
                          f"cooling_func:{cooling_func}, cooling_rate:{cooling_rate}")
                    sa.optimize(instance_obj)
                    report = sa.return_report()
                    df = pd.concat([df, pd.DataFrame(
                        {
                            "name":instance,
                            "optimal_cost:":instance_obj.optimal_value,
                            "initial_temp":init_temp,
                            "cooling_func": cooling_func,
                            "cooling_rate": cooling_rate,
                            **report,
                        }
                        )],
                        ignore_index=True)
        print(df)
    df.to_csv("best_params_search_results_updated.csv", index=False)   
    
def run_all_instances(root: str = "./instances", instance_folders = ["A","B","F"]):
    for folder in instance_folders:
        df = pd.DataFrame()
        for instance in tqdm(list(filter(lambda x: True if x.endswith(".vrp") else False,
                               os.listdir(os.path.join(root, folder)))),
                             desc=f"Rodando instâncias de {folder}"):
            instance_obj = CVRP(os.path.join(root, folder, instance))
            for i in range(5):
                sa = SimulatedAnnealing(
                    initial_temp=100,
                    cooling_func="log",
                    cooling_rate=0.5555,
                    time_limit=300.0
                    )
                sa.optimize(instance_obj)
                report = sa.return_report()
                df = pd.concat([df, pd.DataFrame(
                            {
                                "name":instance,
                                "instance_no": i,
                                "optimal_cost:":instance_obj.optimal_value,
                                **report,
                            }
                            )],
                        ignore_index=True
                    )
            df.to_csv("all_instances_run.csv", index=False)
        
def run_single_instance(path):
    pass
    
if __name__ == "__main__":
    # search_best_parameters()
    run_all_instances()