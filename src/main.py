from simulated_annealing import SimulatedAnnealing
from cvrp import CVRP
import pandas as pd
import os

def search_best_parameters(
        root: str = "./instances",
        instance_paths: list[str] = ["A/A-n32-k5", "A/A-n46-k7", "A/A-n80-k10","B/B-n50-k8", "B/B-n78-k10"]
    ):
    
    param_temps_initial = [100, 250, 500]
    params_alpha = [0.333, 0.555, 0.999]
    df = pd.DataFrame()
    for instance in instance_paths:
        for init_temp, cooling_rate in zip(param_temps_initial, params_alpha):
            instance_obj = CVRP(os.path.join(root, instance + '.vrp'))
            sa = SimulatedAnnealing(initial_temp=init_temp, alpha=cooling_rate, time_limit=30.0)
            print(f"instance: {instance}, initial_temp: {init_temp}, cooling_rate:{cooling_rate}")
            sa.optimize(instance_obj)
            report = sa.return_report()
            df = pd.concat([df, pd.DataFrame({"name":instance ,** report, "optimal_cost:":instance_obj.optimal_value})])
            print(df)
    
    
if __name__ == "__main__":
    search_best_parameters()