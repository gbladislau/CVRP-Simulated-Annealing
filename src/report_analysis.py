import pandas as pd
import numpy as np
df = pd.read_csv("/home/braga/work/CVRP/best_params_search_results_updated.csv")

vals_list = []
vals_dict = []
for initial_temp in set(df.initial_temp):
    for cooling_func in set(df.cooling_func):
        for cooling_rate in set(df.cooling_rate):
            vals = df[
                    (df.initial_temp == initial_temp) &
                    (df.cooling_func == cooling_func) &
                    (df.cooling_rate == cooling_rate)
                    ].best_cost
            med_val = vals.mean()
            vals_list.append(med_val)
            vals_dict.append(
                {
                    "temp": initial_temp, 
                    "cooling_func": cooling_func,
                    "cooling_rate":cooling_rate,
                    "mean":float(med_val)
                }
            ) 
            
print()
print(np.array(vals_list).min())
print(vals_dict[np.array(vals_list).argmin()])