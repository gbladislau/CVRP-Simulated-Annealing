from simulated_annealing import SimulatedAnnealing
from cvrp import CVRP
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import os

def search_best_parameters(
        root: str = "./instances",
        instance_paths: list[str] = ["A/A-n32-k5", "A/A-n46-k7", "A/A-n80-k10","B/B-n50-k8", "B/B-n78-k10"]
    ):
    
    param_temps_initial = [100, 1000, 5000]
    params_cooling_func = ["exp", "lin", "log"]
    params_cooling_rate = [0.5555, 0.7777, 0.9999]
    df = pd.DataFrame()
    for instance in tqdm(instance_paths):
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
    
def run_all_instances_serial(root: str = "./instances", instance_folders = ["A","B","F"]):
    df = pd.DataFrame()
    for folder in instance_folders:
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
        
def process_instance(folder, instance, root="./instances"):
    instance_path = os.path.join(root, folder, instance)
    instance_obj = CVRP(instance_path)  
    reports = []
    for i in range(5):
        print(instance, i)
        sa = SimulatedAnnealing(
            initial_temp=100,
            cooling_func="log",
            cooling_rate=0.5555,
            time_limit=300.0
        )
        sa.optimize(instance_obj)
        report = sa.return_report()
        # Add additional info
        report["name"] = instance
        report["instance_no"] = i
        report["optimal_cost:"] = instance_obj.optimal_value
        reports.append(report)
    return reports

def run_instances_parallel(root="./instances", instance_folders=["A", "B", "F"], processes=4):
    tasks = []
    for folder in instance_folders:
        folder_path = os.path.join(root, folder)
        instance_files = [f for f in os.listdir(folder_path) if f.endswith(".vrp")]
        for instance in instance_files:
            tasks.append((folder, instance))
    
    all_reports = []

    with Pool(processes=processes) as pool:
        # Roda as instâncias e pega tos resultados
        results_list = list(tqdm(pool.starmap(process_instance, tasks), total=len(tasks),
                                 desc="Rodando Instâncias"))
    
    # Junta os resultados
    for reports in results_list:
        all_reports.extend(reports)
    
    df = pd.DataFrame(all_reports)
    df.to_csv("all_instances_run.csv", index=False)
    
def gen_chart(root="./instances/A", instance="A-n32-k5.vrp"):
    """Gera o gráfico de valor da função objetiva por iteração

    Args:
        root (str, optional): pasta da instância. Defaults to "./instances/A".
        instance (str, optional): nome da instância. Defaults to "A-n32-k5".
    """
    instance_obj = CVRP(os.path.join(root, instance))  
    
    sa = SimulatedAnnealing(
        initial_temp=100,
        cooling_func="log",
        cooling_rate=0.5555,
        time_limit=300
    )
    chart_info = sa.optimize(instance_obj, gen_chart=True)
        
        # Find max values
    max_x = chart_info["iterations"][-1]
    min_y = min(chart_info["f_obj_val"])
    min_y_i = np.array(chart_info["f_obj_val"]).argmin()

    # Downsample Y-axis labels (show values only at each 1000)
    plt.figure(figsize=[12,6])
    plt.title("A-n32-k5 Custo por Iteração - Simulated Annealing")
    plt.plot(chart_info["iterations"][::1000],chart_info["f_obj_val"][::1000], linestyle = "dashdot", color='g')
    plt.axhline(y = instance_obj.optimal_value, color = 'r', linestyle = '--') 
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.scatter([min_y_i], [min_y], color="r", zorder=3)
    plt.text(min_y_i, min_y-50, f"({min_y_i}, {min_y})", fontsize=8, verticalalignment='bottom', horizontalalignment="center", color="blue")
    plt.savefig("example_1000.svg")
    plt.close()
            
    plt.figure(figsize=[12,6])
    plt.title("A-n32-k5 Custo por Iteração - Simulated Annealing")
    plt.plot(chart_info["iterations"][::10000],chart_info["f_obj_val"][::10000], linestyle = "dashdot", color='g')
    plt.axhline(y = instance_obj.optimal_value, color = 'r', linestyle = '--') 
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.scatter([min_y_i], [min_y], color="r", zorder=3)
    plt.text(min_y_i, min_y-50, f"({min_y_i}, {min_y})", fontsize=8, verticalalignment='bottom', horizontalalignment="center", color="blue")
    plt.savefig("example_10000.svg")       
    plt.close()
    
    plt.figure(figsize=[12,6])
    plt.title("A-n32-k5 Custo por Iteração - Simulated Annealing")
    plt.plot(chart_info["iterations"],chart_info["f_obj_val"], linestyle = "dashdot", color='g')
    plt.axhline(y = instance_obj.optimal_value, color = 'r', linestyle = '--') 
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.scatter([min_y_i], [min_y], color="r", zorder=3)
    plt.text(min_y_i, min_y-50, f"({min_y_i}, {min_y})", fontsize=8, verticalalignment='bottom', horizontalalignment="center", color="blue")
    plt.savefig("example_no_downsample.svg")         
            

if __name__ == "__main__":
    # search_best_parameters()
    # run_all_instances()
    # run_instances_parallel()
    gen_chart()
    
    