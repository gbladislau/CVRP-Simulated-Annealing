import pandas as pd
import numpy as np
    
def params_mean_calculate():
    """
    Calcula a média para todas as instâncias na calibração de parâmentros
    """
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
                
    pd.DataFrame(vals_dict).to_csv("mean_report_params.csv", index=False)
    print(np.array(vals_list).min())
    print(vals_dict[np.array(vals_list).argmin()])
    
def gen_all_instances_report(reports=["a_b.csv", "f.csv"]):
    """Gera o report de todas as instâncias a partir dos reports individuais

    Args:
        reports (list, optional): lista com os arquivos da saída das instâncias
                                  Defaults to ["a_b.csv", "f.csv"].
    """
    df = pd.DataFrame()
    for report in reports:
        df = pd.concat([df,pd.read_csv(report)], ignore_index=True)
    print(df)
    
    df_saida = pd.DataFrame(columns=["Instância", "f_otima", "f_MH_min", "f_MH_med", "tempo_min", "tempo_med", "gap_min", "gap_med"])
    for instance in sorted(set(df.name)):
        f_med = 0.0
        f_min = np.inf
        f_optimal = 0.0
        t_med = 0.0
        t_min =  np.inf
        gap_min = np.inf
        gap_med = 0.0
        for instance_no in df[(instance == df.name)].instance_no.values:
            row = df[(instance == df.name) & (instance_no == df.instance_no)]
            # print(row)
            
            f_optimal = row["optimal_cost:"].values[0]
            
            f_med += row["best_cost"].values[0]
            t_med += row["time_for_best_sol"].values[0]
            
            t_min = row["time_for_best_sol"].values[0] if row["time_for_best_sol"].values[0] < t_min else t_min
            f_min = row["best_cost"].values[0]  if row["best_cost"].values[0]  < f_min else f_min
                
        f_med = f_med/5.0
        t_med = t_med/5.0
        
        gap_min = (f_min - f_optimal)/f_optimal
        gap_med = (f_med - f_optimal)/f_optimal

        df_saida = pd.concat([df_saida, pd.DataFrame([
            {
                "Instância": instance,
                "f_otima": f"{f_optimal}",
                "f_MH_min": f"{f_min}",
                "f_MH_med": f"{f_med}",
                "tempo_min": f"{t_min:.4}", 
                "tempo_med": f"{t_med:.4}", 
                "gap_min": f"{gap_min:.4}", 
                "gap_med": f"{gap_med:.4}",
            }]
            )], ignore_index=True)
        df_saida.to_csv("all_instances_report.csv", index=False)
    print(df_saida)
    
if __name__ == '__main__':
    # params_mean_calculate()
    # gen_all_instances_report()
    pass