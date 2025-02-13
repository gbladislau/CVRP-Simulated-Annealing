import random
from cvrp import CVRP
from time import time
import numpy as np
import pandas as pd

class SimulatedAnnealing:
    
    # valores iniciais
    start_temp: float
    alpha: float  # cooling rate (taxa de resfriamento 0-1)
    initial_solution: list[list[int]]
    
    # após otimizar são encontrados estes valores
    best_solution: list[list[int]]
    best_solution_cost: float
    start_time: float
    finish_time: float
    total_time_spent: float
    best_solution_time: float
    
    def __init__(self, initial_temp:float = 0.0,  alpha:float = 0.1,  time_limit:float = 0.0, iteration_limit:float=np.inf):
        self.start_temp = initial_temp
        self.alpha = alpha
        self.time_limit = float(time_limit)
        self.iteration_limit = iteration_limit
        
    
    def __next_temp(self, iteration: int) -> float:
        """Calcula a próxima temperatura usando o cooling schedule 
        logaritmo -> T_k = T0 - (alpha ** k) onde k é o numero da 
        iteracao e alpha a taxa de resfriamento de 0-1

        Args:
            iteration (int): numero da iteracao

        Returns:
            float: nova temp
        """
        return self.start_temp / np.log(1 + iteration)
        
    
    def optimize(self, instance: CVRP) -> dict:
        self.start_time = time()
        
        self.initial_solution = instance.gen_initial_sol()
        
        self.best_solution = self.initial_solution
        self.best_solution_cost = instance.calculate_cost(self.initial_solution)
        
        current_solution = self.best_solution
        current_s_cost = self.best_solution_cost
        
        iteration_n = 1
        time_diff = 0.0 
        self.best_solution_time = 0.0
        actual_temp = self.start_temp
        while time_diff < self.time_limit and iteration_n < self.iteration_limit:
            # busca local
            new_solution = instance.generate_new_solution(current_solution)
            new_cost = instance.calculate_cost(new_solution)
            cost_diff =  new_cost - current_s_cost
            #aceita solução melhor com menor custo
            if cost_diff < 0:
                current_s_cost = new_cost
                current_solution = new_solution
                # se for melhor que a melhor solução troca
                if new_cost < self.best_solution_cost:
                    self.best_solution_cost = new_cost
                    self.best_solution = new_solution
                    self.best_solution_time = time_diff
            # aceita a solucao pior aleatoriamente seguindo uma funcao em relacao a temperatura atual
            elif random.random() <= np.exp(((-cost_diff)/actual_temp)):
                current_s_cost = new_cost
                current_solution = new_solution
            
            actual_temp = self.__next_temp(iteration_n)
            iteration_n += 1
            actual_time_stamp = time()
            time_diff = actual_time_stamp - self.start_time
        self.finish_time = time()
        self.total_time_spent = self.finish_time - self.start_time
    
    def return_report(self) -> dict:
        """
        Retorna o dicionario com com os resultados
        """
        return {
            "best_solution": self.best_solution,
            "best_cost": self.best_solution_cost,
            "time_for_best_sol": self.best_solution_time,
            }