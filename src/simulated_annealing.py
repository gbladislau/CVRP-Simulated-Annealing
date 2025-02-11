from cvrp import CVRP

class SimulatedAnnealing:
    
    temp_atual: int
    
    def gen_initial_sol(instance: CVRP):
        