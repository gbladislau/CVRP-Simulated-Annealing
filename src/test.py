import cvrp
import simulated_annealing



instance = cvrp.CVRP("./instances/A/A-n80-k10.vrp")
print(instance.V)
sa = simulated_annealing.SimulatedAnnealing(500, 0.7, 10)
print(sa)
sa.optimize(instance)