import cvrp
import simulated_annealing



instance = cvrp.CVRP("./instances/A/A-n32-k5.vrp")
sa = simulated_annealing.SimulatedAnnealing(100, 0.999, 300)
sa.optimize(instance)
print(sa.return_report())