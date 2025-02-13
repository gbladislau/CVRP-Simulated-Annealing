import cvrp
import simulated_annealing

instance = cvrp.CVRP("./instances/A/A-n32-k5.vrp")
sa = simulated_annealing.SimulatedAnnealing(500, 0.7, 10)
sa.optimize(instance)