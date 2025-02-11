import vrplib
import re
import numpy as np

class CVRP:
    """ 
    Classe para representar uma instância do CVRP
    com funções para calculo de custo, pertubação da solução.
    """
    instance_dict: dict
    number_of_trucks: int
    distance_matrix: np.ndarray
    vertex_demand: np.ndarray
    truck_capacity: int
    V: np.ndarray
    depot_i: int
    optimal_value: int
    
    def __init__(self, intance_path: str = ""):
        """
        Cria uma instância do problema a partir de um arquivo

        Args:
            intance_path (str, optional): path para o arquivo .vrp. Defaults to "".
        """
        instance_dict = vrplib.read_instance(intance_path, compute_edge_weights=False)
        self.instance_dict = instance_dict
        print(instance_dict)
        # match com numero de caminhões (numero de rotas)
        match = re.search(r'No\s+of\s+trucks:\s*(\d+)', instance_dict["comment"])
        if match:
            print("match com regex funcionou")
            self.number_of_trucks = int(match.group(1))  
        else:
            print("match não funcionou, usando nome do arquivo")
            self.number_of_trucks = int(intance_path.split("k")[1].removesuffix(".vrp"))
            
        self._generate_distance_matrix(instance_dict)
        self.vertex_demand = instance_dict["demand"]
        self.truck_capacity = instance_dict["capacity"]
        self.V = instance_dict["node_coord"]
        self.depot_i = instance_dict["depot"][0]
        
    def _generate_distance_matrix(self, instance_dict: dict):
        """
        Calcula a matrix de distancias euclidiana de todos os vértices para todos
        """
        shape = (instance_dict["dimension"], instance_dict["dimension"]) # matriz quadrada
        temp_array = np.zeros(shape=shape, dtype=float)
        for v in enumerate(instance_dict["node_coord"]):    # v[0] -> indice  e  v[1] -> vetor(x,y)
            for u in enumerate(instance_dict["node_coord"]):
                # norma de dois vetores que é equivalente a distância euclididiana
                # usei a norma pois é mais rápido por conta do calculo com vetores
                # ||v - u|| -> sqrt( sum((x_i - y_i)^2)) )
                temp_array[v[0]][u[0]] = np.linalg.norm(v[1] - u[1]) 
        self.distance_matrix = temp_array.copy()            
    
    def cost_function(self, solution: np.ndarray) -> float:
        cost = 0.0
        # array de vertices
        for route in solution:
            # para todos os vertices da rota (route) pega a distancia dele com o próximo (route[1:])
            cost += sum(self.distance_matrix[a][b] for a, b in zip(route, route[1:]))
        return cost 