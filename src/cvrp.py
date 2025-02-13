import vrplib
import re
import random
import numpy as np
import time
random.seed(time.time())

class CVRP:
    """ 
    Classe para representar uma instância do CVRP
    com funções para calculo de custo, pertubação da solução.
    """
    name: str
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
        # usando somente para ler o arquivo (peço para não computar os pesos)
        # match com numero de caminhões (numero de rotas)
        match = re.search(r'No\s+of\s+trucks:\s*(\d+)', instance_dict["comment"])
        if match:
            # ("match com regex funcionou")
            self.number_of_trucks = int(match.group(1))  
        else:
            # ("match não funcionou, usando nome do arquivo")
            self.number_of_trucks = int(intance_path.split("k")[1].removesuffix(".vrp"))
        self.__generate_distance_matrix(instance_dict)
        self.vertex_demand = instance_dict["demand"]
        self.truck_capacity = instance_dict["capacity"]
        self.V = instance_dict["node_coord"]
        self.depot_i = int(instance_dict["depot"][0])
        self.name = instance_dict["name"]
        
        
    def __generate_distance_matrix(self, instance_dict: dict):
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
        self.distance_matrix = temp_array.astype(int) #salva as distâncias em inteiro            
    
    
    def calculate_cost(self, solution: np.ndarray) -> float:
        """Calcula o custo da solucao

        Args:
            solution (np.ndarray): _description_

        Returns:
            float: _description_
        """
        cost = 0.0
        # array de vertices
        for route in solution:
            # para todos os vertices da rota (route) pega a distancia dele com o próximo (route[1:])
            cost += sum(self.distance_matrix[a][b] for a, b in zip(route, route[1:]))
        return cost 
    
    
    def gen_initial_sol(self) -> list[list[int]]:
        """
        Cria uma solução inicial alatóriamente
        inserindo vertices até atingir a capacidade maxima
        """
        # Pega todos os indices dos vertices para ir criando as rotas
        vertex_list = list(range(1, len(self.V)))
        # verifica se a rota é válida
        while True:
            initial_sol = []
            for _ in range(0, self.number_of_trucks):
                route_demand = 0
                route = []
                while len(vertex_list):
                    # pega um elemento aleatóriamente
                    u = random.choice(vertex_list)
                    # remove ele da lista
                    vertex_list.remove(u)
                    demand = self.vertex_demand[u] 
                    # se a demanda for maior devolve ele pro conjunto e acaba essa rota
                    if demand + route_demand > self.truck_capacity:
                        vertex_list.append(u)
                        break
                    else:
                        route_demand += self.vertex_demand[u]
                        route.append(u)
                # Adiciona deposito
                route.insert(0, self.depot_i) 
                route.append(self.depot_i)
                
                # termina rota e adiciona a solucao
                initial_sol.append(route)
            if self.verifica_solucao(initial_sol):
                return initial_sol
        
    
    def verifica_solucao(self, sol: list[list[int]]) -> bool:
        """Verifica se uma solução é valida

        Args:
            sol (list[list[int]]): solução

        Returns:
            bool: se a solução é válida ou não
        """
        vertex_set = set(range(1, len(self.V)))
        for route in sol:
            demand = sum([self.vertex_demand[v] for v in route])
            # demanda passou do total possível
            if demand > self.truck_capacity:
                return False
            elif route[0] != self.depot_i or route[-1] != self.depot_i:
                return False
            # verifica se elemento estava ou não em alguma outra rota
            try:
                for i in route[1:-2]:
                    vertex_set.remove(i)
            except KeyError:
                 # elemento ja foi removido logo estava em outra rota, 
                 # solução inválida
                return False 
            # atendeu todas as possíveis condições logo é valida
            return True
    
    @staticmethod  # não usa nada da classe e pode ser usado por qualquer obejeto desta classe
    def generate_new_solution(solution: list[list[int]]) -> list[list[int]]:
        """Gera uma nova solução utilizando uma de três possiveis perturbações
        aleatóriamente escolhida (SWAP, TWO-WAY SWAP, 2-opt)

        Args:
            solution (list[list[int]]): solução

        Returns:
            list[list[int]]: nova solução encontrada
        """
        new_solution = [route.copy() for route in solution]
        for _ in range(random.randint(1, len(new_solution))):
            match random.randint(0,2):
                case 0:
                    # SWAP 
                    # realiza de 1 até n (aleatório até o n° de rotas)
                    for _ in range(random.randint(1, len(new_solution))):
                        # escolhe 1 rota
                        route_idx = random.choice(range(len(new_solution)))
                        route = new_solution[route_idx]

                        # seleciona quantos elementos serão trocados (de 1 até len-2)
                        num_elements = random.randint(1, len(route)-2)

                        # escolhe aleatoriamente os índices dos elementos na rota1 (sem incluir primeiro e último)
                        selected_indices = sorted(random.sample(range(1, len(route) - 1), num_elements), reverse=True)
                        selected_elements = [route[i] for i in selected_indices]

                        # remove os elementos selecionados da rota1
                        for i in selected_indices:
                            del route[i]

                        # escolhe uma posição aleatória para inserir os elementos na rota (sem incluir primeiro e último)
                        insert_position = random.randint(1, len(route) - 1)
                        route[insert_position:insert_position] = selected_elements
                case 1:
                    # TWO-WAY SWAP entre rotas
                    # verifica se tem ao menos 2 rotas com pelo menos 3 elementos
                    valid_routes = [i for i in range(len(new_solution)) if len(new_solution[i]) > 2]
                    if len(valid_routes) < 2:
                        return new_solution  # Não a rotas suficientes

                    # escolhe 2 aleatorias
                    route1_idx, route2_idx = random.sample(valid_routes, 2)
                    route1, route2 = new_solution[route1_idx], new_solution[route2_idx]

                    # acha o numero max de elementos que podem ser trocados
                    max_swap = min(len(route1) - 2, len(route2) - 2)
                    if max_swap == 0: # rota só possui os depositos
                        return new_solution 

                    # seleciona o n° de elementos 
                    num_elements = random.randint(1, max_swap)

                    # seleciona os indices dos elementos a serem trocados, ignorando depositos
                    selected_indices1 = sorted(random.sample(range(1, len(route1) - 1), num_elements))
                    selected_indices2 = sorted(random.sample(range(1, len(route2) - 1), num_elements))

                    # troca elementos entre as rotas
                    for i, j in zip(selected_indices1, selected_indices2):
                        route1[i], route2[j] = route2[j], route1[i]
                case 2:
                    # escolhe rotas validas para essa operação
                    valid_routes = [i for i in range(len(new_solution)) if len(new_solution[i]) > 3]
                    if not valid_routes:
                        return new_solution  # não existem rotas validas

                    # rota aleatoria escolhida
                    route = new_solution[random.choice(valid_routes)]

                    # escolher 2 indices aleatórios (mantendo 1 ≤ i < j ≤ len(route)-2)
                    i, j = sorted(random.sample(range(1, len(route) - 1), 2))

                    # reverte a ordem entre i and j
                    route[i:j+1] = reversed(route[i:j+1])
            
        return new_solution