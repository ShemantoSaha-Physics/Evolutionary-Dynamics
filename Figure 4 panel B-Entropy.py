import numpy as np
import networkx as nx
import random as rnd
import pandas as pd
from multiprocessing import Pool

# Constants and helper functions (as in your code)
R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1

def calculate_entropy(agents):
    total = len(agents)
    c_count = sum(1 for a in agents if a.strategy == "C")
    d_count = total - c_count
    p_c = c_count / total
    p_d = d_count / total
    if p_c == 0 or p_d == 0:
        return 0.0
    return -p_c * np.log2(p_c) - p_d * np.log2(p_d)

def A(m):
    return (1 - m) * np.array([[R0, S0], [T0, P0]]) + m * np.array([[R1, S1], [T1, P1]])

def calculate_dilemma_strengths(A_evo):
    Dg_prime = (A_evo[1, 0] - A_evo[0, 0]) / (A_evo[0, 0] - A_evo[1, 1])
    Dr_prime = (A_evo[1, 1] - A_evo[0, 1]) / (A_evo[0, 0] - A_evo[1, 1])
    return Dg_prime, Dr_prime

class Agent:
    def __init__(self, id):
        self.id = id
        self.point = 0
        self.strategy = None
        self.next_strategy = None
        self.neighbors_id = []

class Society:
    def __init__(self, population_size):
        self.size = population_size
        self.agents = self.generate_agents()
        self.topology = self.generate_lattice()
        self.connect_agents()

    def generate_agents(self):
        return [Agent(id_num) for id_num in range(self.size)]

    def generate_lattice(self):
        n, h = self.calculate_grid_dimensions()
        G = nx.grid_graph(dim=[n, h])
        self.add_toroidal_edges(G, n, h)
        return G

    def add_toroidal_edges(self, G, n, h):
        for j in range(h):
            G.add_edge((0, j), (n - 1, j))
        for i in range(n):
            G.add_edge((i, 0), (i, h - 1))

    def calculate_grid_dimensions(self):
        n = int(np.sqrt(self.size))
        while self.size % n != 0:
            n -= 1
        h = self.size // n
        return n, h

    def connect_agents(self):
        n, h = self.calculate_grid_dimensions()
        for focal in self.agents:
            x = focal.id // h
            y = focal.id % h
            focal.neighbors_id = []
            if (x, y) in self.topology.nodes():
                for nb in self.topology.neighbors((x, y)):
                    neighbor_id = nb[0] * h + nb[1]
                    if neighbor_id < self.size:
                        focal.neighbors_id.append(neighbor_id)
            else:
                raise ValueError(f"Node {(x, y)} not found in the graph.")

    def count_fraction(self):
        cooperative_agents = [agent for agent in self.agents if agent.strategy == "C"]
        return len(cooperative_agents) / self.size

class Decision:
    def __init__(self, beta):
        self.kappa = 1 / beta
        self.Dg = None
        self.Dr = None

    def set_dilemma_strengths(self, Dg, Dr):
        self.Dg = Dg
        self.Dr = Dr

    def count_payoff(self, agents):
        for focal in agents:
            focal.point = 0.0
            for neighbor_id in focal.neighbors_id:
                neighbor = agents[neighbor_id]
                if focal.strategy == "C":
                    focal.point += 1 if neighbor.strategy == "C" else -self.Dr
                else:
                    focal.point += 1 + self.Dg if neighbor.strategy == "C" else 0
        return agents

    def pw_fermi(self, agents):
        for focal in agents:
            opp_id = rnd.choice(focal.neighbors_id)
            opp = agents[opp_id]
            if opp.strategy != focal.strategy:
                prob = 1 / (1 + np.exp((focal.point - opp.point) / self.kappa))
                if rnd.random() < prob:
                    focal.next_strategy = opp.strategy
                else:
                    focal.next_strategy = focal.strategy
            else:
                focal.next_strategy = focal.strategy
        return agents

    def update_strategy(self, agents):
        agents = self.pw_fermi(self.count_payoff(agents))
        for focal in agents:
            focal.strategy = focal.next_strategy
        return agents

def choose_init_c(num_agent, initial_fraction):
    return rnd.sample(range(num_agent), k=int(initial_fraction * num_agent))

def init_strategy(agents, init_c):
    for focal in agents:
        focal.strategy = "C" if focal.id in init_c else "D"
    return agents

# --- Simulation function for one episode ---
def simulate_one_episode(args):
    num_agent, total_timesteps, save_every, initial_fraction = args
    society = Society(num_agent)
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)
    m = 0.4
    entropy_list = []
    for t in range(total_timesteps):
        x = society.count_fraction()
        dm = epsilon*m*(1-m)*((1+theta)*x - 1)
        m = np.clip(m + dm, 0, 1)
        A_evo = A(m)
        Dg, Dr = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg, Dr)
        decision_maker.update_strategy(society.agents)
        if t % save_every == 0:
            entropy_list.append(calculate_entropy(society.agents))
    return entropy_list

def run_parallel_simulation(population_sizes, num_episodes=100, total_timesteps=1000000, save_every=10, initial_fraction=0.5):
    results = {}
    save_points = total_timesteps // save_every
    for num_agent in population_sizes:
        print(f"Simulating population size: {num_agent}")
        args_list = [(num_agent, total_timesteps, save_every, initial_fraction) for _ in range(num_episodes)]
        with Pool(processes=48) as pool:
            entropy_matrix = pool.map(simulate_one_episode, args_list)
        entropy_matrix = np.array(entropy_matrix)
        mean_entropy = np.mean(entropy_matrix, axis=0)
        results[num_agent] = mean_entropy
    return results, save_points

def save_entropy_results(results, save_points, save_every, filename="mean_entropy.csv"):
    time = np.arange(0, save_points * save_every, save_every)
    data = {"Time": time}
    for size, entropy in results.items():
        data[f"Entropy_{size}"] = entropy
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    population_sizes = [100, 225, 400, 625, 900, 1225, 1600]
    num_episodes = 100
    total_timesteps = 1000000
    save_every = 10  # Save every 10 steps
    results, save_points = run_parallel_simulation(population_sizes, num_episodes, total_timesteps, save_every)
    save_entropy_results(results, save_points, save_every)

