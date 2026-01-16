import numpy as np
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1

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
    def __init__(self, population_size, network_type):
        self.size = population_size
        self.network_type = network_type
        self.agents = self.generate_agents()
        self.topology = self.create_topology()
        self.connect_agents()

    def create_topology(self):
        if self.network_type == 'lattice':
            return self.generate_lattice()
        else:
            raise ValueError("Unsupported network type")

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

def simulate_one_episode(num_agent, t_steps=100000):
    initial_fraction = 0.5
    t = np.linspace(0, 1000000, t_steps)
    society = Society(num_agent, 'lattice')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)
    m = 0.4
    episode_cooperator_fractions = []
    for time_step in range(len(t)):
        x_lattice = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_lattice - 1)
        m = np.clip(m + dm, 0, 1)
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)
        decision_maker.update_strategy(society.agents)
        episode_cooperator_fractions.append(x_lattice)
    return episode_cooperator_fractions

def save_results_to_csv(time, results_dict, filename="cooperation_fractions.csv"):
    data = {"time": time}
    for num_agent, results in results_dict.items():
        average_cooperator_fraction = np.mean(results["cooperator_fraction"], axis=0)
        data[f"cooperation_fraction_{num_agent}"] = average_cooperator_fraction
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_results(t, results_dict):
    plt.figure(figsize=(12, 8))
    plt.xlabel('Time')
    plt.ylabel('Cooperative Fraction')
    plt.title('Evolution of Cooperative Fraction for Different Population Sizes')
    plt.grid(True)
    for num_agent, results in results_dict.items():
        average_cooperator_fraction = np.mean(results["cooperator_fraction"], axis=0)
        plt.plot(t, average_cooperator_fraction, linestyle='-', label=f"Population Size: {num_agent}")
    plt.legend(loc='upper right')
    plt.savefig('Quasi-stable phase.png', dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    population_sizes = [100, 225, 400, 625, 900, 1225, 1600]
    results_dict = {}
    time = np.linspace(0, 1000000, 100000)
    episodes_per_pop = 100

    for num_agent in population_sizes:
        print(f"Simulating {episodes_per_pop} episodes for population size: {num_agent}")
        with mp.Pool(processes=48) as pool:
            # Each episode gets the same num_agent
            episode_results = pool.starmap(simulate_one_episode, [(num_agent, len(time))] * episodes_per_pop)
        results = {
            "cooperator_fraction": episode_results,
            # Optionally, you can add resource_levels if you want to collect them as well
        }
        results_dict[num_agent] = results

    save_results_to_csv(time, results_dict)
    plot_results(time, results_dict)
