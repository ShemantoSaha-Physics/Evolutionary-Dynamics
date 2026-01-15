import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig
import multiprocessing
from functools import partial
from scipy.stats import gaussian_kde
import csv  

# Constants
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

def simulate_episode(population_size, initial_fraction=0.5, timesteps=100000, _=None):
    society = Society(population_size, 'lattice')
    init_c = choose_init_c(population_size, initial_fraction)
    society.agents = init_strategy(society.agents, init_c)
    decision_maker = Decision(beta=10)
    x_series = []
    m = 0.5  # Feedback variable
    for t in range(timesteps):
        x = society.count_fraction()
        x_series.append(x)
        # Feedback-evolving dynamics
        dm = epsilon * m * (1 - m) * ((1 + theta) * x - 1)
        m = np.clip(m + dm, 0, 1)
        # Evolving payoff matrix and dilemma strengths
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)
        society.agents = decision_maker.update_strategy(society.agents)
    return {'cooperator_fraction': np.array(x_series)}

def compute_potential_landscape(x_series, grid_points=200, bandwidth=0.1):
    x_series = x_series[int(0.1 * len(x_series)):]  # discard transient
    kde = gaussian_kde(x_series, bw_method=bandwidth)
    x_grid = np.linspace(min(x_series), max(x_series), grid_points)
    density = kde(x_grid)
    density = np.clip(density, 1e-12, None)
    potential = -np.log(density)
    potential -= np.min(potential)
    return x_grid, potential


def get_well_and_barrier(potential, centers):
    min_idx = np.argmin(potential)
    well_depth = potential.max() - potential[min_idx]
    barrier_height = potential.max()
    return well_depth, barrier_height, centers[min_idx]

def main():
    population_sizes = [1225, 2025, 3025, 4225, 5625]
    timesteps = 100000
    num_episodes = 100
    initial_fraction = 0.5

    well_depths = []
    barrier_heights = []
    well_positions = []

    # For saving potential landscape data
    potential_landscape_rows = []

    plt.figure(figsize=(10,6))
    for N in population_sizes:
        # Parallelize episodes for each population size
        with multiprocessing.Pool(processes=48) as pool:
            results = pool.map(
                partial(simulate_episode, N, initial_fraction, timesteps),
                range(num_episodes)
            )
        all_x = [result['cooperator_fraction'] for result in results]
        all_x = np.array(all_x)
        mean_x = np.mean(all_x, axis=0)
        centers, potential = compute_potential_landscape(mean_x)
        plt.plot(centers, potential, label=f'N={N}')
        well_depth, barrier_height, well_pos = get_well_and_barrier(potential, centers)
        well_depths.append(well_depth)
        barrier_heights.append(barrier_height)
        well_positions.append(well_pos)
        # Save landscape data for this N
        for c, p in zip(centers, potential):
            potential_landscape_rows.append([N, c, p])

    # Save potential landscape data to CSV
    with open('potential_landscape_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Population Size', 'Center', 'Potential'])
        writer.writerows(potential_landscape_rows)

    plt.xlabel('Cooperator Fraction (x)')
    plt.ylabel('Potential V(x)')
    plt.title('Potential Landscape vs Population Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('potential_landscape_vs_population.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(population_sizes, well_depths, 'o-', label='Well Depth')
    plt.plot(population_sizes, barrier_heights, 's-', label='Barrier Height')
    plt.xlabel('Population Size (N)')
    plt.ylabel('Potential (normalized)')
    plt.title('Well Depth and Barrier Height vs Population Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('well_and_barrier_vs_population.png', dpi=300)
    plt.show()

    # Save well depth, barrier height, and well position data to CSV
    with open('well_and_barrier_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Population Size', 'Well Depth', 'Barrier Height', 'Well Position'])
        for N, wd, bh, wp in zip(population_sizes, well_depths, barrier_heights, well_positions):
            writer.writerow([N, wd, bh, wp])

    print("Well depths:", well_depths)
    print("Barrier heights:", barrier_heights)
    print("Well positions (cooperator fraction):", well_positions)
    print("Plots saved as 'potential_landscape_vs_population.png' and 'well_and_barrier_vs_population.png'.")
    print("CSV files saved as 'potential_landscape_data.csv' and 'well_and_barrier_data.csv'.")

if __name__ == "__main__":
    main()

