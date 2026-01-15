import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig
import csv
import multiprocessing
from functools import partial


# Constants
R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1


# Helper Functions
def A(m):
    return (1 - m) * np.array([[R0, S0], [T0, P0]]) + m * np.array([[R1, S1], [T1, P1]])


def calculate_dilemma_strengths(A_evo):
    Dg_prime = (A_evo[1, 0] - A_evo[0, 0]) / (A_evo[0, 0] - A_evo[1, 1])
    Dr_prime = (A_evo[1, 1] - A_evo[0, 1]) / (A_evo[0, 0] - A_evo[1, 1])
    return Dg_prime, Dr_prime


def jacobian_matrix(x, m):
    A_evo = A(m)
    Dg, Dr = calculate_dilemma_strengths(A_evo)
    J11 = -x * (1 - x) * ((1 - m) * (R0 - S0 - T0 + P0) + m * (R1 - S1 - T1 + P1))
    J12 = x * (1 - x) * ((R1 - R0 - T1 + T0) + (S1 - S0 - P1 + P0))
    J21 = epsilon * m * (1 - m) * (1 + theta)
    J22 = -epsilon * (1 - 2 * m) * ((1 + theta) * x - 1)
    return np.array([[J11, J12], [J21, J22]])


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


def spatial_correlation(society):
    same_strategy = 0
    total_edges = 0
    for agent in society.agents:
        for neighbor_id in agent.neighbors_id:
            if agent.id < neighbor_id:
                neighbor = society.agents[neighbor_id]
                if agent.strategy == neighbor.strategy:
                    same_strategy += 1
                total_edges += 1
    return same_strategy / total_edges if total_edges else 0


def simulate_episode(episode_num, num_agent, average_degree, initial_fraction, t):
    rnd.seed()  # Unique seed per process
    society = Society(num_agent, 'lattice')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    m = 0.4
    x = society.count_fraction()

    episode_data = []
    for time_step in range(len(t)):
        dm = epsilon * m * (1 - m) * ((1 + theta) * x - 1)
        m = np.clip(m + dm, 0, 1)
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)
        society.agents = decision_maker.update_strategy(society.agents)
        x = society.count_fraction()

        J = jacobian_matrix(x, m)
        eig_vals = eig(J)[0]

        spatial_corr = spatial_correlation(society)
        random_expectation = x**2 + (1 - x)**2

        episode_data.append([
            t[time_step], np.real(eig_vals)[0], np.real(eig_vals)[1],
            np.imag(eig_vals)[0], np.imag(eig_vals)[1],
            x, random_expectation, m, spatial_corr, episode_num
        ])
    return episode_data


def main():
    num_agent = 10000
    num_episodes = 100
    initial_fraction = 0.5
    t = np.linspace(0, 5000, 5000)
    csv_filename = 'Lattice_metrices.csv'

    num_processes = multiprocessing.cpu_count()
    print(f"Running with {num_processes} cores")
    pool = multiprocessing.Pool(processes=num_processes)

    episode_nums = range(1, num_episodes + 1)
    partial_simulate_episode = partial(
        simulate_episode,
        num_agent=num_agent,
        average_degree=4,  # corrected average degree for lattice
        initial_fraction=initial_fraction,
        t=t
    )

    results = pool.map(partial_simulate_episode, episode_nums)

    pool.close()
    pool.join()

    # Flatten results and write all episode data
    all_data = [row for episode_data in results for row in episode_data]

    # Convert to numpy array for averaging
    all_data_np = np.array(all_data)

    # Calculate averages over episodes for each time point
    times = np.unique(all_data_np[:, 0])
    avg_rows = []
    for time in times:
        rows_at_time = all_data_np[all_data_np[:, 0] == time]
        mean_vals = np.mean(rows_at_time, axis=0)
        # Mark episode as -1 for averages
        mean_vals[-1] = -1
        avg_rows.append(mean_vals)

    # Write all data + averages to CSV
    with open(csv_filename, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'time', 'real_eig1', 'real_eig2', 'imag_eig1', 'imag_eig2',
            'cooperator_fraction', 'random_expectation', 'resource_level',
            'spatial_correlation', 'episode'
        ])
        csvwriter.writerows(all_data)
        csvwriter.writerows(avg_rows)

    print(f"Simulation complete. Data and averages saved to {csv_filename}")


if __name__ == "__main__":
    main()