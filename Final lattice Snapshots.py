import pandas as pd
import numpy as np
import networkx as nx
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool

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

def snapshot_colored(society, m, t, filename):
    G = society.topology
    pos = {node: node for node in G.nodes()}
    node_colors = ['blue' if agent.strategy == 'C' else 'darkred' for agent in society.agents]

    plt.figure(figsize=(8,8))
    nx.draw(G, pos=pos, node_color=node_colors, node_size=100, edge_color='gray')
    plt.title(f"t={t}s, m={m:.2f}")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

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
                prob = 1 / (1 + np.exp((focal.point - opp.point)/self.kappa))
                focal.next_strategy = opp.strategy if rnd.random() < prob else focal.strategy
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

snapshot_timesteps = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000, 15000, 30000]

def run_single_episode(episode_params):
    num_agent, initial_fraction, max_time, dt, episode = episode_params
    t = np.arange(0, max_time + dt, dt)

    society = Society(num_agent, 'lattice')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    m = 0.4
    episode_cooperator_fractions = []
    episode_resource_levels = []
    episode_strategies_snapshots = {}

    for time_step in range(len(t)):
        x_lattice = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_lattice - 1)
        m = np.clip(m + dm, 0, 1)

        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)
        decision_maker.update_strategy(society.agents)

        episode_cooperator_fractions.append(x_lattice)
        episode_resource_levels.append(m)

        if t[time_step] in snapshot_timesteps:
            episode_strategies_snapshots[int(t[time_step])] = [agent.strategy for agent in society.agents]

    return episode_cooperator_fractions, episode_resource_levels, episode_strategies_snapshots

def simulate_with_multiprocessing():
    num_agent = 625
    num_episodes = 100
    initial_fraction = 0.5
    max_time = 30000
    dt = 1

    episode_params = [(num_agent, initial_fraction, max_time, dt, ep) for ep in range(num_episodes)]

    with Pool(processes=32) as pool:
        results_list = pool.map(run_single_episode, episode_params)

    results = {
        "cooperator_fraction": [],
        "resource_levels": [],
        "all_strategies": {ts: [] for ts in snapshot_timesteps}
    }

    for cf, rl, strat_snap in results_list:
        results["cooperator_fraction"].append(cf)
        results["resource_levels"].append(rl)
        for ts in snapshot_timesteps:
            results["all_strategies"][ts].append(strat_snap[ts])

    return results

def aggregate_snapshots(all_agent_strategies_at_t):
    num_agents = len(all_agent_strategies_at_t[0])
    aggregated_strategy = []
    for agent_idx in range(num_agents):
        strategies = [episode[agent_idx] for episode in all_agent_strategies_at_t]
        most_common_strategy = Counter(strategies).most_common(1)[0][0]
        aggregated_strategy.append(most_common_strategy)
    return aggregated_strategy

def visualize_aggregated_snapshot(society, aggregated_strategy, m, t, filename):
    G = society.topology
    pos = {node: node for node in G.nodes()}
    node_colors = ['blue' if strat == 'C' else 'darkred' for strat in aggregated_strategy]

    plt.figure(figsize=(8,8))
    nx.draw(G, pos=pos, node_color=node_colors, node_size=100, edge_color='gray')
    plt.title(f"Aggregated Snapshot t={t}s, m={m:.2f}")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def create_aggregated_snapshot_figure(results, time_step_to_visualize):
    society = Society(625, 'lattice')
    aggregated_strategy = aggregate_snapshots(results["all_strategies"][time_step_to_visualize])
    m = np.mean([episode[time_step_to_visualize] for episode in results["resource_levels"]])
    visualize_aggregated_snapshot(society, aggregated_strategy, m, time_step_to_visualize, f"aggregated_snapshot_t{time_step_to_visualize}.png")

def plot_results(t, results):
    average_cooperator_fraction = np.mean(results["cooperator_fraction"], axis=0)
    average_resource_levels = np.mean(results["resource_levels"], axis=0)

    Dg_history = []
    Dr_history = []
    for episode_resource_levels in results["resource_levels"]:
        episode_Dg = []
        episode_Dr = []
        for m in episode_resource_levels:
            A_evo = A(m)
            Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
            episode_Dg.append(Dg_prime)
            episode_Dr.append(Dr_prime)
        Dg_history.append(episode_Dg)
        Dr_history.append(episode_Dr)

    average_Dg = np.mean(Dg_history, axis=0)
    average_Dr = np.mean(Dr_history, axis=0)

    plt.figure(figsize=(12, 15))

    plt.subplot(4, 1, 1)
    plt.plot(t, average_cooperator_fraction, color='b', linestyle='-', label="Cooperative Fraction")
    plt.xlabel('Time')
    plt.ylabel('Fraction')
    plt.title('Cooperative Fraction Evolution')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, average_resource_levels, color='g', linestyle='-', label="Resource Level")
    plt.xlabel('Time')
    plt.ylabel('m')
    plt.title('Resource Dynamics')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, average_Dg, 'r--', label="Dg")
    plt.plot(t, average_Dr, 'b--', label="Dr")
    plt.xlabel('Time')
    plt.ylabel('Strength')
    plt.title('Dilemma Strengths')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(average_cooperator_fraction, average_resource_levels, 'g-', label="Feedback")
    plt.xlabel('Cooperation')
    plt.ylabel('Resources')
    plt.title('Cooperation-Resources Relationship')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


if __name__ == "__main__":
    results = simulate_with_multiprocessing()
    plot_results(np.arange(0, 30001, 1), results)

    for ts in snapshot_timesteps:
        create_aggregated_snapshot_figure(results, ts)
