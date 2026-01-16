#Lattice
import numpy as np
import pandas as pd
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import detrend
import time 


R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1
beta = 10 

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

def shuffle_time_series(time_series):
    shuffled_series = time_series.copy()
    np.random.shuffle(shuffled_series)
    return shuffled_series

def phase_randomized_surrogate(time_series):
    n = len(time_series)
    fft_coeffs = np.fft.fft(time_series)
    num_pos = (n - 1) // 2 
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, num_pos))
    if n % 2 == 0:
        phases = np.concatenate(([1], random_phases, [1], np.conjugate(random_phases[::-1])))
    else:
        phases = np.concatenate(([1], random_phases, np.conjugate(random_phases[::-1])))
    surrogate_fft = fft_coeffs * phases
    surrogate_signal = np.fft.ifft(surrogate_fft).real
    return surrogate_signal

def mdfa(time_series, q_values, scales, order=1):
    N = len(time_series)
    F_q = np.zeros((len(q_values), len(scales)))
    Y = np.cumsum(time_series - np.mean(time_series))

    for i, q in enumerate(q_values):
        for j, s in enumerate(scales):
            Ns = N // s  
            F_qs = np.zeros(Ns * 2) 
            for v in range(Ns):
                segment = Y[v * s:(v + 1) * s]
                trend = detrend(segment, type='linear') 
                F_qs[v] = np.sqrt(np.mean(trend**2)) 
            for v in range(Ns):
                segment = Y[N-(v+1)*s:N-v*s]
                trend = detrend(segment, type='linear') 
                F_qs[Ns + v] = np.sqrt(np.mean(trend**2))
            if q == 0:
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(F_qs**2)))
            else:
                F_q[i, j] = (np.mean(F_qs**q))**(1/q)

    
    log_s = np.log(scales)
    h_q = np.zeros(len(q_values))  
    t_q = np.zeros(len(q_values)) 

    for i in range(len(q_values)):
        log_F_q = np.log(F_q[i, :])
        coeffs = np.polyfit(log_s, log_F_q, 1)
        h_q[i] = coeffs[0]
        t_q[i] = q_values[i] * h_q[i] - 1

    
    alpha = np.diff(t_q) / np.diff(q_values)
    alpha = (alpha[:-1] + alpha[1:]) / 2  
    f_alpha = q_values[1:-1] * alpha - t_q[1:-1]

    return F_q, h_q, t_q, alpha, f_alpha

num_agent = 10000
num_episodes = 1
initial_fraction = 0.5
t = np.linspace(0, 5000, 5000)
q_values = np.linspace(-5, 5, 21)  
scales = np.unique(np.logspace(np.log10(5), np.log10(len(t) // 10), num=15, dtype=int)) 
order = 1 

cooperator_fraction_history_across_time = []
resource_levels_across_time = []
alpha_values = []
f_alpha_values = []
tau_q_values = []
h_2_values = []

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")
    society = Society(num_agent, 'lattice')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    episode_cooperator_fractions = []
    episode_resource_levels = []

    m = 0.4 

    for time_step in tqdm(range(len(t)), desc=f"Episode {episode + 1}", leave=False):
        current_time = t[time_step]
        x_lattice = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_lattice - 1)
        m += dm
        m = np.clip(m, 0, 1) 
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)

        society.agents = decision_maker.update_strategy(society.agents)
        cooperator_fraction = society.count_fraction()
        episode_cooperator_fractions.append(cooperator_fraction)
        episode_resource_levels.append(m)

    
        if time_step % 100 == 0:
            ts = np.array(episode_cooperator_fractions)
            F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
            tau_q_values.append(t_q)
            h_2 = np.interp(2, q_values, h_q)
            h_2_values.append(h_2)

            df_original = pd.DataFrame({'q': q_values, 'h_q': h_q.tolist()})
            df_original.to_csv(f"h_q_vs_q_original_t{time_step}.csv", index=False)

            shuffled_series = shuffle_time_series(ts)
            _, h_q_shuffled, *_ = mdfa(shuffled_series, q_values, scales, order)
            df_shuffled = pd.DataFrame({'q': q_values, 'h_q': h_q_shuffled.tolist()})
            df_shuffled.to_csv(f"h_q_vs_q_shuffled_t{time_step}.csv", index=False)

            surrogate_series = phase_randomized_surrogate(ts)
            _, h_q_surrogate, *_ = mdfa(surrogate_series, q_values, scales, order)
            df_surrogate = pd.DataFrame({'q': q_values, 'h_q': h_q_surrogate.tolist()})
            df_surrogate.to_csv(f"h_q_vs_q_surrogate_t{time_step}.csv", index=False)


    cooperator_fraction_history_across_time.append(episode_cooperator_fractions)
    resource_levels_across_time.append(episode_resource_levels)

    ts = np.array(episode_cooperator_fractions)
    F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
    alpha_values.append(alpha)
    f_alpha_values.append(f_alpha)

Dg_history_across_time = []
Dr_history_across_time = []

for episode_resource_levels in resource_levels_across_time:
    episode_Dg = []
    episode_Dr = []
    for m in episode_resource_levels:
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        episode_Dg.append(Dg_prime)
        episode_Dr.append(Dr_prime)
    Dg_history_across_time.append(episode_Dg)
    Dr_history_across_time.append(episode_Dr)

average_cooperator_fraction_over_time = np.mean(np.array(cooperator_fraction_history_across_time), axis=0)
average_resource_levels_over_time = np.mean(np.array(resource_levels_across_time), axis=0)
average_Dg_over_time = np.mean(np.array(Dg_history_across_time), axis=0)
average_Dr_over_time = np.mean(np.array(Dr_history_across_time), axis=0)

df_mf = pd.DataFrame({'alpha': alpha_values[-1], 'f_alpha': f_alpha_values[-1]})
df_mf.to_csv("multifractal_spectrum.csv", index=False)
q_values_list = q_values.tolist() 

data_tau_q = []
for i, tau_q in enumerate(tau_q_values):
    time_step = i * 100  
    tau_q_list = tau_q.tolist()  
    row = {'Time_Step': time_step, **{f'q_{j}': q for j, q in enumerate(q_values_list)}, **{f'tau_q_{j}': tq for j, tq in enumerate(tau_q_list)}}
    data_tau_q.append(row)

df_tau_q = pd.DataFrame(data_tau_q)
df_tau_q.to_csv("tau_q_vs_q.csv", index=False)

df_h2 = pd.DataFrame({'Time_Step': range(0, len(h_2_values) * 100, 100), 'h_2': h_2_values})
df_h2.to_csv("h_2_values_Lattice.csv", index=False)

print("Multifractal spectrum saved to multifractal_spectrum.csv")
print("tau_q vs q values saved to tau_q_vs_q.csv")
print("h_2 values saved to h_2_values_Lattice.csv")

csv_latticevalues = "simulation_results.csv"
df = pd.DataFrame({
    "Time": t,
    "Avg_Cooperator_Fraction": average_cooperator_fraction_over_time,
    "Avg_Resource_Level": average_resource_levels_over_time,
    "Avg_Dg": average_Dg_over_time,
    "Avg_Dr": average_Dr_over_time
})
df.to_csv(csv_latticevalues, index=False)

print(f"Simulation results saved to {csv_latticevalues}")

plt.figure(figsize=(12, 15))

plt.subplot(5, 1, 1)
plt.plot(t, average_cooperator_fraction_over_time, color='b', linestyle='-', label="Average Cooperative Fraction")
plt.xlabel('Time')
plt.ylabel('Average Cooperator Fraction')
plt.title('Average Evolution of Cooperative Fraction Across Episodes')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 2)
plt.plot(t, average_resource_levels_over_time, color='g', linestyle='-', label="Resource Level (m)")
plt.xlabel('Time')
plt.ylabel('Resource Level (m)')
plt.title('Resource Level Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 3)
plt.plot(t, average_Dg_over_time, color='r', linestyle='--', label="Dilemma Strength (Dg)")
plt.plot(t, average_Dr_over_time, color='purple', linestyle=':', label="Dilemma Strength (Dr)")
plt.xlabel('Time')
plt.ylabel('Dilemma Strengths (Dg, Dr)')
plt.title('Dilemma Strengths Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 4)
plt.plot(average_cooperator_fraction_over_time, average_resource_levels_over_time, color='g', linestyle='-', label="evolving path")
plt.xlabel('Frequency of cooperator')
plt.ylabel('Resource Level (m)')
plt.title('relation of evolution and feedback')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 5)
# Plot the multifractal spectrum for the last episode
plt.plot(alpha_values[-1], f_alpha_values[-1], 'b-')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f(\alpha)$')
plt.title('Multifractal Spectrum (MF-DFA)')
plt.grid(True)

plt.tight_layout()
plt.show()


num_surrogates = 50
original_time_series = np.array(cooperator_fraction_history_across_time[-1])
h_q_shuffled_surrogates = np.zeros((num_surrogates, len(q_values)))
h_q_phase_surrogates = np.zeros((num_surrogates, len(q_values)))
F_q_original, h_q_original, _, _, _ = mdfa(original_time_series, q_values, scales, order)

start_time = time.time()  

for i in range(num_surrogates):
    shuffled_time_series = shuffle_time_series(original_time_series)
    _, h_q_shuffled_surrogates[i], _, _, _ = mdfa(shuffled_time_series, q_values, scales, order)

for i in range(num_surrogates):
    phase_randomized_time_series = phase_randomized_surrogate(original_time_series)
    _, h_q_phase_surrogates[i], _, _, _ = mdfa(phase_randomized_time_series, q_values, scales, order)

end_time = time.time() 
total_time = end_time - start_time
print(f"Time taken to generate and analyze {num_surrogates} surrogates: {total_time:.2f} seconds")

h_q_shuffled_mean = np.mean(h_q_shuffled_surrogates, axis=0)
h_q_shuffled_std = np.std(h_q_shuffled_surrogates, axis=0)
h_q_phase_mean = np.mean(h_q_phase_surrogates, axis=0)
h_q_phase_std = np.std(h_q_phase_surrogates, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(q_values, h_q_original, marker='o', label='Original h(q)')
plt.plot(q_values, h_q_shuffled_mean, linestyle='--', color='orange', label='Shuffled Mean')
plt.fill_between(q_values, h_q_shuffled_mean - h_q_shuffled_std, h_q_shuffled_mean + h_q_shuffled_std,
                 color='orange', alpha=0.2, label='Shuffled +/- STD')
plt.plot(q_values, h_q_phase_mean, linestyle='--', color='green', label='Phase Rand Mean')
plt.fill_between(q_values, h_q_phase_mean - h_q_phase_std, h_q_phase_mean + h_q_phase_std,
                 color='green', alpha=0.2, label='Phase Rand +/- STD')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.title('Comparison of h(q) for Original and Surrogate Time Series')
plt.legend()
plt.grid(True)
plt.show()



#ER
import numpy as np
import pandas as pd
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import detrend
import time

R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1
beta = 10  

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
    def __init__(self, population_size, average_degree, network_type):
        self.size = population_size
        self.network_type = network_type
        self.average_degree = average_degree
        self.agents = self.generate_agents()
        self.topology = self.create_topology()
        self.connect_agents()

    def create_topology(self):
        if self.network_type == 'ER':
            return nx.random_regular_graph(self.average_degree, self.size)
        else:
            raise ValueError("Unsupported network type")

    def connect_agents(self):
        for focal in self.agents:
            neighbors = list(self.topology.neighbors(focal.id))
            focal.neighbors_id = neighbors

    def generate_agents(self):
        return [Agent(id_num) for id_num in range(self.size)]

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

def shuffle_time_series(time_series):
    shuffled_series = time_series.copy()
    np.random.shuffle(shuffled_series)
    return shuffled_series

def phase_randomized_surrogate(time_series):
    n = len(time_series)
    fft_coeffs = np.fft.fft(time_series)
    num_pos = (n - 1) // 2 
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, num_pos))

    if n % 2 == 0: 
        phases = np.concatenate(([1], random_phases, [1], np.conjugate(random_phases[::-1])))
    else:  
        phases = np.concatenate(([1], random_phases, np.conjugate(random_phases[::-1])))

    surrogate_fft = fft_coeffs * phases
    surrogate_signal = np.fft.ifft(surrogate_fft).real

    return surrogate_signal

def mdfa(time_series, q_values, scales, order=1):

    N = len(time_series)
    F_q = np.zeros((len(q_values), len(scales)))
    Y = np.cumsum(time_series - np.mean(time_series))

    for i, q in enumerate(q_values):
        for j, s in enumerate(scales):
            Ns = N // s 
            F_qs = np.zeros(Ns * 2)
            for v in range(Ns):
                segment = Y[v * s:(v + 1) * s]
                trend = detrend(segment, type='linear') 
                F_qs[v] = np.sqrt(np.mean(trend**2)) 

            for v in range(Ns):
                segment = Y[N-(v+1)*s:N-v*s]
                trend = detrend(segment, type='linear') 
                F_qs[Ns + v] = np.sqrt(np.mean(trend**2))

            if q == 0:
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(F_qs**2)))
            else:
                F_q[i, j] = (np.mean(F_qs**q))**(1/q)

    log_s = np.log(scales)
    h_q = np.zeros(len(q_values))  
    t_q = np.zeros(len(q_values)) 

    for i in range(len(q_values)):
        log_F_q = np.log(F_q[i, :])
        coeffs = np.polyfit(log_s, log_F_q, 1)
        h_q[i] = coeffs[0]
        t_q[i] = q_values[i] * h_q[i] - 1

    alpha = np.diff(t_q) / np.diff(q_values)
    alpha = (alpha[:-1] + alpha[1:]) / 2  
    f_alpha = q_values[1:-1] * alpha - t_q[1:-1]

    return F_q, h_q, t_q, alpha, f_alpha

num_agent = 10000
num_episodes = 1
initial_fraction = 0.5
average_degree = 4  
t = np.linspace(0, 5000, 5000)
q_values = np.linspace(-5, 5, 21) 
scales = np.unique(np.logspace(np.log10(5), np.log10(len(t) // 10), num=15, dtype=int))  
order = 1 

cooperator_fraction_history_across_time = []
resource_levels_across_time = []
alpha_values = []
f_alpha_values = []
tau_q_values = []
h_2_values = []

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")
    society = Society(num_agent, average_degree, 'ER')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    episode_cooperator_fractions = []
    episode_resource_levels = []

    m = 0.4  

    for time_step in tqdm(range(len(t)), desc=f"Episode {episode + 1}", leave=False):
        current_time = t[time_step]
        x_er = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_er - 1)
        m += dm
        m = np.clip(m, 0, 1)  # Ensure m stays in [0, 1]
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)

        society.agents = decision_maker.update_strategy(society.agents)
        cooperator_fraction = society.count_fraction()
        episode_cooperator_fractions.append(cooperator_fraction)
        episode_resource_levels.append(m)

        if time_step % 100 == 0:
            ts = np.array(episode_cooperator_fractions)
            F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
            tau_q_values.append(t_q)
            h_2 = np.interp(2, q_values, h_q)
            h_2_values.append(h_2)
            
            df_original = pd.DataFrame({'q': q_values, 'h_q': h_q.tolist()})
            df_original.to_csv(f"h_q_vs_q_original_t{time_step}.csv", index=False)

            shuffled_series = shuffle_time_series(ts)
            _, h_q_shuffled, *_ = mdfa(shuffled_series, q_values, scales, order)
            df_shuffled = pd.DataFrame({'q': q_values, 'h_q': h_q_shuffled.tolist()})
            df_shuffled.to_csv(f"h_q_vs_q_shuffled_t{time_step}.csv", index=False)

            surrogate_series = phase_randomized_surrogate(ts)
            _, h_q_surrogate, *_ = mdfa(surrogate_series, q_values, scales, order)
            df_surrogate = pd.DataFrame({'q': q_values, 'h_q': h_q_surrogate.tolist()})
            df_surrogate.to_csv(f"h_q_vs_q_surrogate_t{time_step}.csv", index=False)

    cooperator_fraction_history_across_time.append(episode_cooperator_fractions)
    resource_levels_across_time.append(episode_resource_levels)

    ts = np.array(episode_cooperator_fractions)
    F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
    alpha_values.append(alpha)
    f_alpha_values.append(f_alpha)

Dg_history_across_time = []
Dr_history_across_time = []

for episode_resource_levels in resource_levels_across_time:
    episode_Dg = []
    episode_Dr = []
    for m in episode_resource_levels:
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        episode_Dg.append(Dg_prime)
        episode_Dr.append(Dr_prime)
    Dg_history_across_time.append(episode_Dg)
    Dr_history_across_time.append(episode_Dr)

average_cooperator_fraction_over_time = np.mean(np.array(cooperator_fraction_history_across_time), axis=0)
average_resource_levels_over_time = np.mean(np.array(resource_levels_across_time), axis=0)
average_Dg_over_time = np.mean(np.array(Dg_history_across_time), axis=0)
average_Dr_over_time = np.mean(np.array(Dr_history_across_time), axis=0)

df_mf = pd.DataFrame({'alpha': alpha_values[-1], 'f_alpha': f_alpha_values[-1]})
df_mf.to_csv("multifractal_spectrum_ER.csv", index=False)

q_values_list = q_values.tolist()
data_tau_q = []

for i, tau_q in enumerate(tau_q_values):
    time_step = i * 100 
    tau_q_list = tau_q.tolist()  
    row = {'Time_Step': time_step, **{f'q_{j}': q for j, q in enumerate(q_values_list)}, **{f'tau_q_{j}': tq for j, tq in enumerate(tau_q_list)}}
    data_tau_q.append(row)

df_tau_q = pd.DataFrame(data_tau_q)
df_tau_q.to_csv("tau_q_vs_q_ER.csv", index=False)

df_h2 = pd.DataFrame({'Time_Step': range(0, len(h_2_values) * 100, 100), 'h_2': h_2_values})
df_h2.to_csv("h_2_values_ER.csv", index=False)

print("Multifractal spectrum saved to multifractal_spectrum.csv")
print("tau_q vs q values saved to tau_q_vs_q.csv")
print("h_2 values saved to h_2_values_Lattice.csv")

csv_ERvalues = "simulation_results_ER.csv"
df = pd.DataFrame({
    "Time": t,
    "Avg_Cooperator_Fraction": average_cooperator_fraction_over_time,
    "Avg_Resource_Level": average_resource_levels_over_time,
    "Avg_Dg": average_Dg_over_time,
    "Avg_Dr": average_Dr_over_time
})
df.to_csv(csv_ERvalues, index=False)

print(f"Simulation results saved to {csv_ERvalues}")

plt.figure(figsize=(12, 15))

plt.subplot(5, 1, 1)
plt.plot(t, average_cooperator_fraction_over_time, color='b', linestyle='-', label="Average Cooperative Fraction")
plt.xlabel('Time')
plt.ylabel('Average Cooperator Fraction')
plt.title('Average Evolution of Cooperative Fraction Across Episodes')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 2)
plt.plot(t, average_resource_levels_over_time, color='g', linestyle='-', label="Resource Level (m)")
plt.xlabel('Time')
plt.ylabel('Resource Level (m)')
plt.title('Resource Level Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 3)
plt.plot(t, average_Dg_over_time, color='r', linestyle='--', label="Dilemma Strength (Dg)")
plt.plot(t, average_Dr_over_time, color='purple', linestyle=':', label="Dilemma Strength (Dr)")
plt.xlabel('Time')
plt.ylabel('Dilemma Strengths (Dg, Dr)')
plt.title('Dilemma Strengths Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 4)
plt.plot(average_cooperator_fraction_over_time, average_resource_levels_over_time, color='g', linestyle='-', label="evolving path")
plt.xlabel('Frequency of cooperator')
plt.ylabel('Resource Level (m)')
plt.title('relation of evolution and feedback')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 5)
# Plot the multifractal spectrum for the last episode
plt.plot(alpha_values[-1], f_alpha_values[-1], 'b-')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f(\alpha)$')
plt.title('Multifractal Spectrum (MF-DFA)')
plt.grid(True)

plt.tight_layout()
plt.show()


num_surrogates = 50

original_time_series = np.array(cooperator_fraction_history_across_time[-1])
h_q_shuffled_surrogates = np.zeros((num_surrogates, len(q_values)))
h_q_phase_surrogates = np.zeros((num_surrogates, len(q_values)))
F_q_original, h_q_original, _, _, _ = mdfa(original_time_series, q_values, scales, order)

start_time = time.time() 

for i in range(num_surrogates):
    shuffled_time_series = shuffle_time_series(original_time_series)
    _, h_q_shuffled_surrogates[i], _, _, _ = mdfa(shuffled_time_series, q_values, scales, order)

for i in range(num_surrogates):
    phase_randomized_time_series = phase_randomized_surrogate(original_time_series)
    _, h_q_phase_surrogates[i], _, _, _ = mdfa(phase_randomized_time_series, q_values, scales, order)

end_time = time.time() 
total_time = end_time - start_time
print(f"Time taken to generate and analyze {num_surrogates} surrogates: {total_time:.2f} seconds")

h_q_shuffled_mean = np.mean(h_q_shuffled_surrogates, axis=0)
h_q_shuffled_std = np.std(h_q_shuffled_surrogates, axis=0)
h_q_phase_mean = np.mean(h_q_phase_surrogates, axis=0)
h_q_phase_std = np.std(h_q_phase_surrogates, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(q_values, h_q_original, marker='o', label='Original h(q)')
plt.plot(q_values, h_q_shuffled_mean, linestyle='--', color='orange', label='Shuffled Mean')
plt.fill_between(q_values, h_q_shuffled_mean - h_q_shuffled_std, h_q_shuffled_mean + h_q_shuffled_std,
                 color='orange', alpha=0.2, label='Shuffled +/- STD')
plt.plot(q_values, h_q_phase_mean, linestyle='--', color='green', label='Phase Rand Mean')
plt.fill_between(q_values, h_q_phase_mean - h_q_phase_std, h_q_phase_mean + h_q_phase_std,
                 color='green', alpha=0.2, label='Phase Rand +/- STD')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.title('Comparison of h(q) for Original and Surrogate Time Series')
plt.legend()
plt.grid(True)
plt.show()

#BA
import numpy as np
import pandas as pd
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import detrend
import time

R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1
beta = 10 

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
    def __init__(self, population_size, network_type='BA'): 
        self.size = population_size
        self.network_type = network_type
        self.agents = self.generate_agents()
        self.topology = self.create_topology()
        self.connect_agents()

    def create_topology(self):
        if self.network_type == 'BA':
            return self.generate_BA_network()
        else:
            raise ValueError("Unsupported network type")

    def connect_agents(self):
        for agent in self.agents:
            agent.neighbors_id = list(self.topology.neighbors(agent.id))

    def generate_agents(self):
        return [Agent(id_num) for id_num in range(self.size)]

    def generate_BA_network(self):
        m = int(np.sqrt(self.size)) '
        if m > self.size:
            m = self.size//2 #Safe Guard for m
        if m==0:
            m=1

        return nx.barabasi_albert_graph(self.size, m)

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

def shuffle_time_series(time_series):
    shuffled_series = time_series.copy()
    np.random.shuffle(shuffled_series)
    return shuffled_series

def phase_randomized_surrogate(time_series):
    n = len(time_series)
    fft_coeffs = np.fft.fft(time_series)
    num_pos = (n - 1) // 2
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, num_pos))

    if n % 2 == 0:  
        phases = np.concatenate(([1], random_phases, [1], np.conjugate(random_phases[::-1])))
    else: 
        phases = np.concatenate(([1], random_phases, np.conjugate(random_phases[::-1])))

    surrogate_fft = fft_coeffs * phases
    surrogate_signal = np.fft.ifft(surrogate_fft).real

    return surrogate_signal

def mdfa(time_series, q_values, scales, order=1):
   
    N = len(time_series)
    F_q = np.zeros((len(q_values), len(scales)))
    Y = np.cumsum(time_series - np.mean(time_series))
    for i, q in enumerate(q_values):
        for j, s in enumerate(scales):
            Ns = N // s 
            F_qs = np.zeros(Ns * 2) 

            for v in range(Ns):
                segment = Y[v * s:(v + 1) * s]
                trend = detrend(segment, type='linear') 
                F_qs[v] = np.sqrt(np.mean(trend**2)) 
            for v in range(Ns):
                segment = Y[N-(v+1)*s:N-v*s]
                trend = detrend(segment, type='linear') 
                F_qs[Ns + v] = np.sqrt(np.mean(trend**2)) 
            if q == 0:
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(F_qs**2)))
            else:
                F_q[i, j] = (np.mean(F_qs**q))**(1/q)

    log_s = np.log(scales)
    h_q = np.zeros(len(q_values))  
    t_q = np.zeros(len(q_values)) 

    for i in range(len(q_values)):
        log_F_q = np.log(F_q[i, :])
        coeffs = np.polyfit(log_s, log_F_q, 1)
        h_q[i] = coeffs[0]
        t_q[i] = q_values[i] * h_q[i] - 1

    alpha = np.diff(t_q) / np.diff(q_values)
    alpha = (alpha[:-1] + alpha[1:]) / 2  
    f_alpha = q_values[1:-1] * alpha - t_q[1:-1]

    return F_q, h_q, t_q, alpha, f_alpha

num_agent = 10000  
num_episodes = 1
initial_fraction = 0.5
t = np.linspace(0, 5000, 5000)
q_values = np.linspace(-5, 5, 21)  
scales = np.unique(np.logspace(np.log10(5), np.log10(len(t) // 10), num=15, dtype=int))
order = 1 

cooperator_fraction_history_across_time = []
resource_levels_across_time = []
alpha_values = []
f_alpha_values = []
tau_q_values = []
h_2_values = [] 

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")
    society = Society(num_agent, 'BA') 
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    episode_cooperator_fractions = []
    episode_resource_levels = []

    m = 0.4  

    for time_step in tqdm(range(len(t)), desc=f"Episode {episode + 1}", leave=False):
        current_time = t[time_step]
        
        x_lattice = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_lattice - 1)
        m += dm
        m = np.clip(m, 0, 1) 
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)

        society.agents = decision_maker.update_strategy(society.agents)
        cooperator_fraction = society.count_fraction()
        episode_cooperator_fractions.append(cooperator_fraction)
        episode_resource_levels.append(m)

        if time_step % 100 == 0:
            ts = np.array(episode_cooperator_fractions)
            F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
            tau_q_values.append(t_q)
            # Interpolate h_q to get h_2 (if q_values doesn't contain 2 exactly)
            h_2 = np.interp(2, q_values, h_q)
            h_2_values.append(h_2)

            df_original = pd.DataFrame({'q': q_values, 'h_q': h_q.tolist()})
            df_original.to_csv(f"h_q_vs_q_original_t{time_step}.csv", index=False)

            shuffled_series = shuffle_time_series(ts)
            _, h_q_shuffled, *_ = mdfa(shuffled_series, q_values, scales, order)
            df_shuffled = pd.DataFrame({'q': q_values, 'h_q': h_q_shuffled.tolist()})
            df_shuffled.to_csv(f"h_q_vs_q_shuffled_t{time_step}.csv", index=False)

            surrogate_series = phase_randomized_surrogate(ts)
            _, h_q_surrogate, *_ = mdfa(surrogate_series, q_values, scales, order)
            df_surrogate = pd.DataFrame({'q': q_values, 'h_q': h_q_surrogate.tolist()})
            df_surrogate.to_csv(f"h_q_vs_q_surrogate_t{time_step}.csv", index=False)

    cooperator_fraction_history_across_time.append(episode_cooperator_fractions)
    resource_levels_across_time.append(episode_resource_levels)

    ts = np.array(episode_cooperator_fractions)
    F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
    alpha_values.append(alpha)
    f_alpha_values.append(f_alpha)

Dg_history_across_time = []
Dr_history_across_time = []

for episode_resource_levels in resource_levels_across_time:
    episode_Dg = []
    episode_Dr = []
    for m in episode_resource_levels:
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        episode_Dg.append(Dg_prime)
        episode_Dr.append(Dr_prime)
    Dg_history_across_time.append(episode_Dg)
    Dr_history_across_time.append(episode_Dr)

average_cooperator_fraction_over_time = np.mean(np.array(cooperator_fraction_history_across_time), axis=0)
average_resource_levels_over_time = np.mean(np.array(resource_levels_across_time), axis=0)
average_Dg_over_time = np.mean(np.array(Dg_history_across_time), axis=0)
average_Dr_over_time = np.mean(np.array(Dr_history_across_time), axis=0)

df_mf = pd.DataFrame({'alpha': alpha_values[-1], 'f_alpha': f_alpha_values[-1]})
df_mf.to_csv("multifractal_spectrum_BA.csv", index=False)

q_values_list = q_values.tolist()  
data_tau_q = []

for i, tau_q in enumerate(tau_q_values):
    time_step = i * 100  
    tau_q_list = tau_q.tolist()  
    row = {'Time_Step': time_step, **{f'q_{j}': q for j, q in enumerate(q_values_list)}, **{f'tau_q_{j}': tq for j, tq in enumerate(tau_q_list)}}
    data_tau_q.append(row)

df_tau_q = pd.DataFrame(data_tau_q)
df_tau_q.to_csv("tau_q_vs_q_BA.csv", index=False)

df_h2 = pd.DataFrame({'Time_Step': range(0, len(h_2_values) * 100, 100), 'h_2': h_2_values})
df_h2.to_csv("h_2_values_BA.csv", index=False)

print("Multifractal spectrum saved to multifractal_spectrum_BA.csv")
print("tau_q vs q values saved to tau_q_vs_q_BA.csv")
print("h_2 values saved to h_2_values_BA.csv")

csv_BAvalues = "simulation_results_BA.csv"
df = pd.DataFrame({
    "Time": t,
    "Avg_Cooperator_Fraction": average_cooperator_fraction_over_time,
    "Avg_Resource_Level": average_resource_levels_over_time,
    "Avg_Dg": average_Dg_over_time,
    "Avg_Dr": average_Dr_over_time
})
df.to_csv(csv_BAvalues, index=False)

print(f"Simulation results saved to {csv_BAvalues}")

plt.figure(figsize=(12, 15))

plt.subplot(5, 1, 1)
plt.plot(t, average_cooperator_fraction_over_time, color='b', linestyle='-', label="Average Cooperative Fraction")
plt.xlabel('Time')
plt.ylabel('Average Cooperator Fraction')
plt.title('Average Evolution of Cooperative Fraction Across Episodes')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 2)
plt.plot(t, average_resource_levels_over_time, color='g', linestyle='-', label="Resource Level (m)")
plt.xlabel('Time')
plt.ylabel('Resource Level (m)')
plt.title('Resource Level Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 3)
plt.plot(t, average_Dg_over_time, color='r', linestyle='--', label="Dilemma Strength (Dg)")
plt.plot(t, average_Dr_over_time, color='purple', linestyle=':', label="Dilemma Strengths (Dg, Dr)")
plt.xlabel('Time')
plt.ylabel('Dilemma Strengths (Dg, Dr)')
plt.title('Dilemma Strengths Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 4)
plt.plot(average_cooperator_fraction_over_time, average_resource_levels_over_time, color='g', linestyle='-', label="evolving path")
plt.xlabel('Frequency of cooperator')
plt.ylabel('Resource Level (m)')
plt.title('relation of evolution and feedback')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 5)
# Plot the multifractal spectrum for the last episode
plt.plot(alpha_values[-1], f_alpha_values[-1], 'b-')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f(\alpha)$')
plt.title('Multifractal Spectrum (MF-DFA)')
plt.grid(True)

plt.tight_layout()
plt.show()



num_surrogates = 50

original_time_series = np.array(cooperator_fraction_history_across_time[-1])
h_q_shuffled_surrogates = np.zeros((num_surrogates, len(q_values)))
h_q_phase_surrogates = np.zeros((num_surrogates, len(q_values)))
F_q_original, h_q_original, _, _, _ = mdfa(original_time_series, q_values, scales, order)

start_time = time.time() 

for i in range(num_surrogates):
    shuffled_time_series = shuffle_time_series(original_time_series)
    _, h_q_shuffled_surrogates[i], _, _, _ = mdfa(shuffled_time_series, q_values, scales, order)

for i in range(num_surrogates):
    phase_randomized_time_series = phase_randomized_surrogate(original_time_series)
    _, h_q_phase_surrogates[i], _, _, _ = mdfa(phase_randomized_time_series, q_values, scales, order)

end_time = time.time() 
total_time = end_time - start_time
print(f"Time taken to generate and analyze {num_surrogates} surrogates: {total_time:.2f} seconds")

h_q_shuffled_mean = np.mean(h_q_shuffled_surrogates, axis=0)
h_q_shuffled_std = np.std(h_q_shuffled_surrogates, axis=0)
h_q_phase_mean = np.mean(h_q_phase_surrogates, axis=0)
h_q_phase_std = np.std(h_q_phase_surrogates, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(q_values, h_q_original, marker='o', label='Original h(q)')
plt.plot(q_values, h_q_shuffled_mean, linestyle='--', color='orange', label='Shuffled Mean')
plt.fill_between(q_values, h_q_shuffled_mean - h_q_shuffled_std, h_q_shuffled_mean + h_q_shuffled_std,
                 color='orange', alpha=0.2, label='Shuffled +/- STD')
plt.plot(q_values, h_q_phase_mean, linestyle='--', color='green', label='Phase Rand Mean')
plt.fill_between(q_values, h_q_phase_mean - h_q_phase_std, h_q_phase_mean + h_q_phase_std,
                 color='green', alpha=0.2, label='Phase Rand +/- STD')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.title('Comparison of h(q) for Original and Surrogate Time Series')
plt.legend()
plt.grid(True)
plt.show()



#WS
import numpy as np
import pandas as pd
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import detrend
import time

R0, S0, T0, P0 = 5.0, 1.0, 3.0, 0.0
R1, S1, T1, P1 = 3.0, 0.0, 5.0, 1.0
theta = 2.0
epsilon = 0.1
beta = 10 

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
    def __init__(self, population_size, average_degree, network_type):
        self.size = population_size
        self.network_type = network_type
        self.average_degree = average_degree
        self.agents = self.generate_agents()
        self.topology = self.create_topology()
        self.connect_agents()
        self.snapshots = [] 

    def create_topology(self):
        if self.network_type == 'WS':
            return nx.watts_strogatz_graph(self.size, self.average_degree, 0.05)
        else:
            raise ValueError("Unsupported network type")

    def connect_agents(self):
        for focal in self.agents:
            neighbors = list(self.topology.neighbors(focal.id))
            focal.neighbors_id = neighbors

    def generate_agents(self):
        return [Agent(id_num) for id_num in range(self.size)]

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

def shuffle_time_series(time_series):
    shuffled_series = time_series.copy()
    np.random.shuffle(shuffled_series)
    return shuffled_series

def phase_randomized_surrogate(time_series):
    n = len(time_series)
    fft_coeffs = np.fft.fft(time_series)
    num_pos = (n - 1) // 2  
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, num_pos))

    if n % 2 == 0:  
        phases = np.concatenate(([1], random_phases, [1], np.conjugate(random_phases[::-1])))
    else: 
        phases = np.concatenate(([1], random_phases, np.conjugate(random_phases[::-1])))

    surrogate_fft = fft_coeffs * phases
    surrogate_signal = np.fft.ifft(surrogate_fft).real

    return surrogate_signal

def mdfa(time_series, q_values, scales, order=1):
   
    N = len(time_series)
    F_q = np.zeros((len(q_values), len(scales)))
    Y = np.cumsum(time_series - np.mean(time_series))
    for i, q in enumerate(q_values):
        for j, s in enumerate(scales):
            Ns = N // s 
            F_qs = np.zeros(Ns * 2) 
            
            for v in range(Ns):
                segment = Y[v * s:(v + 1) * s]
                trend = detrend(segment, type='linear') 
                F_qs[v] = np.sqrt(np.mean(trend**2)) 

            for v in range(Ns):
                segment = Y[N-(v+1)*s:N-v*s]
                trend = detrend(segment, type='linear') 
                F_qs[Ns + v] = np.sqrt(np.mean(trend**2)) 

            if q == 0:
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(F_qs**2)))
            else:
                F_q[i, j] = (np.mean(F_qs**q))**(1/q)

    log_s = np.log(scales)
    h_q = np.zeros(len(q_values)) 
    t_q = np.zeros(len(q_values)) 

    for i in range(len(q_values)):
        log_F_q = np.log(F_q[i, :])
        coeffs = np.polyfit(log_s, log_F_q, 1)
        h_q[i] = coeffs[0]
        t_q[i] = q_values[i] * h_q[i] - 1

    alpha = np.diff(t_q) / np.diff(q_values)
    alpha = (alpha[:-1] + alpha[1:]) / 2 
    f_alpha = q_values[1:-1] * alpha - t_q[1:-1]

    return F_q, h_q, t_q, alpha, f_alpha

num_agent = 10000
num_episodes = 1
average_degree = 6
initial_fraction = 0.5 

t = np.linspace(0, 5000, 5000)

q_values = np.linspace(-5, 5, 21) 
scales = np.unique(np.logspace(np.log10(5), np.log10(len(t) // 10), num=15, dtype=int)) 
order = 1 

cooperator_fraction_history_across_time = []
resource_levels_across_time = []
alpha_values = []
f_alpha_values = []
tau_q_values = []
h_2_values = []

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}/{num_episodes}")
    society = Society(num_agent, average_degree, 'WS')
    init_cooperators = choose_init_c(num_agent, initial_fraction)
    society.agents = init_strategy(society.agents, init_cooperators)
    decision_maker = Decision(beta=10)

    episode_cooperator_fractions = []
    episode_resource_levels = []

    m = 0.4 

    for time_step in tqdm(range(len(t)), desc=f"Episode {episode + 1}", leave=False):
        current_time = t[time_step]

        x_ws = society.count_fraction()
        dm = epsilon * m * (1 - m) * ((1 + theta) * x_ws - 1)
        m += dm
        m = np.clip(m, 0, 1)  # Ensure m stays in [0, 1]
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        decision_maker.set_dilemma_strengths(Dg_prime, Dr_prime)

        society.agents = decision_maker.update_strategy(society.agents)
        cooperator_fraction = society.count_fraction()
        episode_cooperator_fractions.append(cooperator_fraction)
        episode_resource_levels.append(m)

        if time_step % 100 == 0:
            ts = np.array(episode_cooperator_fractions)
            F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
            tau_q_values.append(t_q)
            h_2 = np.interp(2, q_values, h_q)
            h_2_values.append(h_2)

            df_original = pd.DataFrame({'q': q_values, 'h_q': h_q.tolist()})
            df_original.to_csv(f"h_q_vs_q_original_t{time_step}.csv", index=False)

            shuffled_series = shuffle_time_series(ts)
            _, h_q_shuffled, *_ = mdfa(shuffled_series, q_values, scales, order)
            df_shuffled = pd.DataFrame({'q': q_values, 'h_q': h_q_shuffled.tolist()})
            df_shuffled.to_csv(f"h_q_vs_q_shuffled_t{time_step}.csv", index=False)

            surrogate_series = phase_randomized_surrogate(ts)
            _, h_q_surrogate, *_ = mdfa(surrogate_series, q_values, scales, order)
            df_surrogate = pd.DataFrame({'q': q_values, 'h_q': h_q_surrogate.tolist()})
            df_surrogate.to_csv(f"h_q_vs_q_surrogate_t{time_step}.csv", index=False)

    cooperator_fraction_history_across_time.append(episode_cooperator_fractions)
    resource_levels_across_time.append(episode_resource_levels)

    ts = np.array(episode_cooperator_fractions)
    F_q, h_q, t_q, alpha, f_alpha = mdfa(ts, q_values, scales, order)
    alpha_values.append(alpha)
    f_alpha_values.append(f_alpha)

Dg_history_across_time = []
Dr_history_across_time = []

for episode_resource_levels in resource_levels_across_time:
    episode_Dg = []
    episode_Dr = []
    for m in episode_resource_levels:
        A_evo = A(m)
        Dg_prime, Dr_prime = calculate_dilemma_strengths(A_evo)
        episode_Dg.append(Dg_prime)
        episode_Dr.append(Dr_prime)
    Dg_history_across_time.append(episode_Dg)
    Dr_history_across_time.append(episode_Dr)

average_cooperator_fraction_over_time = np.mean(np.array(cooperator_fraction_history_across_time), axis=0)
average_resource_levels_over_time = np.mean(np.array(resource_levels_across_time), axis=0)
average_Dg_over_time = np.mean(np.array(Dg_history_across_time), axis=0)
average_Dr_over_time = np.mean(np.array(Dr_history_across_time), axis=0)

df_mf = pd.DataFrame({'alpha': alpha_values[-1], 'f_alpha': f_alpha_values[-1]})
df_mf.to_csv("multifractal_spectrum_WS.csv", index=False)

q_values_list = q_values.tolist()
data_tau_q = []

for i, tau_q in enumerate(tau_q_values):
    time_step = i * 100  # Calculate the corresponding time step
    tau_q_list = tau_q.tolist()  # Convert tau_q array to list
    # Create a dictionary for each row with q_values and tau_q data
    row = {'Time_Step': time_step, **{f'q_{j}': q for j, q in enumerate(q_values_list)}, **{f'tau_q_{j}': tq for j, tq in enumerate(tau_q_list)}}
    data_tau_q.append(row)

df_tau_q = pd.DataFrame(data_tau_q)
df_tau_q.to_csv("tau_q_vs_q_WS.csv", index=False)

df_h2 = pd.DataFrame({'Time_Step': range(0, len(h_2_values) * 100, 100), 'h_2': h_2_values})
df_h2.to_csv("h_2_values_WS.csv", index=False)

print("Multifractal spectrum saved to multifractal_spectrum.csv")
print("tau_q vs q values saved to tau_q_vs_q.csv")
print("h_2 values saved to h_2_values_Lattice.csv")

csv_wsvalues = "simulation_results_WS.csv"
df = pd.DataFrame({
    "Time": t,
    "Avg_Cooperator_Fraction": average_cooperator_fraction_over_time,
    "Avg_Resource_Level": average_resource_levels_over_time,
    "Avg_Dg": average_Dg_over_time,
    "Avg_Dr": average_Dr_over_time
})
df.to_csv(csv_wsvalues, index=False)

print(f"Simulation results saved to {csv_wsvalues}")

plt.figure(figsize=(12, 15))

plt.subplot(5, 1, 1)
plt.plot(t, average_cooperator_fraction_over_time, color='b', linestyle='-', label="Average Cooperative Fraction")
plt.xlabel('Time')
plt.ylabel('Average Cooperator Fraction')
plt.title('Average Evolution of Cooperative Fraction Across Episodes')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 2)
plt.plot(t, average_resource_levels_over_time, color='g', linestyle='-', label="Resource Level (m)")
plt.xlabel('Time')
plt.ylabel('Resource Level (m)')
plt.title('Resource Level Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 3)
plt.plot(t, average_Dg_over_time, color='r', linestyle='--', label="Dilemma Strength (Dg)")
plt.plot(t, average_Dr_over_time, color='purple', linestyle=':', label="Dilemma Strength (Dr)")
plt.xlabel('Time')
plt.ylabel('Dilemma Strengths (Dg, Dr)')
plt.title('Dilemma Strengths Over Time')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 4)
plt.plot(average_cooperator_fraction_over_time, average_resource_levels_over_time, color='g', linestyle='-', label="evolving path")
plt.xlabel('Frequency of cooperator')
plt.ylabel('Resource Level (m)')
plt.title('relation of evolution and feedback')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(5, 1, 5)
# Plot the multifractal spectrum for the last episode
plt.plot(alpha_values[-1], f_alpha_values[-1], 'b-')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$f(\alpha)$')
plt.title('Multifractal Spectrum (MF-DFA)')
plt.grid(True)

plt.tight_layout()
plt.show()


num_surrogates = 50

original_time_series = np.array(cooperator_fraction_history_across_time[-1])
h_q_shuffled_surrogates = np.zeros((num_surrogates, len(q_values)))
h_q_phase_surrogates = np.zeros((num_surrogates, len(q_values)))
F_q_original, h_q_original, _, _, _ = mdfa(original_time_series, q_values, scales, order)
start_time = time.time()  

for i in range(num_surrogates):
    shuffled_time_series = shuffle_time_series(original_time_series)
    _, h_q_shuffled_surrogates[i], _, _, _ = mdfa(shuffled_time_series, q_values, scales, order)

for i in range(num_surrogates):
    phase_randomized_time_series = phase_randomized_surrogate(original_time_series)
    _, h_q_phase_surrogates[i], _, _, _ = mdfa(phase_randomized_time_series, q_values, scales, order)

end_time = time.time()  
total_time = end_time - start_time
print(f"Time taken to generate and analyze {num_surrogates} surrogates: {total_time:.2f} seconds")

h_q_shuffled_mean = np.mean(h_q_shuffled_surrogates, axis=0)
h_q_shuffled_std = np.std(h_q_shuffled_surrogates, axis=0)
h_q_phase_mean = np.mean(h_q_phase_surrogates, axis=0)
h_q_phase_std = np.std(h_q_phase_surrogates, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(q_values, h_q_original, marker='o', label='Original h(q)')
plt.plot(q_values, h_q_shuffled_mean, linestyle='--', color='orange', label='Shuffled Mean')
plt.fill_between(q_values, h_q_shuffled_mean - h_q_shuffled_std, h_q_shuffled_mean + h_q_shuffled_std,
                 color='orange', alpha=0.2, label='Shuffled +/- STD')
plt.plot(q_values, h_q_phase_mean, linestyle='--', color='green', label='Phase Rand Mean')
plt.fill_between(q_values, h_q_phase_mean - h_q_phase_std, h_q_phase_mean + h_q_phase_std,
                 color='green', alpha=0.2, label='Phase Rand +/- STD')
plt.xlabel('q')
plt.ylabel('h(q)')
plt.title('Comparison of h(q) for Original and Surrogate Time Series')
plt.legend()
plt.grid(True)
plt.show()
