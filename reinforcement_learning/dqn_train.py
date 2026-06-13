"""
DQN para el levitador vertical (1 sensor, 6 valores PWM).
Entrena en PC con un simulador 1D del tubo y exporta los pesos
para inferencia en el ESP32.

Compara la politica aprendida por la DQN contra la tabla Q
del agente tabular (mismo espacio de estados 11x6 usado por
aprendizaje1.py en el ESP32).
"""
import os
import csv
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 1. Entorno simulado del levitador (1D)
# ============================================================
class LevitadorEnv:
    """
    Simulador 1D de la pelota de icopor en el tubo de acrilico.

    Modelo: sistema de primer orden. Cada PWM tiene una posicion de
    equilibrio p_eq(PWM) hacia la que la pelota converge con una
    constante de tiempo TAU. Esto aproxima el comportamiento real del
    tubo, donde para una PWM fija la pelota se estabiliza a una altura.

    Estado: [position, velocity]
        - position: cm desde el sensor (arriba). 0 = pegada al sensor,
          40 = fondo del tubo.
        - velocity: cm/s, positivo = cayendo, negativo = subiendo.

    Setpoint por defecto: 15 cm.
    Accion: indice 0..5 -> PWM [275, 330, 385, 440, 520, 600].

    Dinamica:
        p_eq = 25 - 20 * (pwm - 200) / 400   (lineal: PWM 275 -> 21.25 cm,
                                              PWM 600 ->  5 cm)
        pos += (p_eq - pos) * DT / TAU
        vel  = (pos - pos_prev) / DT
        pos += ruido gaussiano (sigma = NOISE_STD)
    """
    POS_MIN, POS_MAX = 1.0, 39.0
    SETPOINT = 15.0
    ACTIONS = np.array([275, 330, 410, 490, 600, 750], dtype=np.float32)
    DT = 0.05
    TAU = 0.35
    NOISE_STD = 0.15
    MAX_STEPS = 150

    BIN_CENTERS = np.arange(10.0, 21.0)

    def __init__(self, setpoint=15.0, seed=None):
        self.setpoint = setpoint
        self.rng = np.random.default_rng(seed)
        self.pos = 15.0
        self.vel = 0.0
        self.t = 0

    @staticmethod
    def equilibrium(pwm):
        return 25.0 - 20.0 * (pwm - 200.0) / 400.0

    def reset(self):
        self.pos = float(self.rng.uniform(12.0, 18.0))
        self.vel = 0.0
        self.t = 0
        return self._state()

    def _state(self):
        return np.array([self.pos, self.vel], dtype=np.float32)

    def step(self, action):
        pwm = float(self.ACTIONS[int(action)])
        p_eq = self.equilibrium(pwm)
        prev = self.pos
        self.pos += (p_eq - self.pos) * self.DT / self.TAU
        self.pos += float(self.rng.normal(0.0, self.NOISE_STD))
        self.vel = (self.pos - prev) / self.DT
        self.t += 1

        err = abs(self.pos - self.setpoint)
        r = -0.3 * err
        if err < 1.0:
            r += 3.0
        if err < 0.3:
            r += 4.0

        done = (self.pos < self.POS_MIN or self.pos > self.POS_MAX
                or self.t >= self.MAX_STEPS)
        return self._state(), float(r), bool(done), False, {}

    @classmethod
    def discretize(cls, pos):
        return int(np.argmin(np.abs(cls.BIN_CENTERS - pos)))


# ============================================================
# 2. Q-Learning tabular (baseline, mismo espacio 11x6)
# ============================================================
def train_qlearning(env, episodes=1500, alpha=0.10, gamma=0.90,
                    eps_start=0.5, eps_end=0.01, eps_decay=0.995):
    NUM_STATES = 11
    NUM_ACTIONS = 6
    Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
    eps = eps_start
    rewards = []
    for ep in range(episodes):
        env.reset()
        s = env.discretize(env.pos)
        total_r = 0.0
        for _ in range(env.MAX_STEPS):
            if np.random.rand() < eps:
                a = np.random.randint(NUM_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))
            _, r, done, _, _ = env.step(a)
            if done:
                target = 0.0
            else:
                s_next = env.discretize(env.pos)
                target = np.max(Q[s_next])
                s = s_next
            Q[s, a] = Q[s, a] + alpha * (r + gamma * target - Q[s, a])
            total_r += r
            if done:
                break
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards


# ============================================================
# 3. Normalizacion de estado (compartida por entrenamiento e inference)
# ============================================================
POS_NORM = 40.0
VEL_NORM = 200.0

def normalize_state(s):
    p = float(np.clip(s[0] / POS_NORM, 0.0, 1.0))
    v = float(np.clip(s[1] / VEL_NORM, -1.0, 1.0))
    return np.array([p, v], dtype=np.float32)


# ============================================================
# 4. Red DQN y agente
# ============================================================
class DQNNet(nn.Module):
    def __init__(self, state_dim=2, action_dim=6, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch):
        b = random.sample(self.buf, batch)
        s, a, r, s2, d = zip(*b)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_dqn(env, episodes=800, gamma=0.95, lr=1e-3,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              batch_size=32, target_update=10, buffer_cap=2000,
              seed=0, verbose=True, fuzzy_seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    net = DQNNet()
    target = DQNNet()
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buf = ReplayBuffer(buffer_cap)

    # Sembrar replay buffer con transiciones expertas de lógica difusa
    if fuzzy_seed:
        for trans in fuzzy_seed:
            buf.push(*trans)
        print(f"  Replay buffer sembrado con {len(fuzzy_seed)} transiciones expertas fuzzy.")

    eps = eps_start
    rewards = []
    moving = []

    for ep in range(episodes):
        s_raw = env.reset()
        s = normalize_state(s_raw)
        total_r = 0.0
        for _ in range(env.MAX_STEPS):
            if np.random.rand() < eps:
                a = np.random.randint(6)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(net(torch.from_numpy(s))).item())
            s2_raw, r, done, _, _ = env.step(a)
            s2 = normalize_state(s2_raw)
            buf.push(s, a, r, s2, float(done))
            s = s2
            total_r += r

            if len(buf) >= batch_size:
                bs, ba, br, bs2, bd = buf.sample(batch_size)
                q = net(torch.from_numpy(bs)).gather(1, torch.from_numpy(ba).unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(torch.from_numpy(bs2)).max(1)[0]
                    target_q = torch.from_numpy(br) + (1.0 - torch.from_numpy(bd)) * gamma * q_next
                loss = loss_fn(q, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if done:
                break

        rewards.append(total_r)
        moving.append(np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards))
        eps = max(eps_end, eps * eps_decay)
        if (ep + 1) % target_update == 0:
            target.load_state_dict(net.state_dict())
        if verbose and (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d} | reward {total_r:7.2f} | "
                  f"media50 {moving[-1]:7.2f} | eps {eps:.3f}")

    return net, rewards, moving


# ============================================================
# 5. Comparacion contra la tabla Q
# ============================================================
def evaluate_policy(env, policy_fn, n_episodes=200):
    rewards, errors, steps_ok = [], [], []
    for _ in range(n_episodes):
        env.reset()
        total_r = 0.0
        errs = []
        ok = 0
        for t in range(env.MAX_STEPS):
            a = policy_fn(env)
            _, r, done, _, _ = env.step(a)
            total_r += r
            errs.append(abs(env.pos - env.SETPOINT))
            if abs(env.pos - env.SETPOINT) < 1.0:
                ok += 1
            if done:
                break
        rewards.append(total_r)
        errors.append(float(np.mean(errs)))
        steps_ok.append(ok / max(1, len(errs)))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_error_cm": float(np.mean(errors)),
        "frac_steps_within_1cm": float(np.mean(steps_ok)),
    }


def compare_policies(Q, net, seed=1, n_episodes=200):
    def ql_policy(env):
        return int(np.argmax(Q[env.discretize(env.pos)]))

    def dqn_policy(env):
        s = normalize_state(env._state())
        with torch.no_grad():
            return int(torch.argmax(net(torch.from_numpy(s))).item())

    env = LevitadorEnv(seed=seed)
    ql_stats = evaluate_policy(env, ql_policy, n_episodes)
    env = LevitadorEnv(seed=seed)
    dqn_stats = evaluate_policy(env, dqn_policy, n_episodes)
    return ql_stats, dqn_stats


def heuristic_policy(env):
    """Regla de control proporcional discreta (sirve solo para validar el env)."""
    err = env.pos - env.SETPOINT
    if err > 4:
        return 5
    elif err > 1.5:
        return 4
    elif err > 0.3:
        return 3
    elif err > -0.3:
        return 2
    elif err > -1.5:
        return 1
    else:
        return 0


def policy_agreement(Q, net):
    """Para los 11 estados discretos con vel=0, compara argmax(Q) vs argmax(DQN)."""
    matches = 0
    rows = []
    for i, c in enumerate(LevitadorEnv.BIN_CENTERS):
        s = normalize_state(np.array([c, 0.0], dtype=np.float32))
        with torch.no_grad():
            q_net = net(torch.from_numpy(s)).numpy()
        q_tab = Q[i]
        a_tab = int(np.argmax(q_tab))
        a_net = int(np.argmax(q_net))
        ok = a_tab == a_net
        matches += int(ok)
        rows.append((int(i), float(c), a_tab, a_net, bool(ok)))
    return matches / len(rows), rows


# ============================================================
# 6a. Carga de transiciones expertas desde los CSVs de lógica difusa
# ============================================================
_FUZZY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logica_difusa")
_FUZZY_CSVS = [
    os.path.join(_FUZZY_DIR, "datos_esp32_centroide.csv"),
    os.path.join(_FUZZY_DIR, "datos_esp32_bisector.csv"),
    os.path.join(_FUZZY_DIR, "datos_esp32_mom.csv"),
]


def _nearest_action_idx(pwm_val):
    """Índice de la acción PWM más cercana al valor dado."""
    actions = LevitadorEnv.ACTIONS
    return int(np.argmin(np.abs(actions - pwm_val)))


def load_fuzzy_transitions():
    """
    Carga los 3 CSVs de lógica difusa y devuelve una lista de transiciones
    (s, a, r, s2, done) listas para sembrar un ReplayBuffer.

    Columnas CSV: tiempo, distancia, setpoint, error, derivada, integral,
                  delta_pwm, pwm
    - state  = [distancia/POS_NORM, derivada/VEL_NORM]
    - action = índice de la acción PWM más cercana al 'pwm' actual
    - reward = -0.3*|error| + 3*(|error|<1) + 4*(|error|<0.3)
    - s2     = state del siguiente paso (misma sesión)
    """
    transitions = []
    for fcsv in _FUZZY_CSVS:
        fcsv = os.path.normpath(fcsv)
        if not os.path.isfile(fcsv):
            print(f"  [INFO] Fuzzy CSV no encontrado: {os.path.basename(fcsv)}")
            continue
        rows = []
        with open(fcsv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'dist':   float(row['distancia']),
                        'deriv':  float(row['derivada']),
                        'error':  float(row['error']),
                        'pwm':    float(row['pwm']),
                    })
                except (KeyError, ValueError):
                    continue
        # Construir transiciones de paso consecutivo
        for i in range(len(rows) - 1):
            r0, r1 = rows[i], rows[i + 1]
            s  = normalize_state(np.array([r0['dist'], r0['deriv']], dtype=np.float32))
            s2 = normalize_state(np.array([r1['dist'], r1['deriv']], dtype=np.float32))
            a  = _nearest_action_idx(r0['pwm'])
            e  = abs(r0['error'])
            rew = -0.3 * e
            if e < 1.0: rew += 3.0
            if e < 0.3: rew += 4.0
            transitions.append((s, a, float(rew), s2, 0.0))
        print(f"  + {os.path.basename(fcsv)}: {len(rows)-1} transiciones expertas")
    return transitions


def init_qtable_from_fuzzy(Q, bin_centers, n_actions):
    """
    Pre-inicializa la Q-table usando los datos reales del hardware.
    Para cada transición fuzzy, refuerza Q[estado][acción_aplicada] con
    la recompensa observada (actualización de una pasada, α=1).
    """
    transitions = load_fuzzy_transitions()
    # Acumular recompensas por (estado, acción) y promediar
    acc   = np.zeros((len(bin_centers), n_actions), dtype=np.float32)
    count = np.zeros((len(bin_centers), n_actions), dtype=np.int32)
    for s, a, r, s2, _ in transitions:
        pos_cm = s[0] * POS_NORM
        si = int(np.argmin(np.abs(bin_centers - pos_cm)))
        acc[si, a]   += r
        count[si, a] += 1
    # Escribir solo celdas con al menos 1 muestra
    mask = count > 0
    Q[mask] = acc[mask] / count[mask]
    n_filled = int(mask.sum())
    print(f"  Q-table pre-inicializada: {n_filled}/{Q.size} celdas con datos reales")
    return Q


# ============================================================
# 6. Plot
# ============================================================
def smooth(y, w=50):
    y = np.asarray(y, dtype=np.float32)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def plot_results(ql_rewards, dqn_rewards, dqn_moving, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ql_rewards, alpha=0.25, color="C0", label="Q-Learning")
    axes[0].plot(smooth(ql_rewards), color="C0", label="Q-Learning (media 50)")
    axes[0].plot(dqn_rewards, alpha=0.25, color="C1", label="DQN")
    axes[0].plot(dqn_moving, color="C1", label="DQN (media 50)")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Recompensa total")
    axes[0].set_title("Curva de aprendizaje")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(
        [ql_rewards[-100:], dqn_rewards[-100:]],
        labels=["Q-Learning", "DQN"],
    )
    axes[1].set_ylabel("Recompensa total (ultimos 100 ep.)")
    axes[1].set_title("Distribucion final")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot guardado en {out_path}")


# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures"), exist_ok=True)

    print("=" * 60)
    print("0) Cargando transiciones expertas de lógica difusa")
    print("=" * 60)
    fuzzy_transitions = load_fuzzy_transitions()
    print(f"  Total transiciones expertas cargadas: {len(fuzzy_transitions)}")

    print()
    print("=" * 60)
    print("0b) Test de politica heuristica (validacion del entorno)")
    print("=" * 60)
    env = LevitadorEnv(seed=0)
    h_stats = evaluate_policy(env, heuristic_policy, n_episodes=100)
    print(f"  Heuristica  : mean_reward={h_stats['mean_reward']:7.2f} | "
          f"err={h_stats['mean_error_cm']:.2f} cm | "
          f"dentro1cm={h_stats['frac_steps_within_1cm']*100:.1f}%")

    print()
    print("=" * 60)
    print("1) Q-Learning tabular (baseline 11x6) — pre-inicializado con fuzzy")
    print("=" * 60)
    # Estados del Q-learning real del ESP32 (aprendizaje1.py) — rango físico completo
    QL_STATES = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 26], dtype=np.float32)
    Q_init = np.zeros((11, 6), dtype=np.float32)
    if fuzzy_transitions:
        Q_init = init_qtable_from_fuzzy(Q_init, QL_STATES, 6)
    env = LevitadorEnv(seed=42)
    Q, ql_rewards = train_qlearning(env, episodes=1500)
    # Blend: arrancar con Q pre-inicializado (ya está en Q_init, train_qlearning
    # parte desde cero internamente — re-entrenamos con warm start)
    if fuzzy_transitions:
        Q_warm = Q_init.copy()
        env2 = LevitadorEnv(seed=42)
        eps = 0.5
        alpha, gamma = 0.10, 0.90
        for ep in range(1500):
            env2.reset()
            s = env2.discretize(env2.pos)
            for _ in range(env2.MAX_STEPS):
                if np.random.rand() < eps:
                    a = np.random.randint(6)
                else:
                    a = int(np.argmax(Q_warm[s]))
                _, r, done, _, _ = env2.step(a)
                s_next = env2.discretize(env2.pos)
                target = 0.0 if done else float(np.max(Q_warm[s_next]))
                Q_warm[s, a] += alpha * (r + gamma * target - Q_warm[s, a])
                s = s_next
                if done:
                    break
            eps = max(0.01, eps * 0.995)
        Q = Q_warm
        print("  Q-table entrenada con warm start fuzzy.")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable_levitador.npy"), Q)
    print("Q-table final (filas = estado 10..20 cm, cols = accion):")
    print(np.round(Q, 2))

    print()
    print("=" * 60)
    print("2) DQN (2 -> 24 -> 24 -> 6) — replay buffer sembrado con datos fuzzy")
    print("=" * 60)
    env = LevitadorEnv(seed=42)
    net, dqn_rewards, dqn_moving = train_dqn(
        env, episodes=800,
        fuzzy_seed=fuzzy_transitions if fuzzy_transitions else None
    )
    torch.save(net.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_model.pth"))

    print()
    print("=" * 60)
    print("3) Acuerdo de politicas (Q-table vs DQN en los 11 estados)")
    print("=" * 60)
    agree, rows = policy_agreement(Q, net)
    print(f"  Acuerdo argmax: {agree*100:.1f}%")
    print(f"  {'idx':>3} {'pos':>5} {'a_tab':>5} {'a_net':>5} {'match':>5}")
    for i, p, at, an, ok in rows:
        print(f"  {i:3d} {p:5.1f} {at:5d} {an:5d} {str(ok):>5}")

    print()
    print("=" * 60)
    print("4) Comparacion en simulacion (200 episodios, greedy)")
    print("=" * 60)
    ql_stats, dqn_stats = compare_policies(Q, net, seed=1, n_episodes=200)
    print(f"  Q-Learning : mean_reward={ql_stats['mean_reward']:7.2f} +- {ql_stats['std_reward']:.2f} | "
          f"err={ql_stats['mean_error_cm']:.2f} cm | dentro1cm={ql_stats['frac_steps_within_1cm']*100:.1f}%")
    print(f"  DQN        : mean_reward={dqn_stats['mean_reward']:7.2f} +- {dqn_stats['std_reward']:.2f} | "
          f"err={dqn_stats['mean_error_cm']:.2f} cm | dentro1cm={dqn_stats['frac_steps_within_1cm']*100:.1f}%")

    plot_results(ql_rewards, dqn_rewards, dqn_moving,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, "figures", "dqn_training.png"))
    print("\nListo. Archivos generados:")
    print("  qtable_levitador.npy")
    print("  dqn_model.pth")
    print("  figures/dqn_training.png")
