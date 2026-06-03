import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- Entorno con Tres Sensores ---
class TresSensoresEnv(gym.Env):
    def __init__(self, distancia_objetivo=30):
        super(TresSensoresEnv, self).__init__()
        self.distancia_objetivo = distancia_objetivo
        # Distancias iniciales para los tres sensores (izquierdo, frente, derecho)
        self.distancias = np.random.randint(0, 100, size=3).astype(np.float64)
        self.action_space = spaces.Discrete(5) # 0: avanzar, 1: retroceder, 2: girar izquierda, 3: girar derecha, 4: mantener
        # Cada sensor tiene 4 estados discretos
        self.observation_space = spaces.MultiDiscrete([4, 4, 4])

    def reset(self, seed=None, options=None):
        self.distancias = np.random.randint(0, 100, size=3).astype(np.float64)
        return self._obtener_estado(), {}

    def _obtener_estado(self):
        estado = []
        for distancia in self.distancias:
            if distancia >= 40:
                estado.append(0) # Muy lejos
            elif distancia >= 20:
                estado.append(1) # Lejos
            elif distancia >= 5:
                estado.append(2) # Cerca
            else:
                estado.append(3) # Muy cerca
        return np.array(estado, dtype=np.int32)

    def step(self, action):
        # Simular cambio en las distancias con ruido
        ruido = np.random.normal(loc=0, scale=2, size=3)

        if action == 0: # Avanzar
            self.distancias -= np.array([2, 5, 2]) + ruido
        elif action == 1: # Retroceder
            self.distancias += np.array([2, 5, 2]) + ruido
        elif action == 2: # Girar izquierda
            self.distancias[0] += 3 + ruido[0] # Sensor izquierdo se aleja
            self.distancias[2] -= 3 + ruido[2] # Sensor derecho se acerca
        elif action == 3: # Girar derecha
            self.distancias[0] -= 3 + ruido[0] # Sensor izquierdo se acerca
            self.distancias[2] += 3 + ruido[2] # Sensor derecho se aleja
        # Mantener no hace nada (action == 4)

        # Asegurar que las distancias estén en un rango realista
        self.distancias = np.clip(self.distancias, 0, 100)

        # Calcular recompensa basada en el sensor frontal
        distancia_frontal = self.distancias[1]
        reward = -abs(distancia_frontal - self.distancia_objetivo)

        # Verificar si se alcanzó el objetivo (±5 unidades en el sensor frontal)
        done = abs(distancia_frontal - self.distancia_objetivo) <= 5

        return self._obtener_estado(), reward, done, False, {}

# --- Red Neuronal para DQN ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Agente DQN ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# --- Entrenamiento DQN ---
def entrenar_dqn(env, episodios=1000, batch_size=32, archivo_modelo="dqn_tres_sensores.pth"):
    state_size = 3 # Tres sensores, cada uno con un estado discreto (0-3)
    action_size = 5 # Cinco acciones posibles
    agent = DQNAgent(state_size, action_size)

    if os.path.exists(archivo_modelo):
        agent.load(archivo_modelo)
        print("Cargando modelo preentrenado...")

    for episodio in range(episodios):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if done:
            print(f"Episodio: {episodio + 1}, Recompensa total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        agent.replay(batch_size)

        if episodio % 100 == 0:
            agent.save(archivo_modelo)
            print(f"Modelo guardado en {archivo_modelo}")

    return agent

# --- Prueba del Agente DQN ---
def probar_dqn(env, agent):
    state, _ = env.reset()
    done = False

    print("--- Prueba del Agente DQN ---")
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        print(f"Estado: {next_state}, Acción: {action}, Recompensa: {reward:.2f}")
        state = next_state

# --- Ejecución ---
if __name__ == "__main__":
    env = TresSensoresEnv(distancia_objetivo=30)
    agent = entrenar_dqn(env, episodios=1000, archivo_modelo="dqn_tres_sensores.pth")
    probar_dqn(env, agent)
