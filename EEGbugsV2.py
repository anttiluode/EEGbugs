import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk
import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
from math import cos, sin, radians, sqrt, atan2, degrees
import cv2
from typing import List, Tuple, Dict
import random
import threading
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent exceptions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame  # For audio playback

# Initialize pygame mixer for audio
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Configuration and Parameters
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 49),
    'low_gamma': (49, 60),
    'high_gamma': (60, 70)
}

latent_dim = 64
eeg_batch_size = 64
eeg_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

class EEGAutoencoder(nn.Module):
    def __init__(self, channels=5, frequency_bands=7, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2))
        )
        
        # This matches the saved state dimensions
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, latent_dim)  # 160 = 32 * 5 * 1
        self.fc2 = nn.Linear(latent_dim, 160)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1,2), stride=(1,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1,2), stride=(1,2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(-1, 32, 5, 1)  # Reshape to match decoder input
        x = self.decoder(x)
        x = x.squeeze(1)  # Remove channel dimension
        return x, latent


class BrainCoupler:
    def __init__(self, eeg_model, small_brain, coupling_rate=0.1):
        self.eeg_model = eeg_model
        self.small_brain = small_brain
        self.coupling_rate = coupling_rate
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(small_brain.parameters(), lr=0.001)
        
    def train_step(self, eeg_data, t):
        if len(eeg_data.shape) == 2:
            eeg_data = eeg_data.unsqueeze(0).unsqueeze(0)
            
        # Get EEG latent vector
        with torch.no_grad():
            _, eeg_latent = self.eeg_model(eeg_data)
        
        # Detach latent vector to prevent backward through EEG model
        eeg_latent = eeg_latent.detach()
        
        # Get small brain output
        brain_output = self.small_brain(eeg_latent, t)
        
        # Compute loss
        loss = self.loss_fn(brain_output, eeg_latent)
        
        # Backward pass with retain_graph
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss.item()

class SmallBrain(nn.Module):
    def __init__(self, num_neurons=16, latent_dim=64, coupling_strength=0.1):
        super().__init__()
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim
        
        # Wave parameters
        self.frequencies = nn.Parameter(torch.rand(num_neurons) * 2.0)
        self.phases = nn.Parameter(torch.rand(num_neurons) * 2 * np.pi)
        self.amplitudes = nn.Parameter(torch.rand(num_neurons) * 0.5 + 0.5)
        
        # Coupling matrix
        self.coupling = nn.Parameter(torch.randn(num_neurons, num_neurons) * coupling_strength)
        
        # Neural projections
        self.input_proj = nn.Linear(latent_dim, num_neurons)
        self.output_proj = nn.Linear(num_neurons, latent_dim)
        
        # Register state as a buffer to exclude it from gradient computations
        self.register_buffer('state', torch.zeros(num_neurons))
        self.memory = []
        
    def forward(self, x, t):
        # Handle device consistency
        if self.state.device != x.device:
            self.state = self.state.to(x.device)
        
        # Project input to neuron space
        neuron_input = self.input_proj(x)
        
        # Generate oscillations
        t_tensor = torch.tensor(t, dtype=torch.float32, device=x.device)
        oscillations = self.amplitudes * torch.sin(
            2 * np.pi * self.frequencies * t_tensor + self.phases
        ).unsqueeze(0).repeat(x.size(0), 1)
        
        # Update state without tracking gradients
        with torch.no_grad():
            self.state += 0.1 * (
                oscillations[0] + 
                torch.matmul(self.state, self.coupling) +
                neuron_input[0]
            )
            self.state = torch.tanh(self.state)
        
        # Project to output space
        output = self.output_proj(self.state.unsqueeze(0))
        
        # Update memory
        self.memory.append(self.state.clone())
        if len(self.memory) > 100:
            self.memory.pop(0)
        
        return output

    
class EEGWaveNeuron:
    def __init__(self, frequency=None, amplitude=None, phase=None, memory_size=100):
        self.frequency = frequency if frequency is not None else np.random.uniform(0.1, 1.0)
        self.amplitude = amplitude if amplitude is not None else np.random.uniform(0.5, 1.0)
        self.phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        self.output = 0.0
        self.memory = np.zeros(memory_size)
        self.memory_pointer = 0
        self.resonance_weight = 0.9

    def activate(self, input_signal, eeg_signal, t):
        eeg_influence = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) * eeg_signal
        past_resonance = np.mean(self.memory) * self.resonance_weight
        self.output = eeg_influence + input_signal + past_resonance
        self.memory[self.memory_pointer] = self.output
        self.memory_pointer = (self.memory_pointer + 1) % len(self.memory)
        return self.output

class ResonantBrain:
    def __init__(self, num_neurons=16, memory_size=100):
        self.neurons = [EEGWaveNeuron(memory_size=memory_size) for _ in range(num_neurons)]
        self.connections = self._initialize_connections()

    def _initialize_connections(self):
        connections = {}
        for n1 in self.neurons:
            for n2 in self.neurons:
                if n1 != n2:
                    connections[(n1, n2)] = np.random.uniform(0.1, 0.5)
        return connections

    def update(self, eeg_latent: np.ndarray, dt=0.1):
        # Update neurons
        for neuron, latent_value in zip(self.neurons, eeg_latent):
            input_signal = np.mean([
                self.connections.get((neuron, other), 0) * other.output
                for other in self.neurons if other != neuron
            ])
            neuron.activate(input_signal, latent_value, dt)

        # Hebbian Learning
        for (pre, post), weight in self.connections.items():
            if pre.output > 0.5 and post.output > 0.5:
                self.connections[(pre, post)] = min(weight + 0.01, 1.0)
            else:
                self.connections[(pre, post)] = max(weight - 0.01, 0.0)

class DynamicWaveEEGProcessor:
    def __init__(self, eeg_model_path: str, latent_dim=64, num_neurons=16):
        self.eeg_model = EEGAutoencoder(channels=5, frequency_bands=7, latent_dim=latent_dim)
        try:
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location='cpu', weights_only=True), strict=False)
        except TypeError:
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location='cpu'))
        self.eeg_model.eval()
        self.brain = ResonantBrain(num_neurons=num_neurons)
        self.time = 0.0
        self.memory = []

    def process_and_update(self, eeg_data: np.ndarray) -> np.ndarray:
        # Reshape eeg_data to correct dimensions [batch, channels, frequency_bands]
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.squeeze(0)  # Remove extra dimension if present
        if len(eeg_data.shape) == 2:
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected EEG data shape: {eeg_data.shape}")

        with torch.no_grad():
            _, latent_vector = self.eeg_model(eeg_tensor)
            
        latent_vector = latent_vector.squeeze(0).numpy()
        self.brain.update(latent_vector, dt=0.1)
        self.time += 0.1

        self.memory.append(latent_vector)
        if len(self.memory) > 50:
            self.memory.pop(0)

        return latent_vector

class Bug:
    def __init__(self, canvas_width: int, canvas_height: int, color: str, name: str, 
                 processor: DynamicWaveEEGProcessor, bug_radius: int = 20, num_waveneurons: int = 16):
        self.position = [random.randint(bug_radius, canvas_width - bug_radius),
                        random.randint(bug_radius, canvas_height - bug_radius)]
        self.direction = random.uniform(0, 360)
        self.speed = 5.0
        self.color = color
        self.name = name
        self.processor = processor
        self.bug_radius = bug_radius
        self.vision_angle = 90
        self.vision_range = 100
        self.state = "exploring"
        self.trail = []
        self.can_talk = False
        self.genetic_traits = self.initialize_genetic_traits()
        self.echo_trails = []
        self.num_waveneurons = num_waveneurons
        self.particles = []  # For particle effects
        self.sound_neuron_index = random.randint(0, num_waveneurons - 1)
        self.hearing_neuron_index = random.randint(0, num_waveneurons - 1)
        self.heard_sounds = []
        self.sound_channel = pygame.mixer.Channel(random.randint(0, 7))  # Assign each bug a different channel
        self.sound_frequency = None  # To store the current frequency
        self.last_sound_time = 0  # To control sound intervals
        self.sound_interval = random.uniform(1.0, 5.0)  # Bugs will make sounds every 1-5 seconds

    def initialize_genetic_traits(self) -> dict:
        return {
            'trail_thickness': random.uniform(1.0, 3.0),
            'echo_duration': random.randint(5, 15),
            'echo_spread': random.uniform(0.5, 1.5),
            'draw_activation_threshold': random.uniform(0.6, 0.9)
        }

    def move(self):
        dx = cos(radians(self.direction)) * self.speed
        dy = sin(radians(self.direction)) * self.speed
        new_x = max(self.bug_radius, min(800 - self.bug_radius, self.position[0] + dx))
        new_y = max(self.bug_radius, min(600 - self.bug_radius, self.position[1] + dy))
        self.position = [new_x, new_y]
        self.trail.append(tuple(self.position))
        if len(self.trail) > 50:
            self.trail.pop(0)

    def avoid_others(self, other_bugs: List['Bug']):
        collision = False
        for other in other_bugs:
            if other == self:
                continue
            dist = sqrt((self.position[0] - other.position[0]) ** 2 +
                       (self.position[1] - other.position[1]) ** 2)
            if dist < self.bug_radius * 2:
                angle_to_other = degrees(atan2(
                    other.position[1] - self.position[1],
                    other.position[0] - self.position[0]
                ))
                self.direction += 180 + random.uniform(-30, 30)
                collision = True
                # Generate particle effect
                self.create_particles()
        if collision:
            self.state = "avoiding"
            self.can_talk = True
        else:
            self.state = "exploring"

    def detect_in_vision(self, other_bugs: List['Bug'], webcam_input: np.ndarray) -> np.ndarray:
        vision_data = []
        for other in other_bugs:
            if other == self:
                continue
            dx = other.position[0] - self.position[0]
            dy = other.position[1] - self.position[1]
            distance = sqrt(dx**2 + dy**2)
            if distance > self.vision_range:
                continue
            angle_to_other = degrees(atan2(dy, dx)) % 360
            angle_diff = (angle_to_other - self.direction) % 360
            if angle_diff > 180:
                angle_diff -= 360
            if abs(angle_diff) <= self.vision_angle / 2:
                vision_data.append((distance, angle_to_other))

        vision_signal = np.zeros((5, 7))
        if webcam_input is not None:
            normalized_brightness = np.mean(webcam_input) / 255.0
            vision_signal[0, 0] = normalized_brightness

        for i, (distance, angle) in enumerate(vision_data):
            normalized_dist = max(0, (self.vision_range - distance) / self.vision_range)
            vision_signal[i % 5, i % 7] += normalized_dist
        return vision_signal

    def create_echo_trace(self, x: float, y: float):
        echo = {
            'position': (x, y),
            'thickness': self.genetic_traits['trail_thickness'],
            'duration': self.genetic_traits['echo_duration'],
            'spread': self.genetic_traits['echo_spread'],
            'remaining': self.genetic_traits['echo_duration']
        }
        self.echo_trails.append(echo)

    def create_particles(self):
        for _ in range(10):
            particle = {
                'position': self.position.copy(),
                'velocity': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'life': random.randint(5, 15)
            }
            self.particles.append(particle)

    def update_particles(self):
        for particle in self.particles.copy():
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def is_near(self, other_bug, threshold=50):
        dx = self.position[0] - other_bug.position[0]
        dy = self.position[1] - other_bug.position[1]
        distance = sqrt(dx**2 + dy**2)
        return distance < threshold

    def think_and_act(self, environment_input: np.ndarray, other_bugs: List['Bug'], 
                     webcam_input: np.ndarray, mouse_position: Tuple[int, int]):
        self.avoid_others(other_bugs)
        vision_signal = self.detect_in_vision(other_bugs, webcam_input)
        combined_input = environment_input + vision_signal
        latent_vector = self.processor.process_and_update(combined_input)
        oscillatory_energy = np.mean([abs(neuron.output) for neuron in self.processor.brain.neurons])

        # Adjust speed based on EEG data
        eeg_activity = np.mean(combined_input)
        self.speed = 2.0 + eeg_activity * 5.0  # Speed ranges from 2 to 7

        if self.state == "exploring":
            self.direction += random.uniform(-15, 15) * oscillatory_energy
        elif self.state == "avoiding":
            self.direction += random.uniform(-30, 30) * oscillatory_energy

        self.direction %= 360
        self.move()
        self.update_particles()

        neuron_states = np.array([neuron.output for neuron in self.processor.brain.neurons])

        current_time = time.time()
        # Produce sound only if enough time has passed
        if current_time - self.last_sound_time > self.sound_interval:
            # Generate sound based on sound neuron
            self.generate_sound(neuron_states[self.sound_neuron_index], mouse_position)
            self.last_sound_time = current_time
            # Randomize the next sound interval
            self.sound_interval = random.uniform(1.0, 5.0)

        # Process heard sounds
        self.process_heard_sounds(neuron_states)

        return latent_vector, "", self.echo_trails

    def generate_sound(self, neuron_state: float, mouse_position: Tuple[int, int]):
        # Map neuron_state to frequency between 1000 Hz and 1500 Hz
        frequency = 1000 + ((neuron_state + 1) / 2) * 500  # Map from [-1,1] to [1000,1500] Hz
        self.sound_frequency = frequency  # Store the frequency for other bugs to "hear"

        # Generate sound only once and reuse
        if not hasattr(self, 'sound_samples'):
            self.pre_generate_sounds()

        # Adjust volume based on mouse proximity
        distance_to_mouse = sqrt((self.position[0] - mouse_position[0])**2 + (self.position[1] - mouse_position[1])**2)
        max_distance = 200  # Distance at which volume is minimal
        volume = max(0.2, 1 - (distance_to_mouse / max_distance))  # Volume decreases with distance
        volume = min(1.0, volume)

        # Select the closest pre-generated sound
        freq_index = int((frequency - 1000) / 10)
        freq_index = min(len(self.sound_samples) - 1, max(0, freq_index))
        sound = self.sound_samples[freq_index]
        sound.set_volume(volume)

        # Play the sound
        self.sound_channel.play(sound)

    def pre_generate_sounds(self):
        # Pre-generate sounds for frequencies between 1000 Hz and 1500 Hz
        self.sound_samples = []
        sample_rate = 22050
        duration = 0.2  # Longer duration for smoother sound
        for freq in range(1000, 1501, 10):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            samples = (np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sound_array = np.stack([samples, samples], axis=-1)  # Stereo sound
            sound = pygame.sndarray.make_sound((sound_array * 32767).astype(np.int16))
            self.sound_samples.append(sound)

    def process_heard_sounds(self, neuron_states: np.ndarray):
        # Adjust the hearing neuron based on heard sounds
        if self.heard_sounds:
            # Average the frequencies of the heard sounds
            avg_frequency = np.mean(self.heard_sounds)
            # Map frequency back to neuron activation
            neuron_activation = ((avg_frequency - 1000) / 500) * 2 - 1  # Map from [1000,1500] Hz to [-1,1]
            # Update the hearing neuron state
            neuron_states[self.hearing_neuron_index] = neuron_activation
            # Clear the heard sounds
            self.heard_sounds = []

    def hear_sound(self, frequency: float):
        self.heard_sounds.append(frequency)

class EnhancedBug(Bug):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small_brain = SmallBrain(
            num_neurons=self.num_waveneurons,
            latent_dim=64
        )
        self.brain_coupler = BrainCoupler(
            self.processor.eeg_model,
            self.small_brain
        )
        self.learning_rate = 0.1
        
    def think_and_act(self, environment_input, other_bugs, webcam_input, mouse_position):
        # Remove state transfer when bugs overlap (no longer calling super().think_and_act)
        self.avoid_others(other_bugs)
        vision_signal = self.detect_in_vision(other_bugs, webcam_input)
        combined_input = environment_input + vision_signal
        latent_vector = self.processor.process_and_update(combined_input)
        oscillatory_energy = np.mean([abs(neuron.output) for neuron in self.processor.brain.neurons])

        # Adjust speed based on EEG data
        eeg_activity = np.mean(combined_input)
        self.speed = 2.0 + eeg_activity * 5.0  # Speed ranges from 2 to 7

        if self.state == "exploring":
            self.direction += random.uniform(-15, 15) * oscillatory_energy
        elif self.state == "avoiding":
            self.direction += random.uniform(-30, 30) * oscillatory_energy

        self.direction %= 360
        self.move()
        self.update_particles()

        t = time.time()
        loss = self.brain_coupler.train_step(
            torch.FloatTensor(environment_input).unsqueeze(0),
            t
        )

        brain_output = self.small_brain(
            torch.FloatTensor(latent_vector),
            t
        )

        combined_output = (
            latent_vector * (1 - self.learning_rate) + 
            brain_output.detach().numpy() * self.learning_rate
        )

        neuron_states = np.array([neuron.output for neuron in self.processor.brain.neurons])

        current_time = time.time()
        # Produce sound only if enough time has passed
        if current_time - self.last_sound_time > self.sound_interval:
            # Generate sound based on sound neuron
            self.generate_sound(neuron_states[self.sound_neuron_index], mouse_position)
            self.last_sound_time = current_time
            # Randomize the next sound interval
            self.sound_interval = random.uniform(1.0, 5.0)

        # Process heard sounds
        self.process_heard_sounds(neuron_states)

        return combined_output, "", self.echo_trails

class EEGBugSimulatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EEG Bug Simulator")
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        self.model_path = tk.StringVar()
        self.webcam_index = tk.IntVar(value=0)
        self.background_image_path = tk.StringVar()
        self.num_waveneurons = tk.IntVar(value=16)
        self.simulation_running = False

        self.bug_speed = tk.DoubleVar(value=5.0)
        self.coupling_strength = tk.DoubleVar(value=0.1)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_configuration_tab()
        self.create_simulation_tab()
        self.create_help_tab()

    def create_configuration_tab(self):
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text='Configuration')

        # Model Selection
        model_frame = ttk.LabelFrame(self.config_frame, text="1. Select EEG Autoencoder Model (.pth)", padding=10)
        model_frame.pack(fill=tk.X, padx=20, pady=10)

        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=80, state='readonly')
        self.model_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT)

        # Input Source Selection
        input_frame = ttk.LabelFrame(self.config_frame, text="2. Select Input Source", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.input_option = tk.IntVar(value=1)
        ttk.Radiobutton(input_frame, text="Use Webcam", variable=self.input_option, value=1, 
                       command=self.toggle_input_option).grid(row=0, column=0, sticky='w', pady=5)
        ttk.Radiobutton(input_frame, text="Use Background Image", variable=self.input_option, value=2,
                       command=self.toggle_input_option).grid(row=1, column=0, sticky='w', pady=5)

        self.webcam_frame = ttk.Frame(input_frame)
        self.webcam_frame.grid(row=0, column=1, sticky='w', pady=5, padx=10)
        ttk.Label(self.webcam_frame, text="Webcam Index:").pack(side=tk.LEFT)
        self.webcam_spinbox = ttk.Spinbox(self.webcam_frame, from_=0, to=10, width=5, textvariable=self.webcam_index)
        self.webcam_spinbox.pack(side=tk.LEFT, padx=(5,0))

        self.image_frame = ttk.Frame(input_frame)
        self.image_frame.grid(row=1, column=1, sticky='w', pady=5, padx=10)
        self.image_entry = ttk.Entry(self.image_frame, textvariable=self.background_image_path, width=60, state='disabled')
        self.image_entry.pack(side=tk.LEFT, padx=(0,10))
        self.image_browse_button = ttk.Button(self.image_frame, text="Browse", command=self.browse_background_image, state='disabled')
        self.image_browse_button.pack(side=tk.LEFT)

        # Wave Neurons Configuration
        neurons_frame = ttk.LabelFrame(self.config_frame, text="3. Configure Wave Neurons per Bug", padding=10)
        neurons_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(neurons_frame, text="Number of Wave Neurons:").grid(row=0, column=0, sticky='w', pady=5)
        self.neurons_spinbox = ttk.Spinbox(neurons_frame, from_=1, to=1500, increment=1, textvariable=self.num_waveneurons, width=5)
        self.neurons_spinbox.grid(row=0, column=1, sticky='w', pady=5, padx=(5,0))

        self.start_button = ttk.Button(self.config_frame, text="Start Simulation", command=self.start_simulation, state='disabled')
        self.start_button.pack(pady=20)

    def create_simulation_tab(self):
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text='Simulation')

        self.main_frame = tk.Frame(self.simulation_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Motion>", self.update_mouse_position)

        self.sidebar = tk.Frame(self.main_frame, width=300, bg="grey")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.cap = None
        self.background_image = None
        self.bugs = []
        self.simulation_running = False
        self.mouse_position = (0, 0)

        # Create control panel
        self.create_control_panel()
        # Create heatmap display
        self.create_heatmap_display()

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.sidebar, text="Control Panel", padding=10)
        control_frame.pack(pady=10, fill=tk.X)

        ttk.Label(control_frame, text="Bug Speed").pack()
        self.speed_scale = ttk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.bug_speed)
        self.speed_scale.pack(fill=tk.X)

        ttk.Label(control_frame, text="Coupling Strength").pack()
        self.coupling_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.coupling_strength)
        self.coupling_scale.pack(fill=tk.X)

    def create_heatmap_display(self):
        self.heatmap_frame = tk.Frame(self.sidebar)
        self.heatmap_frame.pack(fill=tk.BOTH, expand=True)
        self.heatmap_canvases = {}
        self.heatmap_figures = {}

    def update_mouse_position(self, event):
        self.mouse_position = (event.x, event.y)

    def toggle_input_option(self):
        option = self.input_option.get()
        if option == 1:
            self.webcam_spinbox.config(state='normal')
            self.background_image_path.set('')
            self.image_entry.config(state='disabled')
            self.image_browse_button.config(state='disabled')
        elif option == 2:
            self.webcam_spinbox.config(state='disabled')
            self.image_entry.config(state='normal')
            self.image_browse_button.config(state='normal')

    def browse_model(self):
        file_path = filedialog.askopenfilename(title="Select EEG Autoencoder Model", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)
            self.check_ready_to_start()

    def browse_background_image(self):
        file_path = filedialog.askopenfilename(title="Select Background Image", 
                                             filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.background_image_path.set(file_path)

    def check_ready_to_start(self):
        if self.model_path.get():
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disabled')

    def draw_particles(self, bug: Bug):
        for particle in bug.particles:
            x, y = particle['position']
            life_ratio = particle['life'] / 15
            color = f"#{int(255*life_ratio):02x}{int(255*life_ratio):02x}00"
            self.canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2,
                fill=color, outline=""
            )

    def start_simulation(self):
        if self.simulation_running:
            messagebox.showwarning("Simulation Running", "The simulation is already running.")
            return

        if not self.model_path.get():
            messagebox.showwarning("No Model Selected", "Please select an EEG autoencoder model before starting.")
            return

        # Initialize input source
        if self.input_option.get() == 1:
            webcam_idx = self.webcam_index.get()
            self.cap = cv2.VideoCapture(webcam_idx)
            if not self.cap.isOpened():
                messagebox.showerror("Webcam Error", f"Cannot open webcam with index {webcam_idx}.")
                return
        else:
            bg_path = self.background_image_path.get()
            if not os.path.exists(bg_path):
                messagebox.showerror("Image Not Found", f"Background image not found at {bg_path}.")
                return
            self.background_image = Image.open(bg_path).resize((self.canvas_width, self.canvas_height))
            self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Initialize processor and bugs
        processor = DynamicWaveEEGProcessor(
            eeg_model_path=self.model_path.get(),
            latent_dim=latent_dim,
            num_neurons=self.num_waveneurons.get()
        )

        bug_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        self.bugs = []
        for name in bug_names:
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            bug = EnhancedBug(
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
                color=color,
                name=name,
                processor=processor,
                bug_radius=20,
                num_waveneurons=self.num_waveneurons.get()
            )
            self.bugs.append(bug)

        self.simulation_running = True
        self.create_bug_heatmaps()
        self.run_simulation()

    def create_bug_heatmaps(self):
        for bug in self.bugs:
            fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
            canvas = FigureCanvasTkAgg(fig, master=self.heatmap_frame)
            canvas.get_tk_widget().pack()
            self.heatmap_canvases[bug.name] = canvas
            self.heatmap_figures[bug.name] = (fig, ax)

    def update_bug_heatmaps(self):
        for bug in self.bugs:
            fig, ax = self.heatmap_figures[bug.name]
            ax.clear()
            neuron_states = np.array([neuron.output for neuron in bug.processor.brain.neurons])
            size = int(np.ceil(np.sqrt(len(neuron_states))))
            data = np.zeros((size, size))
            data.flat[:len(neuron_states)] = neuron_states
            ax.imshow(data, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title(bug.name, fontsize=8)
            ax.axis('off')
            self.heatmap_canvases[bug.name].draw()

    def run_simulation(self):
        if not self.simulation_running:
            return

        self.canvas.delete("all")

        # Handle background
        if self.background_image is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Handle webcam
        webcam_input = None
        if self.input_option.get() == 1 and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.webcam_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.webcam_image)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                webcam_input = cv2.resize(gray_frame, (self.canvas_width, self.canvas_height))

        # Environment input
        environment_input = np.random.rand(5, 7)

        # Update and draw bugs
        for bug in self.bugs:
            latent_vector, _, echo_trails = bug.think_and_act(
                environment_input, self.bugs, webcam_input, self.mouse_position
            )
            x, y = bug.position

            # Trail customization based on neuron states
            neuron_states = np.array([neuron.output for neuron in bug.processor.brain.neurons])
            avg_activation = np.mean(neuron_states)
            # Ensure avg_activation is within [-1, 1]
            avg_activation = np.clip(avg_activation, -1, 1)

            trail_color_intensity = int(255 * (avg_activation + 1) / 2)
            trail_color_intensity = max(0, min(255, trail_color_intensity))

            inverse_intensity = 255 - trail_color_intensity
            inverse_intensity = max(0, min(255, inverse_intensity))

            trail_color = f"#{trail_color_intensity:02x}00{inverse_intensity:02x}"

            # Adjust trail thickness
            trail_thickness = 1 + 3 * (avg_activation + 1) / 2
            trail_thickness = max(0.5, min(4.0, trail_thickness))

            if len(bug.trail) > 1:
                self.canvas.create_line(bug.trail, fill=trail_color, width=trail_thickness, smooth=True)

            self.canvas.create_oval(
                x - bug.bug_radius, y - bug.bug_radius,
                x + bug.bug_radius, y + bug.bug_radius,
                fill=bug.color, outline=""
            )

            self.draw_particles(bug)

            for echo in echo_trails.copy():
                if echo['remaining'] > 0:
                    self.draw_echo(echo)
                    echo['remaining'] -= 1
                else:
                    bug.echo_trails.remove(echo)

        # Simulate hearing sounds from nearby bugs
        for bug in self.bugs:
            for other_bug in self.bugs:
                if bug != other_bug and bug.is_near(other_bug):
                    if other_bug.sound_frequency is not None:
                        bug.hear_sound(other_bug.sound_frequency)

        self.update_bug_heatmaps()

        self.root.after(50, self.run_simulation)

    def draw_echo(self, echo: dict):
        x, y = echo['position']
        thickness = echo['thickness']
        spread = echo['spread']
        self.canvas.create_oval(
            x - spread * 10, y - spread * 10,
            x + spread * 10, y + spread * 10,
            outline='white', width=thickness, fill=''
        )

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.simulation_running = False
        self.root.destroy()
        pygame.quit()

    def create_help_tab(self):
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text='Help')

        help_text = """
        EEG Bug Simulator

        This simulator combines EEG data processing with neural oscillators to create interactive agents.

        Key Features:
        - EEG Model Integration: Processes brain activity patterns
        - SmallBrain System: Mini neural networks that learn from EEG patterns
        - Wave-based Communication: Bugs share internal states
        - Dynamic Visualization: See neural activity and interactions
        - Neural Activity Heatmaps: View bugs' neural heatmaps in the sidebar
        - Particle Effects: Visualize bug interactions
        - Audio Language: Bugs produce evolving sounds
        - Control Panel: Adjust simulation parameters in real-time
        - Audio Zoom: Sound volume adjusts based on mouse position

        Usage:
        1. Select trained EEG model (.pth file)
        2. Choose input source (webcam/image)
        3. Configure number of wave neurons
        4. Start simulation

        Brain States:
        - Heatmaps display neural activity
        - Trails show movement history
        - Particle effects when bugs interact
        """

        help_label = ttk.Label(self.help_frame, text=help_text, wraplength=700, justify='left')
        help_label.pack(padx=20, pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGBugSimulatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
