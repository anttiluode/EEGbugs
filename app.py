import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from math import cos, sin, radians, sqrt, atan2, degrees
import cv2
from typing import List, Tuple
import random
import logging
from datetime import datetime
import threading

# ==============================
# Configuration and Parameters
# ==============================

# Define frequency bands
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 49),
    'low_gamma': (49, 60),
    'high_gamma': (60, 70)
}

# Define Autoencoder parameters
latent_dim = 64
eeg_batch_size = 64
eeg_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging configuration
logging.basicConfig(
    filename='eeg_bug_simulator.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ==============================
# Reproducibility
# ==============================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

# ==============================
# EEG Autoencoder Model
# ==============================
class EEGAutoencoder(nn.Module):
    def __init__(self, channels=5, frequency_bands=7, latent_dim=64):
        super(EEGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.flattened_dim = self._calculate_flattened_dim(channels, frequency_bands)
        self.fc1 = nn.Linear(self.flattened_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 2), stride=(1, 2), output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1, 2), stride=(1, 2), output_padding=(0, 1)),
            nn.Sigmoid(),
        )

    def _calculate_flattened_dim(self, channels, frequency_bands):
        test_input = torch.zeros(1, 1, channels, frequency_bands)
        with torch.no_grad():
            output = self.encoder(test_input)
        return output.numel()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(x.size(0), 32, -1, 1)
        x = self.decoder(x)
        return x, latent

# ==============================
# EEG Wave Neuron with Memory
# ==============================
class EEGWaveNeuron:
    def __init__(self, frequency=None, amplitude=None, phase=None, memory_size=100):
        self.frequency = frequency if frequency is not None else np.random.uniform(0.1, 1.0)
        self.amplitude = amplitude if amplitude is not None else np.random.uniform(0.5, 1.0)
        self.phase = phase if phase is not None else np.random.uniform(0, 2 * np.pi)
        self.output = 0.0
        self.memory = np.zeros(memory_size)
        self.memory_pointer = 0
        self.resonance_weight = 0.9  # Stronger feedback

    def activate(self, input_signal, eeg_signal, t):
        eeg_influence = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) * eeg_signal
        past_resonance = np.mean(self.memory) * self.resonance_weight
        self.output = eeg_influence + input_signal + past_resonance
        self.memory[self.memory_pointer] = self.output
        self.memory_pointer = (self.memory_pointer + 1) % len(self.memory)
        return self.output

# ==============================
# Resonant Brain with Hebbian Learning
# ==============================
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

        # Hebbian Learning: Adjust connections based on co-activation
        for (pre, post), weight in self.connections.items():
            if pre.output > 0.5 and post.output > 0.5:
                # Increase weight if both neurons are active
                self.connections[(pre, post)] = min(weight + 0.01, 1.0)
            else:
                # Decrease weight if not active together
                self.connections[(pre, post)] = max(weight - 0.01, 0.0)

# ==============================
# Dynamic EEG Processor with Memory and Learning
# ==============================
class DynamicWaveEEGProcessor:
    def __init__(self, eeg_model_path: str, latent_dim=64, num_neurons=16):
        self.eeg_model = EEGAutoencoder(channels=5, frequency_bands=7, latent_dim=latent_dim)
        try:
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location=torch.device('cpu')), strict=False)
        except TypeError:
            # If weights_only is not supported in current PyTorch version
            self.eeg_model.load_state_dict(torch.load(eeg_model_path, map_location=torch.device('cpu')))
        self.eeg_model.eval()
        self.brain = ResonantBrain(num_neurons=num_neurons)
        self.time = 0.0
        self.memory = []  # Long-term memory

    def process_and_update(self, eeg_data: np.ndarray) -> np.ndarray:
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,5,7)
        with torch.no_grad():
            _, latent_vector = self.eeg_model(eeg_tensor)
        latent_vector = latent_vector.squeeze(0).numpy()
        self.brain.update(latent_vector, dt=0.1)
        self.time += 0.1

        # Update memory with current latent vector
        self.memory.append(latent_vector)
        if len(self.memory) > 50:  # Keep last 50 vectors
            self.memory.pop(0)

        return latent_vector

# ==============================
# Bug System with Enhanced Intelligence and Drawing Capability
# ==============================
class Bug:
    def __init__(self, canvas_width: int, canvas_height: int, color: str, name: str, processor: DynamicWaveEEGProcessor, bug_radius: int = 20, num_waveneurons: int = 16):
        self.position = [random.randint(bug_radius, canvas_width - bug_radius),
                         random.randint(bug_radius, canvas_height - bug_radius)]
        self.direction = random.uniform(0, 360)
        self.speed = 5.0
        self.color = color
        self.name = name  # Unique name for each bug
        self.processor = processor
        self.bug_radius = bug_radius
        self.vision_angle = 90  # Degrees
        self.vision_range = 100  # Pixels
        self.state = "exploring"  # Other states: "avoiding", "seeking"
        self.trail = []  # To store past positions for visualization
        self.can_talk = False  # Flag to indicate if the bug can talk
        self.genetic_traits = self.initialize_genetic_traits()
        self.echo_trails = []  # To store echo traces
        self.num_waveneurons = num_waveneurons

    def initialize_genetic_traits(self) -> dict:
        """
        Initialize genetic traits for the bug. These traits can influence drawing behavior.
        """
        traits = {
            'trail_thickness': random.uniform(1.0, 3.0),
            'echo_duration': random.randint(5, 15),  # Number of frames the echo lasts
            'echo_spread': random.uniform(0.5, 1.5),  # Spread multiplier for echo size
            'draw_activation_threshold': random.uniform(0.6, 0.9)  # Threshold to activate drawing
        }
        return traits

    def move(self):
        dx = cos(radians(self.direction)) * self.speed
        dy = sin(radians(self.direction)) * self.speed
        new_x = max(self.bug_radius, min(800 - self.bug_radius, self.position[0] + dx))
        new_y = max(self.bug_radius, min(600 - self.bug_radius, self.position[1] + dy))
        self.position = [new_x, new_y]
        self.trail.append(tuple(self.position))
        if len(self.trail) > 20:
            self.trail.pop(0)

    def avoid_others(self, other_bugs: List['Bug']):
        collision = False
        for other in other_bugs:
            if other == self:
                continue
            dist = sqrt((self.position[0] - other.position[0]) ** 2 +
                        (self.position[1] - other.position[1]) ** 2)
            if dist < self.bug_radius * 2:  # Collision radius
                angle_to_other = degrees(atan2(
                    other.position[1] - self.position[1],
                    other.position[0] - self.position[0]
                ))
                self.direction += 180 + random.uniform(-30, 30)  # Reverse and jitter
                collision = True
        if collision:
            self.state = "avoiding"
            self.can_talk = True  # Trigger talking on collision
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

        # Incorporate webcam input (brightness or motion detection)
        vision_signal = np.zeros((5, 7))
        if webcam_input is not None:
            normalized_brightness = np.mean(webcam_input) / 255.0
            vision_signal[0, 0] = normalized_brightness

        for i, (distance, angle) in enumerate(vision_data):
            normalized_dist = max(0, (self.vision_range - distance) / self.vision_range)
            vision_signal[i % 5, i % 7] += normalized_dist
        return vision_signal

    def generate_talk(self, neuron_states: np.ndarray) -> str:
        """
        Generate a string based on the neuron states by converting them to ASCII characters.
        """
        # Normalize neuron_states to 0-1
        min_val = np.min(neuron_states)
        max_val = np.max(neuron_states)
        if max_val - min_val == 0:
            normalized_states = np.zeros_like(neuron_states)
        else:
            normalized_states = (neuron_states - min_val) / (max_val - min_val)
        
        # Scale to printable ASCII range (32-126)
        ascii_codes = (normalized_states * (126 - 32) + 32).astype(int)
        # Convert to characters
        ascii_chars = ''.join([chr(code) for code in ascii_codes])
        # Optional: Limit the length of the message
        return ascii_chars[:20]  # Limit to 20 characters

    def generate_draw_command(self, neuron_states: np.ndarray) -> bool:
        """
        Decide whether to activate drawing based on neuron states and genetic traits.
        """
        avg_activation = np.mean(neuron_states)
        return avg_activation > self.genetic_traits['draw_activation_threshold']

    def create_echo_trace(self, x: float, y: float):
        """
        Create an echo trace based on genetic traits.
        """
        echo = {
            'position': (x, y),
            'thickness': self.genetic_traits['trail_thickness'],
            'duration': self.genetic_traits['echo_duration'],
            'spread': self.genetic_traits['echo_spread'],
            'remaining': self.genetic_traits['echo_duration']
        }
        self.echo_trails.append(echo)

    def think_and_act(self, environment_input: np.ndarray, other_bugs: List['Bug'], webcam_input: np.ndarray) -> Tuple[np.ndarray, str, List[dict]]:
        self.avoid_others(other_bugs)
        vision_signal = self.detect_in_vision(other_bugs, webcam_input)
        combined_input = environment_input + vision_signal
        latent_vector = self.processor.process_and_update(combined_input)
        oscillatory_energy = np.mean([abs(neuron.output) for neuron in self.processor.brain.neurons])

        # Decision-making based on oscillatory energy and state
        if self.state == "exploring":
            self.direction += random.uniform(-15, 15) * oscillatory_energy
        elif self.state == "avoiding":
            # More aggressive turning when avoiding
            self.direction += random.uniform(-30, 30) * oscillatory_energy

        # Normalize direction
        self.direction %= 360

        self.move()

        # Always define neuron_states
        neuron_states = np.array([neuron.output for neuron in self.processor.brain.neurons])

        # Handle talking
        talk_message = ""
        if self.can_talk:
            # Generate talk based on neuron states
            talk_message = f"{self.name}: {self.generate_talk(neuron_states)}"
            self.can_talk = False  # Reset talk flag

        # Handle drawing
        draw_command = self.generate_draw_command(neuron_states)
        if draw_command:
            self.create_echo_trace(self.position[0], self.position[1])

        return latent_vector, talk_message, self.echo_trails

# ==============================
# Tkinter App with Enhanced Features
# ==============================
class EEGBugSimulatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EEG Bug Simulator")
        self.root.geometry("1200x800")
        self.root.resizable(False, False)

        # Initialize variables
        self.model_path = tk.StringVar()
        self.webcam_index = tk.IntVar(value=0)
        self.background_image_path = tk.StringVar()
        self.num_waveneurons = tk.IntVar(value=16)
        self.simulation_running = False

        # Setup Notebook (Tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create Tabs
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

        # Input Source Selection (Webcam or Background Image)
        input_frame = ttk.LabelFrame(self.config_frame, text="2. Select Input Source", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        self.input_option = tk.IntVar(value=1)  # 1: Webcam, 2: Background Image
        ttk.Radiobutton(input_frame, text="Use Webcam", variable=self.input_option, value=1, command=self.toggle_input_option).grid(row=0, column=0, sticky='w', pady=5)
        ttk.Radiobutton(input_frame, text="Use Background Image", variable=self.input_option, value=2, command=self.toggle_input_option).grid(row=1, column=0, sticky='w', pady=5)

        # Webcam Selection
        self.webcam_frame = ttk.Frame(input_frame)
        self.webcam_frame.grid(row=0, column=1, sticky='w', pady=5, padx=10)

        ttk.Label(self.webcam_frame, text="Webcam Index:").pack(side=tk.LEFT)
        self.webcam_spinbox = ttk.Spinbox(self.webcam_frame, from_=0, to=10, width=5, textvariable=self.webcam_index)
        self.webcam_spinbox.pack(side=tk.LEFT, padx=(5,0))

        # Background Image Selection
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
        self.neurons_spinbox = ttk.Spinbox(neurons_frame, from_=1, to=100, increment=1, textvariable=self.num_waveneurons, width=5)
        self.neurons_spinbox.grid(row=0, column=1, sticky='w', pady=5, padx=(5,0))

        # Start Simulation Button
        self.start_button = ttk.Button(self.config_frame, text="Start Simulation", command=self.start_simulation, state='disabled')
        self.start_button.pack(pady=20)

    def toggle_input_option(self):
        option = self.input_option.get()
        if option == 1:
            # Enable webcam selection
            self.webcam_spinbox.config(state='normal')
            # Disable background image selection
            self.background_image_path.set('')
            self.image_entry.config(state='disabled')
            self.image_browse_button.config(state='disabled')
        elif option == 2:
            # Disable webcam selection
            self.webcam_spinbox.config(state='disabled')
            # Enable background image selection
            self.image_entry.config(state='normal')
            self.image_browse_button.config(state='normal')

    def browse_model(self):
        file_path = filedialog.askopenfilename(title="Select EEG Autoencoder Model", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)
            self.check_ready_to_start()

    def browse_background_image(self):
        file_path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.background_image_path.set(file_path)

    def check_ready_to_start(self):
        if self.model_path.get():
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disabled')

    def create_simulation_tab(self):
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text='Simulation')

        # Create main frame
        self.main_frame = tk.Frame(self.simulation_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas for bugs
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Create sidebar for discussions
        self.sidebar = tk.Frame(self.main_frame, width=300, bg="grey")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Create text widget for displaying discussions
        self.discussion_text = tk.Text(self.sidebar, wrap=tk.WORD, bg="lightgrey", state=tk.DISABLED)
        self.discussion_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Initialize webcam or background
        self.cap = None
        self.background_image = None

        # Initialize simulation variables
        self.bugs = []
        self.simulation_running = False

    def create_help_tab(self):
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text='Help')

        help_text = (
            "### EEG Bug Simulator\n\n"
            "The **EEG Bug Simulator** is an interactive simulation where intelligent bugs navigate an environment influenced by EEG data. These bugs possess a configurable number of wave neurons, enabling them to perceive their surroundings, share their inner states, and interact dynamically.\n\n"
            "**Key Features:**\n"
            "- **Wave Neurons:** Each bug has a set number of wave neurons that process EEG-derived latent vectors, influencing their behavior and interactions.\n"
            "- **Vision:** Bugs have a defined vision cone through which they can detect other bugs and environmental inputs.\n"
            "- **Communication:** Bugs can share their internal states by generating ASCII-based messages derived from their neural activations.\n"
            "- **Drawing Capability:** Based on their neural states, bugs can leave trails and echo traces, visualizing their interactions and behaviors.\n\n"
            "**How It Works:**\n"
            "1. **Model Integration:** The simulator uses a pre-trained EEG autoencoder model to process EEG data and extract latent vectors.\n"
            "2. **Neural Processing:** These latent vectors are fed into each bug's wave neurons, determining their oscillatory behavior and decision-making processes.\n"
            "3. **Dynamic Interaction:** Bugs interact with each other and the environment based on their perceptions and internal states, leading to emergent behaviors within the simulation.\n\n"
            "**Usage Instructions:**\n"
            "1. **Configuration Tab:** Select your EEG autoencoder model, choose between using a webcam or a background image, and set the number of wave neurons per bug.\n"
            "2. **Start Simulation:** Once configured, click the 'Start Simulation' button to launch the simulation.\n"
            "3. **Monitor Simulation:** Observe the bugs as they navigate the environment, communicate, and leave visual traces of their interactions.\n\n"
            "**Technical Details:**\n"
            "- **EEG Autoencoder:** Processes EEG data to generate latent vectors that influence bug behavior.\n"
            "- **Hebbian Learning:** Bugs adapt their neural connections based on interactions, enabling more sophisticated and responsive behaviors over time.\n\n"
            "**Support:**\n"
            "For any questions or support, please contact [Your Contact Information].\n\n"
            "Enjoy exploring the fascinating dynamics of intelligent EEG-driven bugs!"
        )

        help_label = tk.Label(self.help_frame, text=help_text, justify=tk.LEFT, anchor='nw', bg="white", padx=10, pady=10, wraplength=760)
        help_label.pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        if self.simulation_running:
            messagebox.showwarning("Simulation Running", "The simulation is already running.")
            return

        if not self.model_path.get():
            messagebox.showwarning("No Model Selected", "Please select an EEG autoencoder model before starting the simulation.")
            return

        # Initialize webcam or background image
        if self.input_option.get() == 1:
            # Use webcam
            webcam_idx = self.webcam_index.get()
            self.cap = cv2.VideoCapture(webcam_idx)
            if not self.cap.isOpened():
                messagebox.showerror("Webcam Error", f"Cannot open webcam with index {webcam_idx}.")
                return
        elif self.input_option.get() == 2:
            # Use background image
            bg_path = self.background_image_path.get()
            if not os.path.exists(bg_path):
                messagebox.showerror("Image Not Found", f"Background image not found at {bg_path}.")
                return
            self.background_image = Image.open(bg_path).resize((self.canvas_width, self.canvas_height))
            self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Initialize EEG processor
        processor = DynamicWaveEEGProcessor(eeg_model_path=self.model_path.get(), latent_dim=latent_dim, num_neurons=self.num_waveneurons.get())

        # Initialize bugs
        num_bugs = 5  # You can make this configurable if desired
        bug_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
        selected_names = random.sample(bug_names, num_bugs)

        self.bugs = []
        for i in range(num_bugs):
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            name = selected_names[i]
            bug = Bug(
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
        self.run_simulation()

    def run_simulation(self):
        if not self.simulation_running:
            return

        self.canvas.delete("all")

        # Display background image if selected
        if self.background_image is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Read webcam input if using webcam
        if self.input_option.get() == 1 and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.webcam_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.webcam_image)
                # Convert to grayscale for vision processing
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                webcam_input = cv2.resize(gray_frame, (self.canvas_width, self.canvas_height))
            else:
                webcam_input = None
        else:
            webcam_input = None

        # Simulated environmental input
        environment_input = np.random.rand(5, 7)  # Adjust as needed

        # Collect all talk messages and draw commands to display
        talk_messages = []

        # Update bugs
        for bug in self.bugs:
            latent_vector, talk_message, echo_trails = bug.think_and_act(environment_input, self.bugs, webcam_input)
            x, y = bug.position

            # Draw trail
            if len(bug.trail) > 1:
                self.canvas.create_line(
                    bug.trail, fill=bug.color, width=2, smooth=True
                )

            # Draw bug
            self.canvas.create_oval(
                x - bug.bug_radius, y - bug.bug_radius,
                x + bug.bug_radius, y + bug.bug_radius,
                fill=bug.color, outline=""
            )

            # Draw vision cone
            self.draw_vision_cone(bug)

            # Draw state indicator
            self.draw_state_indicator(bug)

            # Collect talk messages
            if talk_message:
                talk_messages.append(talk_message)

            # Handle echo trails
            for echo in echo_trails.copy():  # Use copy to avoid modification during iteration
                if echo['remaining'] > 0:
                    self.draw_echo(echo)
                    echo['remaining'] -= 1
                else:
                    bug.echo_trails.remove(echo)

        # Update the discussion sidebar
        if talk_messages:
            self.append_discussion(talk_messages)

        self.root.after(50, self.run_simulation)  # Update every 50 ms

    def draw_vision_cone(self, bug: Bug):
        x, y = bug.position
        vision_start = bug.direction - bug.vision_angle / 2
        vision_end = bug.direction + bug.vision_angle / 2

        # Calculate the two outer points of the vision cone
        end1 = (
            x + cos(radians(vision_start)) * bug.vision_range,
            y + sin(radians(vision_start)) * bug.vision_range
        )
        end2 = (
            x + cos(radians(vision_end)) * bug.vision_range,
            y + sin(radians(vision_end)) * bug.vision_range
        )

        # Draw the vision cone as a polygon
        self.canvas.create_polygon(
            [x, y, end1[0], end1[1], end2[0], end2[1]],
            fill=bug.color, stipple="gray25", outline=""
        )

    def draw_state_indicator(self, bug: Bug):
        x, y = bug.position
        state_color = {
            "exploring": "green",
            "avoiding": "red",
            "seeking": "blue"
        }
        indicator_color = state_color.get(bug.state, "white")
        self.canvas.create_rectangle(
            x - bug.bug_radius, y + bug.bug_radius + 5,
            x + bug.bug_radius, y + bug.bug_radius + 15,
            fill=indicator_color, outline=""
        )
        # Tooltip for state
        self.canvas.create_text(
            x, y + bug.bug_radius + 10,
            text=bug.state, fill="white", font=("Helvetica", 8)
        )

    def draw_echo(self, echo: dict):
        """
        Draw an echo trace based on the bug's genetic traits.
        """
        x, y = echo['position']
        thickness = echo['thickness']
        spread = echo['spread']
        # Draw a circle with spread factor
        self.canvas.create_oval(
            x - spread * 10, y - spread * 10,
            x + spread * 10, y + spread * 10,
            outline=self.bugs[0].color, width=thickness, fill=''
        )

    def append_discussion(self, messages: List[str]):
        """
        Append new messages to the discussion sidebar with timestamps.
        """
        self.discussion_text.config(state=tk.NORMAL)
        for msg in messages:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.discussion_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.discussion_text.see(tk.END)  # Auto-scroll to the end
        self.discussion_text.config(state=tk.DISABLED)

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    app = EEGBugSimulatorApp(root)
    root.mainloop()
