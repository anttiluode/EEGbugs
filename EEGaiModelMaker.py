import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
from scipy.signal import butter, lfilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import threading
from PIL import Image, ImageTk

# ===============================
# 0. Configuration and Parameters
# ===============================

# Define paths
data_dir = r'G:\DocsHouse\59 eeg to imagse'  # Update as per your directory
edf_file = os.path.join(data_dir, 'SC4001E0-PSG.edf')  # Update as per your EEG data file
features_dir = os.path.join(data_dir, 'features')
os.makedirs(features_dir, exist_ok=True)

# Define frequency bands
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 49)  # Adjusted to be less than Nyquist (50 Hz)
}

# Define Autoencoder parameters
latent_dim = 64
eeg_batch_size = 64
eeg_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging configuration
logging.basicConfig(
    filename='eeg_autoencoder.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ===============================
# 1. Reproducibility
# ===============================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

set_seed(42)

# ===============================
# 2. Data Preprocessing for EEG
# ===============================

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    high = min(high, 0.99)  # Ensure high is less than 1.0

    if low >= high:
        raise ValueError(f"Invalid band: lowcut={lowcut}Hz, highcut={highcut}Hz.")

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=1)
    return y

def preprocess_eeg(raw, epoch_length=1):
    """
    Preprocess EEG data by filtering into frequency bands and epoching.

    Returns:
    - epochs_normalized: NumPy array of shape (epochs, channels, frequency_bands)
    - frequency_bands: Dict of frequency band names and their ranges
    """
    fs = raw.info['sfreq']
    channels = raw.info['nchan']
    samples_per_epoch = int(epoch_length * fs)
    num_epochs = raw.n_times // samples_per_epoch

    print(f"EEG Sampling Frequency: {fs} Hz")
    print(f"Number of Channels: {channels}")
    print(f"Number of Epochs: {num_epochs}")

    processed_data = []

    for epoch in tqdm(range(num_epochs), desc="Preprocessing EEG Epochs"):
        start_sample = epoch * samples_per_epoch
        end_sample = start_sample + samples_per_epoch
        epoch_data = raw.get_data(start=start_sample, stop=end_sample)  # Shape: (channels, samples)

        band_powers = []
        for band_name, band in frequency_bands.items():
            try:
                filtered = bandpass_filter(epoch_data, band[0], band[1], fs)
                power = np.mean(filtered ** 2, axis=1)  # Power per channel
                band_powers.append(power)
            except ValueError as ve:
                logging.error(f"Skipping band {band_name} for epoch {epoch+1}: {ve}")
                band_powers.append(np.zeros(channels))  # Placeholder for invalid bands

        # Stack frequency bands: shape (frequency_bands, channels)
        band_powers = np.stack(band_powers, axis=1)
        processed_data.append(band_powers)

    # Convert to NumPy array: shape (epochs, frequency_bands, channels)
    processed_data = np.array(processed_data)
    # Transpose to (epochs, channels, frequency_bands)
    processed_data = np.transpose(processed_data, (0, 2, 1))

    # Normalize the data
    epochs_mean = np.mean(processed_data, axis=(0, 1), keepdims=True)
    epochs_std = np.std(processed_data, axis=(0, 1), keepdims=True)
    epochs_normalized = (processed_data - epochs_mean) / epochs_std

    print(f'Processed EEG Data Shape: {epochs_normalized.shape}')  # (epochs, channels, frequency_bands)

    return epochs_normalized, frequency_bands

# ===============================
# 3. Dataset Classes
# ===============================

class EEGDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the EEGDataset.
        Parameters:
        - data: NumPy array of shape (epochs, channels, frequency_bands)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Input and target are the same for autoencoder

# ===============================
# 4. Model Definitions
# ===============================

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
        x = x.squeeze(1)
        return x, latent

# ===============================
# 5. Training Functions
# ===============================

def train_autoencoder(model, dataloader, epochs=50, learning_rate=1e-3, device='cpu', save_every=10, progress_callback=None, log_callback=None):
    """
    Trains the autoencoder model and saves checkpoints.

    Parameters:
    - model: The autoencoder model to train.
    - dataloader: DataLoader for the training data.
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - device: Device to train on ('cpu' or 'cuda').
    - save_every: Save a checkpoint every 'save_every' epochs.
    - progress_callback: Function to call to update progress (e.g., update progress bar).
    - log_callback: Function to call to log messages (e.g., append to text area).
    
    Returns:
    - model: Trained model.
    - loss_history: List of loss values per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in tqdm(dataloader, desc=f"Training Epoch {epoch}/{epochs}", leave=False):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        message = f'Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}'
        print(message)
        logging.info(message)
        if log_callback:
            log_callback(message)

        # Update progress bar if callback is provided
        if progress_callback:
            progress_callback(epoch, epochs)

        # Save checkpoint every 'save_every' epochs
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint_path = os.path.join(features_dir, f'eeg_autoencoder_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            message = f'Checkpoint saved to {checkpoint_path}'
            print(message)
            logging.info(message)
            if log_callback:
                log_callback(message)

    message = 'Autoencoder Training complete.'
    print(message)
    logging.info(message)
    if log_callback:
        log_callback(message)
    return model, loss_history

# ===============================
# 6. Hidden Vector Extraction
# ===============================

def extract_hidden_vectors_eeg(model, dataloader, device='cpu', progress_callback=None, log_callback=None):
    """
    Extracts hidden vectors from the trained EEG autoencoder.

    Parameters:
    - model: The trained autoencoder model.
    - dataloader: DataLoader for the data to extract vectors from.
    - device: Device to perform extraction on ('cpu' or 'cuda').
    - progress_callback: Function to call to update progress (e.g., update progress bar).
    - log_callback: Function to call to log messages (e.g., append to text area).

    Returns:
    - hidden_vectors: NumPy array of latent vectors.
    """
    model.eval()
    hidden_vectors = []
    total_batches = len(dataloader)
    with torch.no_grad():
        for idx, (data, _) in enumerate(tqdm(dataloader, desc="Extracting EEG Hidden Vectors", leave=False), 1):
            data = data.to(device)
            _, latent = model(data)
            hidden_vectors.append(latent.cpu().numpy())

            # Update progress if callback is provided
            if progress_callback:
                progress_callback(idx, total_batches)

    hidden_vectors = np.concatenate(hidden_vectors, axis=0)
    message = f'EEG Hidden Vectors Shape: {hidden_vectors.shape}'
    print(message)
    logging.info(message)
    if log_callback:
        log_callback(message)
    return hidden_vectors

# ===============================
# 7. GUI Application
# ===============================

class EEGAIModelMakerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG AI Model Maker")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Initialize variables
        self.edf_file_path = tk.StringVar()
        self.training = False
        self.loss_history = []

        # Create Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create Tabs
        self.create_simulation_tab()
        self.create_help_tab()

    def create_simulation_tab(self):
        self.simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.simulation_frame, text='EEG AI Model Maker')

        # EDF File Selection
        edf_frame = ttk.LabelFrame(self.simulation_frame, text="1. Load EDF File", padding=10)
        edf_frame.pack(fill=tk.X, padx=10, pady=10)

        self.edf_label = ttk.Label(edf_frame, textvariable=self.edf_file_path, relief=tk.SUNKEN, anchor='w')
        self.edf_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.load_button = ttk.Button(edf_frame, text="Browse", command=self.browse_edf)
        self.load_button.pack(side=tk.RIGHT)

        # Training Parameters
        params_frame = ttk.LabelFrame(self.simulation_frame, text="2. Training Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        # Epoch Length
        ttk.Label(params_frame, text="Epoch Length (seconds):").grid(row=0, column=0, sticky='w', pady=5)
        self.epoch_length_spin = ttk.Spinbox(params_frame, from_=1, to=10, increment=1)
        self.epoch_length_spin.set(1)
        self.epoch_length_spin.grid(row=0, column=1, sticky='w', pady=5, padx=5)

        # Batch Size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky='w', pady=5)
        self.batch_size_spin = ttk.Spinbox(params_frame, from_=16, to=256, increment=16)
        self.batch_size_spin.set(64)
        self.batch_size_spin.grid(row=1, column=1, sticky='w', pady=5, padx=5)

        # Number of Epochs
        ttk.Label(params_frame, text="Number of Epochs:").grid(row=2, column=0, sticky='w', pady=5)
        self.epochs_spin = ttk.Spinbox(params_frame, from_=10, to=100, increment=10)
        self.epochs_spin.set(50)
        self.epochs_spin.grid(row=2, column=1, sticky='w', pady=5, padx=5)

        # Learning Rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=3, column=0, sticky='w', pady=5)
        self.lr_entry = ttk.Entry(params_frame)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=3, column=1, sticky='w', pady=5, padx=5)

        # Start Training Button
        self.train_button = ttk.Button(self.simulation_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(pady=10)
        self.train_button.state(['disabled'])

        # Progress Bar
        progress_frame = ttk.Frame(self.simulation_frame, padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT)

        # Status Log
        log_frame = ttk.LabelFrame(self.simulation_frame, text="Status Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Post-Training Actions
        post_frame = ttk.LabelFrame(self.simulation_frame, text="3. Post-Training", padding=10)
        post_frame.pack(fill=tk.X, padx=10, pady=10)

        self.model_button = ttk.Button(post_frame, text="View/Download Model", command=self.view_model, state=tk.DISABLED)
        self.model_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.hidden_button = ttk.Button(post_frame, text="View/Download Hidden Vectors", command=self.view_hidden, state=tk.DISABLED)
        self.hidden_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.plot_button = ttk.Button(post_frame, text="View Training Loss Plot", command=self.view_plot, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5, pady=5)

    def create_help_tab(self):
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text='Help')

        help_text = (
            "### EEG AI Model Maker\n\n"
            "The **EEG AI Model Maker** is a professional tool designed to train an autoencoder model on EEG (Electroencephalography) data. This application allows you to load your own EDF (European Data Format) EEG files, configure training parameters, and initiate the training process with ease.\n\n"
            "**Features:**\n"
            "- **Load EDF Files:** Easily browse and select your EEG data files.\n"
            "- **Configure Training Parameters:** Set epoch length, batch size, number of epochs, and learning rate.\n"
            "- **Real-Time Training Progress:** Monitor the training process through a progress bar and live status updates.\n"
            "- **Post-Training Access:** After training, view and download the trained autoencoder model and the extracted hidden vectors. Additionally, view a plot of the training loss over epochs.\n\n"
            "**How to Use:**\n"
            "1. **Load EDF File:** Click the 'Browse' button to select your EDF file.\n"
            "2. **Set Parameters:** Adjust the training parameters as needed.\n"
            "3. **Start Training:** Click the 'Start Training' button to begin the training process. Monitor progress and logs in real-time.\n"
            "4. **Access Results:** Once training is complete, use the provided buttons to view/download the trained model, hidden vectors, and training loss plot.\n\n"
            "**Technical Details:**\n"
            "- **Autoencoder Architecture:** The model consists of convolutional and transpose convolutional layers designed to compress and reconstruct EEG data effectively.\n"
            "- **Logging:** All training activities and errors are logged in the 'eeg_autoencoder.log' file for reference.\n\n"
            "**Support:**\n"
            "For any questions or support, please contact [your.email@example.com](mailto:your.email@example.com).\n\n"
            "Enjoy training your EEG models with ease!"
        )

        help_label = tk.Label(self.help_frame, text=help_text, justify=tk.LEFT, anchor='nw', bg="white", padx=10, pady=10, wraplength=760)
        help_label.pack(fill=tk.BOTH, expand=True)

    def browse_edf(self):
        file_path = filedialog.askopenfilename(title="Select EDF File", filetypes=[("EDF Files", "*.edf")])
        if file_path:
            self.edf_file_path.set(file_path)
            self.train_button.state(['!disabled'])  # Enable training button

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress['value'] = progress
        self.root.update_idletasks()

    def start_training(self):
        if not self.edf_file_path.get():
            messagebox.showwarning("No EDF File", "Please select an EDF file to train the model.")
            return

        # Disable buttons during training
        self.train_button.state(['disabled'])
        self.load_button.state(['disabled'])
        self.model_button.state(['disabled'])
        self.hidden_button.state(['disabled'])
        self.plot_button.state(['disabled'])

        # Clear previous logs
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Start training in a separate thread to keep the GUI responsive
        training_thread = threading.Thread(target=self.run_training)
        training_thread.start()

    def run_training(self):
        try:
            edf_file = self.edf_file_path.get()
            epoch_length = int(self.epoch_length_spin.get())
            batch_size = int(self.batch_size_spin.get())
            epochs = int(self.epochs_spin.get())
            lr = float(self.lr_entry.get())

            # Load EEG data using MNE
            self.log_message("Loading EEG data...")
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            self.log_message("EEG Data Loaded Successfully.")
            self.log_message(f"EEG Info: {raw.info}")

            # Preprocess EEG data
            self.log_message("Preprocessing EEG data...")
            epochs_normalized, frequency_bands_used = preprocess_eeg(raw, epoch_length=epoch_length)

            # Prepare EEG Dataset and DataLoader
            self.log_message("Preparing Dataset and DataLoader...")
            eeg_dataset = EEGDataset(epochs_normalized)
            eeg_dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            # Define and Train EEG Autoencoder
            self.log_message("Initializing EEG Autoencoder...")
            channels = epochs_normalized.shape[1]  # 5
            frequency_bands_count = epochs_normalized.shape[2]  #7
            eeg_autoencoder = EEGAutoencoder(channels=channels, frequency_bands=frequency_bands_count, latent_dim=latent_dim)
            self.log_message("EEG Autoencoder Initialized.")

            self.log_message("Starting Training...")
            eeg_autoencoder, loss_history = train_autoencoder(
                eeg_autoencoder, eeg_dataloader, epochs=epochs, 
                learning_rate=lr, device=device, save_every=10,
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )

            # Save the Final Model
            final_model_path = os.path.join(features_dir, 'eeg_autoencoder_final.pth')
            torch.save(eeg_autoencoder.state_dict(), final_model_path)
            self.log_message(f'Final model saved to {final_model_path}')

            # Extract EEG Hidden Vectors
            self.log_message("Extracting EEG Hidden Vectors...")
            eeg_dataloader_no_shuffle = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            eeg_hidden_vectors = extract_hidden_vectors_eeg(
                eeg_autoencoder, eeg_dataloader_no_shuffle, device=device,
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )

            # Save EEG Hidden Vectors
            eeg_hidden_path = os.path.join(features_dir, 'eeg_hidden_vectors.npy')
            np.save(eeg_hidden_path, eeg_hidden_vectors)
            self.log_message(f'EEG Hidden Vectors Saved to {eeg_hidden_path}')

            # Plot Loss Curve
            self.log_message("Plotting Training Loss...")
            plt.figure(figsize=(10,5))
            plt.plot(range(1, epochs+1), loss_history, marker='o')
            plt.title('Training Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.grid(True)
            loss_plot_path = os.path.join(features_dir, 'training_loss.png')
            plt.savefig(loss_plot_path)
            plt.close()
            self.log_message(f'Training loss plot saved to {loss_plot_path}')

            # Enable post-training buttons
            self.model_button.state(['!disabled'])
            self.hidden_button.state(['!disabled'])
            self.plot_button.state(['!disabled'])

            self.log_message("Training Completed Successfully.")

        except Exception as e:
            error_message = f"Error during training: {e}"
            print(error_message)
            logging.error(error_message)
            self.log_message(error_message)
            messagebox.showerror("Training Error", error_message)
        finally:
            # Re-enable load button
            self.load_button.state(['!disabled'])

    def view_model(self):
        model_path = os.path.join(features_dir, 'eeg_autoencoder_final.pth')
        if os.path.exists(model_path):
            try:
                if os.name == 'nt':  # For Windows
                    os.startfile(model_path)
                elif os.name == 'posix':  # For macOS and Linux
                    os.system(f'open "{model_path}"')
                else:
                    messagebox.showinfo("Model Path", f"Model saved at: {model_path}")
            except Exception as e:
                messagebox.showerror("Error Opening Model", f"Could not open model file: {e}")
        else:
            messagebox.showerror("File Not Found", f"Model file not found at {model_path}")

    def view_hidden(self):
        hidden_path = os.path.join(features_dir, 'eeg_hidden_vectors.npy')
        if os.path.exists(hidden_path):
            try:
                if os.name == 'nt':  # For Windows
                    os.startfile(hidden_path)
                elif os.name == 'posix':  # For macOS and Linux
                    os.system(f'open "{hidden_path}"')
                else:
                    messagebox.showinfo("Hidden Vectors Path", f"Hidden vectors saved at: {hidden_path}")
            except Exception as e:
                messagebox.showerror("Error Opening Hidden Vectors", f"Could not open hidden vectors file: {e}")
        else:
            messagebox.showerror("File Not Found", f"Hidden vectors file not found at {hidden_path}")

    def view_plot(self):
        plot_path = os.path.join(features_dir, 'training_loss.png')
        if os.path.exists(plot_path):
            try:
                if os.name == 'nt':  # For Windows
                    os.startfile(plot_path)
                elif os.name == 'posix':  # For macOS and Linux
                    os.system(f'open "{plot_path}"')
                else:
                    messagebox.showinfo("Plot Path", f"Training loss plot saved at: {plot_path}")
            except Exception as e:
                messagebox.showerror("Error Opening Plot", f"Could not open training loss plot: {e}")
        else:
            messagebox.showerror("File Not Found", f"Training loss plot not found at {plot_path}")

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGAIModelMakerApp(root)
    root.mainloop()
