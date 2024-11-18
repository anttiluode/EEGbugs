import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
from scipy.signal import butter, lfilter
from tqdm import tqdm
import logging
import gradio as gr
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
from typing import Tuple, Dict, List  # Correctly import Tuple, Dict, List

# ===============================
# 1. Logging Configuration
# ===============================

logging.basicConfig(
    filename='eeg_bug_simulator.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ===============================
# 2. Configuration and Parameters
# ===============================

class Config:
    def __init__(self):
        self.fs = 100.0  # Sampling frequency (Hz)
        self.epoch_length = 1  # Epoch length in seconds
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 49)
        }
        self.latent_dim = 64
        self.eeg_batch_size = 64
        self.eeg_epochs = 50
        self.learning_rate = 1e-3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# 3. Data Preprocessing
# ===============================

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    high = min(high, 0.99)

    if low >= high:
        raise ValueError(f"Invalid band: lowcut={lowcut}Hz, highcut={highcut}Hz.")

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=1)
    return y

def preprocess_eeg(raw, config):
    """Preprocess EEG data by filtering into frequency bands and epoching."""
    fs = raw.info['sfreq']
    channels = raw.info['nchan']
    samples_per_epoch = int(config.epoch_length * fs)
    num_epochs = raw.n_times // samples_per_epoch

    processed_data = []

    for epoch in tqdm(range(num_epochs), desc="Preprocessing EEG Epochs"):
        start_sample = epoch * samples_per_epoch
        end_sample = start_sample + samples_per_epoch
        epoch_data = raw.get_data(start=start_sample, stop=end_sample)

        band_powers = []
        for band_name, band in config.frequency_bands.items():
            try:
                filtered = bandpass_filter(epoch_data, band[0], band[1], fs)
                power = np.mean(filtered ** 2, axis=1)
                band_powers.append(power)
            except ValueError as ve:
                logging.error(f"Skipping band {band_name} for epoch {epoch+1}: {ve}")
                band_powers.append(np.zeros(channels))

        band_powers = np.stack(band_powers, axis=1)
        processed_data.append(band_powers)

    processed_data = np.array(processed_data)
    processed_data = np.transpose(processed_data, (0, 2, 1))

    epochs_mean = np.mean(processed_data, axis=(0, 1), keepdims=True)
    epochs_std = np.std(processed_data, axis=(0, 1), keepdims=True)
    epochs_normalized = (processed_data - epochs_mean) / epochs_std

    return epochs_normalized

# ===============================
# 4. Dataset Class
# ===============================

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# ===============================
# 5. Model Definition
# ===============================

class EEGAutoencoder(nn.Module):
    def __init__(self, channels=5, frequency_bands=7, latent_dim=64):
        super(EEGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2))
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * channels * (frequency_bands // 4), latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * channels * (frequency_bands // 4))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(-1,32,5,1)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x, latent

# ===============================
# 6. Training Functions
# ===============================

def train_autoencoder(model, dataloader, config, progress=gr.Progress()):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(config.device)
    
    progress_text = ""
    for epoch in range(1, config.eeg_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for data, target in progress.tqdm(dataloader, desc=f"Epoch {epoch}/{config.eeg_epochs}"):
            data = data.to(config.device)
            target = target.to(config.device)

            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        progress_text += f'Epoch {epoch}/{config.eeg_epochs}, Loss: {epoch_loss:.6f}\n'
        
    return model, progress_text

# ===============================
# 7. Hidden Vector Extraction
# ===============================

def extract_hidden_vectors(model, dataloader, config, progress=gr.Progress()):
    model.eval()
    hidden_vectors = []
    with torch.no_grad():
        for data, _ in progress.tqdm(dataloader, desc="Extracting Hidden Vectors"):
            data = data.to(config.device)
            _, latent = model(data)
            hidden_vectors.append(latent.cpu().numpy())
    return np.concatenate(hidden_vectors, axis=0)

# ===============================
# 8. EEGAnalyzer Class
# ===============================

class EEGAnalyzer:
    def __init__(self, hidden_vectors: np.ndarray, model: torch.nn.Module):
        """
        Initialize the EEG analyzer with model outputs.

        Args:
            hidden_vectors: The extracted latent vectors from the autoencoder
            model: The trained EEG autoencoder model
        """
        self.hidden_vectors = hidden_vectors
        self.model = model
        self.device = next(model.parameters()).device

    def reduce_dimensions(self, method: str = 'tsne', n_components: int = 3) -> np.ndarray:
        """
        Reduce dimensionality of hidden vectors for visualization.

        Args:
            method: 'tsne', 'umap', or 'pca'
            n_components: Number of dimensions to reduce to

        Returns:
            Reduced dimensionality representation
        """
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced_vectors = reducer.fit_transform(self.hidden_vectors)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_vectors = reducer.fit_transform(self.hidden_vectors)
        else:
            reducer = PCA(n_components=n_components)
            reduced_vectors = reducer.fit_transform(self.hidden_vectors)
        return reduced_vectors

    def cluster_vectors(self, n_clusters: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Perform clustering on hidden vectors and analyze clusters.

        Args:
            n_clusters: Number of clusters for KMeans

        Returns:
            Tuple of (cluster labels, cluster statistics)
        """
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.hidden_vectors)

        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_vectors = self.hidden_vectors[cluster_labels == i]
            cluster_stats[f'Cluster_{i+1}'] = {
                'size': int(len(cluster_vectors)),
                'mean': np.mean(cluster_vectors, axis=0).tolist(),
                'std': np.std(cluster_vectors, axis=0).tolist(),
                'centroid': kmeans.cluster_centers_[i].tolist()
            }

        return cluster_labels, cluster_stats

    def analyze_latent_space(self) -> Dict:
        """
        Analyze the structure and properties of the latent space.

        Returns:
            Dictionary containing latent space analysis results
        """
        # Analyze distribution of latent dimensions
        latent_stats = {
            'mean': np.mean(self.hidden_vectors, axis=0).tolist(),
            'std': np.std(self.hidden_vectors, axis=0).tolist(),
            'min': np.min(self.hidden_vectors, axis=0).tolist(),
            'max': np.max(self.hidden_vectors, axis=0).tolist(),
            'active_dimensions': int(np.sum(np.std(self.hidden_vectors, axis=0) > 0.01)),
            'dimension_correlations': self.corrcoef_list(self.hidden_vectors)
        }

        return latent_stats

    def corrcoef_list(self, data: np.ndarray) -> List[List[float]]:
        """
        Compute the correlation coefficient matrix and convert it to a list.

        Args:
            data: 2D numpy array

        Returns:
            Correlation matrix as a list of lists
        """
        if data.shape[0] < 2:
            return []
        corr_matrix = np.corrcoef(data.T)
        return corr_matrix.tolist()

    def plot_latent_space_visualization(self, method: str = 'tsne') -> go.Figure:
        """
        Create interactive visualization of latent space.

        Args:
            method: Dimensionality reduction method ('tsne', 'umap', or 'pca')

        Returns:
            Plotly figure object
        """
        reduced_vectors = self.reduce_dimensions(method=method)
        cluster_labels, _ = self.cluster_vectors()

        df = pd.DataFrame({
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'z': reduced_vectors[:, 2] if reduced_vectors.shape[1] > 2 else 0,
            'cluster': cluster_labels.astype(str)  # Convert to string for better color handling
        })

        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='cluster',
            title=f'Latent Space Visualization ({method.upper()})',
            template='plotly_dark',
            width=1000, height=800  # Increased size
        )
        return fig

    def plot_latent_space_clusters(self, method: str = 'tsne', n_clusters: int = 5) -> go.Figure:
        """
        Plot latent space with clustering overlay.

        Args:
            method: Dimensionality reduction method ('tsne', 'umap', or 'pca')
            n_clusters: Number of clusters to display

        Returns:
            Plotly figure object
        """
        reduced_vectors = self.reduce_dimensions(method=method)
        cluster_labels, _ = self.cluster_vectors(n_clusters=n_clusters)

        df = pd.DataFrame({
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'z': reduced_vectors[:, 2] if reduced_vectors.shape[1] > 2 else 0,
            'cluster': cluster_labels.astype(str)
        })

        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='cluster',
            title=f'Latent Space Visualization with {n_clusters} Clusters ({method.upper()})',
            template='plotly_dark',
            width=1000, height=800  # Increased size
        )
        return fig

# ===============================
# 9. 3D Plot Generation Function
# ===============================

def generate_3d_plot(latent_vectors: np.ndarray, cluster_labels: np.ndarray = None,
                    azimuth: float = 45, elevation: float = 30, zoom: float = 1.5) -> go.Figure:
    """
    Generate an interactive 3D scatter plot with camera controls.

    Args:
        latent_vectors (np.ndarray): The latent vectors to plot (N x 3).
        cluster_labels (np.ndarray, optional): Cluster labels for coloring. Defaults to None.
        azimuth (float): Azimuth angle in degrees.
        elevation (float): Elevation angle in degrees.
        zoom (float): Zoom level.

    Returns:
        go.Figure: Plotly 3D scatter plot with updated camera position.
    """
    if latent_vectors.shape[1] < 3:
        raise ValueError("Latent vectors must have at least 3 dimensions for 3D plotting.")

    if cluster_labels is not None:
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Dark24
        color_mapping = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        marker_colors = [color_mapping[label] for label in cluster_labels]
    else:
        marker_colors = 'rgba(135, 206, 250, 0.8)'  # Light sky blue

    # Convert azimuth and elevation to radians
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Calculate camera position
    x = zoom * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = zoom * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = zoom * np.sin(elevation_rad)

    fig = go.Figure(data=[go.Scatter3d(
        x=latent_vectors[:, 0],
        y=latent_vectors[:, 1],
        z=latent_vectors[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=marker_colors,
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Latent Dimension 3',
            camera=dict(
                eye=dict(x=x, y=y, z=z)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=800,  # Increased height
        width=1000   # Increased width
    )

    return fig

# ===============================
# 10. Analysis Function
# ===============================

def analyze_eeg_model(hidden_vectors_file, model_file, reduction_method, n_clusters,
                      azimuth, elevation, zoom):
    """
    Function to handle the analysis process and generate 3D plots.

    Args:
        hidden_vectors_file: Uploaded .npy file content as bytes
        model_file: Uploaded .pth file content as bytes (state_dict)
        reduction_method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        n_clusters: Number of clusters for KMeans
        azimuth: Azimuth angle for 3D plot
        elevation: Elevation angle for 3D plot
        zoom: Zoom level for 3D plot

    Returns:
        Tuple containing 3D plot, clustered 3D plot, and analysis results
    """
    try:
        # Verify files were uploaded
        if hidden_vectors_file is None or model_file is None:
            raise ValueError("Please upload both hidden vectors (.npy) and model (.pth) files.")

        # Save uploaded files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_vectors:
            tmp_vectors.write(hidden_vectors_file)  # hidden_vectors_file is bytes
            vectors_path = tmp_vectors.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_model:
            tmp_model.write(model_file)  # model_file is bytes
            model_path = tmp_model.name

        try:
            # Load hidden vectors
            hidden_vectors = np.load(vectors_path)

            # Define fixed model parameters (must match training)
            channels = 5
            frequency_bands = 7
            latent_dim = 64

            # Initialize the model with fixed parameters
            model = EEGAutoencoder(channels=channels, frequency_bands=frequency_bands, latent_dim=latent_dim)

            # Load the state dictionary
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            # Initialize Analyzer
            analyzer = EEGAnalyzer(hidden_vectors, model)

            # Perform Analyses
            latent_space_analysis = analyzer.analyze_latent_space()
            cluster_labels, cluster_stats = analyzer.cluster_vectors(n_clusters=int(n_clusters))

            # Perform Dimensionality Reduction if needed
            if hidden_vectors.shape[1] > 3:
                reduced_vectors = analyzer.reduce_dimensions(method=reduction_method, n_components=3)
            else:
                reduced_vectors = hidden_vectors

            # Generate 3D Plots with Camera Controls
            fig = generate_3d_plot(reduced_vectors, cluster_labels,
                                    azimuth=azimuth, elevation=elevation, zoom=zoom)

            # Compile Analysis Results
            analysis_results = {
                "Latent Space Analysis": latent_space_analysis,
                "Cluster Statistics": cluster_stats
            }

            logging.info("EEG Model Analysis Completed Successfully.")

            return fig, fig, analysis_results  # Returning the same figure twice for simplicity

        finally:
            # Clean up temporary files
            os.unlink(vectors_path)
            os.unlink(model_path)

    except Exception as e:
        logging.error(f"Error during EEG model analysis: {str(e)}")
        raise gr.Error(f"An error occurred: {str(e)}")

# ===============================
# 11. Gradio Interface
# ===============================

def create_gradio_interface():
    with gr.Blocks(title="EEG Autoencoder Explorer") as app:
        gr.Markdown("""
        # EEG Autoencoder Explorer

        Upload your **trained model (.pth)** and **hidden vectors (.npy)** to analyze and visualize the latent space.
        Use the controls to explore the 3D visualization of the latent vectors.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                hidden_vectors_input = gr.File(
                    label="Hidden Vectors (.npy)",
                    file_types=[".npy"],
                    type="binary"  # Correct type
                )
                model_input = gr.File(
                    label="Trained Model (.pth)",
                    file_types=[".pth"],
                    type="binary"  # Correct type
                )
                reduction_method = gr.Radio(
                    choices=["tsne", "umap", "pca"],
                    value="tsne",
                    label="Dimensionality Reduction Method"
                )
                n_clusters_input = gr.Number(
                    value=5,
                    label="Number of Clusters for KMeans",
                    precision=0
                )
                azimuth_slider = gr.Slider(
                    minimum=0,
                    maximum=360,
                    step=1,
                    value=45,
                    label="Azimuth Angle (Degrees)"
                )
                elevation_slider = gr.Slider(
                    minimum=-90,
                    maximum=90,
                    step=1,
                    value=30,
                    label="Elevation Angle (Degrees)"
                )
                zoom_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=1.5,
                    label="Zoom Level"
                )
                analyze_button = gr.Button("Analyze and Explore")

            with gr.Column(scale=2):
                latent_space_plot = gr.Plot(label="Latent Space 3D Visualization")
                clustered_latent_space_plot = gr.Plot(label="Latent Space with Clusters")
                analysis_output = gr.JSON(label="Analysis Results")

        analyze_button.click(
            fn=analyze_eeg_model,
            inputs=[hidden_vectors_input, model_input, reduction_method, n_clusters_input,
                    azimuth_slider, elevation_slider, zoom_slider],
            outputs=[latent_space_plot, clustered_latent_space_plot, analysis_output]
        )

        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. **Upload Files:**
           - **Hidden Vectors (`.npy`):** These are the latent vectors extracted from your EEG data using the autoencoder.
           - **Trained Model (`.pth`):** The state dictionary of your trained EEG autoencoder model.

        2. **Configure Parameters:**
           - **Dimensionality Reduction Method:** Choose between `tsne`, `umap`, or `pca`. This is necessary if your latent vectors have more than 3 dimensions.
           - **Number of Clusters:** Specify how many clusters KMeans should identify in the latent space.

        3. **Adjust Camera Controls:**
           - **Azimuth Angle:** Rotate the plot horizontally.
           - **Elevation Angle:** Tilt the plot vertically.
           - **Zoom Level:** Zoom in or out of the plot.

        4. **Analyze and Explore:**
           - Click the **Analyze and Explore** button to generate the visualization and analysis results.
           - The **Latent Space 3D Visualization** will display the scatter plot with clusters.
           - **Analysis Results** will show cluster statistics, including size and centroids.
        """)

    return app

# ===============================
# 12. Main Execution
# ===============================

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
