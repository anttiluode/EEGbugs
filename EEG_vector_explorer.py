import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, List
import pandas as pd
import gradio as gr
import io
import logging
import tempfile
import os

# ===============================
# 1. Logging Configuration
# ===============================

logging.basicConfig(
    filename='eeg_bug_simulator.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ===============================
# 2. Model Definition
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
        # Calculate the size after convolutions and pooling
        self.fc1 = nn.Linear(32 * channels * (frequency_bands // 4), latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * channels * (frequency_bands // 4))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(-1,32,5,1)  # Reshape for ConvTranspose2d
        x = self.decoder(x)
        x = x.squeeze(1)  # Remove channel dimension
        return x, latent

# ===============================
# 3. EEGAnalyzer Class
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

    def reduce_dimensions(self, method: str = 'tsne', n_components: int = 2) -> np.ndarray:
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
            cluster_stats[f'Cluster_{i}'] = {
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
            'cluster': cluster_labels.astype(str)  # Convert to string for better color handling
        })

        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            title=f'Latent Space Visualization ({method.upper()})',
            template='plotly_dark',
            width=800, height=600
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
            'cluster': cluster_labels.astype(str)
        })

        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            title=f'Latent Space Visualization with {n_clusters} Clusters ({method.upper()})',
            template='plotly_dark',
            width=800, height=600
        )
        return fig

# ===============================
# 4. Analysis Function
# ===============================

def analyze_eeg_model(hidden_vectors_file, model_file, reduction_method, n_clusters):
    """
    Function to handle the analysis process.

    Args:
        hidden_vectors_file: Uploaded .npy file content as bytes
        model_file: Uploaded .pth file content as bytes (state_dict)
        reduction_method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        n_clusters: Number of clusters for KMeans

    Returns:
        Tuple containing latent space plot, clustered latent space plot, and analysis results
    """
    try:
        # Verify files were uploaded
        if hidden_vectors_file is None or model_file is None:
            raise ValueError("Please upload both hidden vectors (.npy) and model (.pth) files.")

        # Save uploaded files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_vectors:
            tmp_vectors.write(hidden_vectors_file)  # Directly write bytes
            vectors_path = tmp_vectors.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_model:
            tmp_model.write(model_file)  # Directly write bytes
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

            # Generate Visualizations
            latent_space_fig = analyzer.plot_latent_space_visualization(method=reduction_method)
            clustered_latent_space_fig = analyzer.plot_latent_space_clusters(
                method=reduction_method,
                n_clusters=int(n_clusters)
            )

            # Compile Analysis Results
            analysis_results = {
                "Latent Space Analysis": latent_space_analysis,
                "Cluster Statistics": cluster_stats
            }

            logging.info("EEG Model Analysis Completed Successfully.")

            return latent_space_fig, clustered_latent_space_fig, analysis_results

        finally:
            # Clean up temporary files
            os.unlink(vectors_path)
            os.unlink(model_path)

    except Exception as e:
        logging.error(f"Error during EEG model analysis: {str(e)}")
        raise gr.Error(f"An error occurred: {str(e)}")

# ===============================
# 5. Gradio Interface
# ===============================

def create_gradio_interface():
    with gr.Blocks(title="EEG Autoencoder Analyzer") as app:
        gr.Markdown("""
        # EEG Bug Simulator Analysis Tools

        Upload your **hidden vectors (.npy)** and **trained model (.pth)** to analyze the model and visualize the latent space.
        """)

        with gr.Row():
            with gr.Column():
                hidden_vectors_input = gr.File(
                    label="Hidden Vectors (.npy)",
                    file_types=[".npy"],
                    type="binary"  # Correctly specify 'binary'
                )
                model_input = gr.File(
                    label="Trained Model (.pth)",
                    file_types=[".pth"],
                    type="binary"  # Correctly specify 'binary'
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
                analyze_button = gr.Button("Analyze Model")

            with gr.Column():
                latent_space_plot = gr.Plot(label="Latent Space Visualization")
                clustered_latent_space_plot = gr.Plot(label="Latent Space with Clusters")
                analysis_output = gr.JSON(label="Analysis Results")

        analyze_button.click(
            fn=analyze_eeg_model,
            inputs=[hidden_vectors_input, model_input, reduction_method, n_clusters_input],
            outputs=[latent_space_plot, clustered_latent_space_plot, analysis_output]
        )

        gr.Markdown("### Logs")
        log_output = gr.Textbox(
            label="Log Output",
            lines=10,
            interactive=False,
            placeholder="Logs are being written to 'eeg_bug_simulator.log'."
        )

    return app

# ===============================
# 6. Main Execution
# ===============================

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)
