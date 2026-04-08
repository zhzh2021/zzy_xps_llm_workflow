"""
XAS Variational Autoencoder (VAE) Module

Compress spectra into a non-linear latent space to capture subtle shifts
in pre-edge or white-line features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from .config_utils import ConfigLoader


@dataclass
class VAEEmbeddingResult:
    embedding: np.ndarray
    recon_error: float
    latent_dim: int
    n_epochs: int
    batch_size: int


class _VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class XASVAEEmbedding:
    """
    Train a VAE on spectra and return latent embeddings.
    """

    def __init__(self, config_path: Optional[Path] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for VAE. Install with: pip install torch")

        self.config = ConfigLoader(config_path)
        self.vae_config = self.config.get_section('vae') if 'vae' in self.config.get_all() else {}

    def _prepare(self, spectra: np.ndarray) -> np.ndarray:
        # Standardize each spectrum (zero mean, unit variance)
        mean = np.mean(spectra, axis=1, keepdims=True)
        std = np.std(spectra, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (spectra - mean) / std

    def fit_transform(self, spectra: np.ndarray) -> VAEEmbeddingResult:
        """
        Train VAE and return latent embeddings.

        Args:
            spectra: (n_samples x n_energy_points) array
        """
        spectra = self._prepare(spectra)
        n_samples, input_dim = spectra.shape

        latent_dim = int(self.vae_config.get('latent_dim', 2))
        hidden_dim = int(self.vae_config.get('hidden_dim', 128))
        batch_size = int(self.vae_config.get('batch_size', 16))
        n_epochs = int(self.vae_config.get('n_epochs', 200))
        lr = float(self.vae_config.get('learning_rate', 1e-3))
        beta = float(self.vae_config.get('beta', 1.0))
        seed = int(self.vae_config.get('random_seed', 42))

        torch.manual_seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = _VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        x = torch.tensor(spectra, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(n_epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction='mean')
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kld
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            x_dev = x.to(device)
            recon, mu, _ = model(x_dev)
            recon_err = nn.functional.mse_loss(recon, x_dev, reduction='mean').item()
            embedding = mu.cpu().numpy()

        return VAEEmbeddingResult(
            embedding=embedding,
            recon_error=recon_err,
            latent_dim=latent_dim,
            n_epochs=n_epochs,
            batch_size=batch_size
        )
