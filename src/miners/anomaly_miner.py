"""
Anomaly-Based Bias Detection Using PyTorch
───────────────────────────────────────────
Uses a PyTorch autoencoder on response embeddings to detect
demographic groups that receive anomalously different responses.

Intuition: If we train an autoencoder on all response embeddings,
responses to certain demographic groups will have higher
reconstruction error — indicating the model treats them differently.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class AnomalyGroup:
    """A demographic group flagged for anomalous LLM behavior."""
    group_key: dict[str, str]
    size: int
    mean_anomaly_score: float
    median_anomaly_score: float
    population_mean_score: float
    z_score: float  # How many SDs above population mean
    anomaly_fraction: float  # Fraction of group above threshold
    example_indices: list[int]  # Indices of most anomalous examples

    def __str__(self):
        desc = " & ".join(f"{k}={v}" for k, v in self.group_key.items())
        return (
            f"[{desc}] (n={self.size})\n"
            f"  anomaly_score: mean={self.mean_anomaly_score:.4f} "
            f"(pop={self.population_mean_score:.4f})\n"
            f"  z_score={self.z_score:.2f} | "
            f"anomaly_fraction={self.anomaly_fraction:.1%}"
        )


class ResponseAutoencoder(nn.Module):
    """PyTorch autoencoder for learning normal response patterns.

    Architecture: symmetric encoder-decoder with configurable hidden dims.
    The bottleneck forces the model to learn a compressed representation
    of "normal" response patterns. Abnormal responses (those directed
    at certain demographics) will have higher reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 384,
        encoder_dims: list[int] = None,
        decoder_dims: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_dims = encoder_dims or [128, 32]
        decoder_dims = decoder_dims or [32, 128]

        # Build encoder
        enc_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder
        dec_layers = []
        prev_dim = encoder_dims[-1]
        for dim in decoder_dims:
            dec_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        # Final reconstruction layer (no activation)
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_representation)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error."""
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


class AnomalyBiasMiner:
    """Detect bias through anomalous response patterns.

    Pipeline:
    1. Get response embeddings (from ResponseAnalyzer)
    2. Train autoencoder to reconstruct embeddings
    3. Compute reconstruction error per response
    4. Group by demographics → find groups with high error
    5. Statistical testing to confirm significance

    High reconstruction error for a demographic group means the
    model's responses to that group are "unusual" compared to the
    overall response distribution — a signal of differential treatment.
    """

    DEMOGRAPHIC_COLUMNS = [
        "gender", "race_ethnicity", "age_group", "socioeconomic", "religion"
    ]

    def __init__(
        self,
        encoder_dims: list[int] = None,
        decoder_dims: list[int] = None,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        anomaly_percentile: float = 95,
        device: Optional[str] = None,
    ):
        self.encoder_dims = encoder_dims or [128, 32]
        self.decoder_dims = decoder_dims or [32, 128]
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.anomaly_percentile = anomaly_percentile

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[ResponseAutoencoder] = None

    def _train_autoencoder(self, embeddings: np.ndarray) -> ResponseAutoencoder:
        """Train the autoencoder on response embeddings."""
        input_dim = embeddings.shape[1]
        model = ResponseAutoencoder(
            input_dim=input_dim,
            encoder_dims=self.encoder_dims,
            decoder_dims=self.decoder_dims,
        ).to(self.device)

        # Prepare data
        tensor_data = torch.FloatTensor(embeddings).to(self.device)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        print(f"Training autoencoder ({input_dim}→{self.encoder_dims}→{input_dim}) "
              f"on {self.device}")

        for epoch in range(self.epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstruction, _ = model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            avg_loss = total_loss / len(embeddings)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return model

    def _compute_anomaly_scores(
        self, model: ResponseAutoencoder, embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute reconstruction error for each response."""
        model.eval()
        tensor_data = torch.FloatTensor(embeddings).to(self.device)

        scores = []
        with torch.no_grad():
            for i in range(0, len(tensor_data), self.batch_size):
                batch = tensor_data[i : i + self.batch_size]
                errors = model.get_reconstruction_error(batch)
                scores.extend(errors.cpu().numpy())

        return np.array(scores)

    def detect(
        self,
        df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
    ) -> list[AnomalyGroup]:
        """Run anomaly-based bias detection.

        Args:
            df: Enriched DataFrame from ResponseAnalyzer.
            embeddings: Pre-computed embeddings. If None, tries df.attrs["embeddings"].

        Returns:
            List of AnomalyGroup objects for flagged demographic groups.
        """
        # Get embeddings
        if embeddings is None:
            embeddings = df.attrs.get("embeddings")
        if embeddings is None:
            raise ValueError(
                "No embeddings found. Run ResponseAnalyzer.analyze() with "
                "compute_embeddings=True first, or pass embeddings directly."
            )

        print(f"Anomaly detection on {len(embeddings)} response embeddings")

        # Step 1: Train autoencoder
        self.model = self._train_autoencoder(embeddings)

        # Step 2: Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(self.model, embeddings)
        df = df.copy()
        df["anomaly_score"] = anomaly_scores

        # Step 3: Compute threshold
        threshold = np.percentile(anomaly_scores, self.anomaly_percentile)
        df["is_anomaly"] = anomaly_scores > threshold
        population_mean = np.mean(anomaly_scores)
        population_std = np.std(anomaly_scores)

        print(f"  Anomaly threshold (p{self.anomaly_percentile}): {threshold:.6f}")
        print(f"  Population: mean={population_mean:.6f}, std={population_std:.6f}")

        # Step 4: Group by demographics and analyze
        demo_cols = [c for c in self.DEMOGRAPHIC_COLUMNS if c in df.columns]
        results = []

        # Check single attributes and pairs
        from itertools import combinations as combos

        for depth in range(1, 3):
            for col_combo in combos(demo_cols, depth):
                groups = df.groupby(list(col_combo))
                for group_vals, group_df in groups:
                    if len(group_df) < 10:
                        continue

                    if isinstance(group_vals, str):
                        group_vals = (group_vals,)

                    group_key = dict(zip(col_combo, group_vals))
                    group_scores = group_df["anomaly_score"].values

                    mean_score = np.mean(group_scores)
                    z_score = (mean_score - population_mean) / max(population_std, 1e-8)
                    anomaly_frac = np.mean(group_df["is_anomaly"])

                    # Flag if significantly anomalous
                    if z_score > 1.5 or anomaly_frac > (100 - self.anomaly_percentile) / 100 * 2:
                        # Get most anomalous examples
                        top_indices = group_df.nlargest(3, "anomaly_score").index.tolist()

                        results.append(AnomalyGroup(
                            group_key=group_key,
                            size=len(group_df),
                            mean_anomaly_score=float(mean_score),
                            median_anomaly_score=float(np.median(group_scores)),
                            population_mean_score=float(population_mean),
                            z_score=float(z_score),
                            anomaly_fraction=float(anomaly_frac),
                            example_indices=top_indices,
                        ))

        results.sort(key=lambda x: x.z_score, reverse=True)
        print(f"  Flagged {len(results)} anomalous demographic groups")
        return results

    def results_to_dataframe(self, groups: list[AnomalyGroup]) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for g in groups:
            desc = " & ".join(f"{k}={v}" for k, v in g.group_key.items())
            rows.append({
                "group": desc,
                "size": g.size,
                "mean_anomaly_score": g.mean_anomaly_score,
                "population_mean": g.population_mean_score,
                "z_score": g.z_score,
                "anomaly_fraction": g.anomaly_fraction,
            })
        return pd.DataFrame(rows)

    def summarize(self, groups: list[AnomalyGroup]) -> str:
        """Generate summary."""
        if not groups:
            return "No anomalous demographic groups detected."

        lines = [f"Flagged {len(groups)} anomalous demographic groups.\n"]
        for i, g in enumerate(groups[:5], 1):
            lines.append(f"  {i}. {g}\n")
        return "\n".join(lines)
