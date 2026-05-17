"""
LSTM forecasting model using PyTorch.

Architecture:
  - 2-layer LSTM (128 → 64 hidden units)
  - Dropout for regularization
  - Dense output layer
  - Trained with Adam + MSE loss + early stopping
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.preprocessor import DataScaler, create_sequences

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# PyTorch LSTM Network
# ──────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class LSTMNetwork(nn.Module):
        def __init__(self, input_size=1, hidden1=128, hidden2=64, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden2, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            out = self.fc(out[:, -1, :])  # take last time step
            return out


# ──────────────────────────────────────────────────────────────
# LSTM Forecaster Class
# ──────────────────────────────────────────────────────────────

class LSTMForecaster:
    """LSTM-based stock price forecaster."""

    def __init__(
        self,
        seq_length: int = 60,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 10,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model. Install with: pip install torch")

        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.scaler = DataScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._last_sequence = None

    def fit(self, train_series: pd.Series) -> 'LSTMForecaster':
        """Train LSTM on the training series (Close prices)."""
        # Scale data
        scaled = self.scaler.fit_transform(train_series)

        # Create sequences
        X, y = create_sequences(scaled, self.seq_length)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, seq_len, features)

        # Store last sequence for forecasting
        self._last_sequence = scaled[-self.seq_length:]

        # Convert to tensors
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = LSTMNetwork().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, steps: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recursive multi-step forecast.
        Returns (forecast, lower_estimate, upper_estimate).
        Lower/upper are simple ±2% bands (LSTM doesn't natively produce CIs).
        """
        self.model.eval()
        current_seq = self._last_sequence.copy()
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                x = torch.FloatTensor(current_seq.reshape(1, self.seq_length, 1)).to(self.device)
                pred = self.model(x).cpu().numpy()[0, 0]
                predictions.append(pred)
                # Shift window: drop first, append prediction
                current_seq = np.append(current_seq[1:], [[pred]], axis=0)

        # Inverse transform
        preds_scaled = np.array(predictions).reshape(-1, 1)
        preds = self.scaler.inverse_transform(preds_scaled).flatten()

        # Simple confidence bands (±2% of prediction)
        margin = preds * 0.02
        return preds, preds - margin, preds + margin

    def predict_in_sample(self, test_series: pd.Series, full_series: pd.Series) -> np.ndarray:
        """
        Predict on test set using rolling window from full series.
        """
        self.model.eval()
        full_scaled = self.scaler.transform(full_series)
        test_start_idx = len(full_series) - len(test_series)
        predictions = []

        with torch.no_grad():
            for i in range(test_start_idx, len(full_scaled)):
                seq = full_scaled[i - self.seq_length:i]
                x = torch.FloatTensor(seq.reshape(1, self.seq_length, 1)).to(self.device)
                pred = self.model(x).cpu().numpy()[0, 0]
                predictions.append(pred)

        preds = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        return preds
