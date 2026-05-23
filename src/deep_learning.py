import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DECISION_THRESHOLD = 0.15
warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        seq = X[i:i+seq_length]
        label = 1 if np.max(y[i:i+seq_length]) == 1 else 0
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_features, seq_length, hidden_size=16):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        flattened_size = 16 * (seq_length // 2)
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.flatten(1)
        return self.fc2(self.relu(self.fc1(x)))

class TimeSeriesGRU(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers):
        super(TimeSeriesGRU, self).__init__()
        self.hidden_size = 32  # 64'ten küçülttük
        self.num_layers = 1    # 2'den küçülttük
        self.gru = nn.GRU(input_features, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        return self.fc(self.dropout(out[:, -1, :]))

def train_model(model, train_loader, val_loader, cfg):
    dl_cfg = cfg["dl_params"]
    epochs = dl_cfg["epochs"]
    lr = dl_cfg["learning_rate"]
    patience = dl_cfg["early_stopping_patience"]
    pw = dl_cfg.get("pos_weight", 10.0)
    model = model.to(DEVICE)
    pos_weight = torch.tensor([pw]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(DEVICE))
            loss = criterion(outputs.squeeze(), y_batch.to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                out_val = model(X_val.to(DEVICE))
                val_loss += criterion(out_val.squeeze(), y_val.to(DEVICE)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(DEVICE))
            preds = (torch.sigmoid(outputs.squeeze()) > DECISION_THRESHOLD).int().cpu().numpy()
            if preds.ndim == 0:
                preds = [preds.item()]
                y_batch = [y_batch.item()]
            y_pred_list.extend(preds)
            y_true.extend(y_batch.numpy())
    return f1_score(y_true, y_pred_list, zero_division=0), precision_score(y_true, y_pred_list, zero_division=0), recall_score(y_true, y_pred_list, zero_division=0)