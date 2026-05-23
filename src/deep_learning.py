import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

# Mac (Apple Silicon) ve diğer donanımlar için otomatik GPU tespiti
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def set_seed(seed):
    """Tüm rastgelelik süreçlerini sabitler (Akademik Reproducibility)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        flattened_size = 32 * (seq_length // 2)
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = x.flatten(1)
        out = self.fc2(self.relu(self.fc1(x)))
        return out

class TimeSeriesGRU(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers):
        super(TimeSeriesGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_features, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

def train_model(model, train_loader, val_loader, cfg):
    """Erken durdurma (Early Stopping) ve dinamik parametrelerle eğitim döngüsü."""
    dl_cfg = cfg["dl_params"]
    epochs = dl_cfg["epochs"]
    lr = dl_cfg["learning_rate"]
    patience = dl_cfg["early_stopping_patience"]
    
    model = model.to(DEVICE)
    # Sınıf dengesizliği (Imbalance) çözümü
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # --- EĞİTİM (TRAIN) ---
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(DEVICE))
            loss = criterion(outputs.squeeze(), y_batch.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        # --- DOĞRULAMA (VALIDATION) VE EARLY STOPPING ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                out_val = model(X_val.to(DEVICE))
                val_loss += criterion(out_val.squeeze(), y_val.to(DEVICE)).item()
        
        val_loss /= len(val_loader)
        
        # Eğer loss düştüyse modeli kaydet (şart değil ama mantığını koruyoruz), sayacı sıfırla
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            # Loss düşmediyse sabır sayacını artır
            patience_counter += 1
            if patience_counter >= patience:
                # Early Stopping tetiklendi!
                break 
                
    return model

def evaluate_model(model, test_loader):
    """Test setinde F1, Precision ve Recall değerlerini hesaplar."""
    model.eval()
    y_true, y_pred_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(DEVICE))
            preds = (torch.sigmoid(outputs.squeeze()) > 0.3).int().cpu().numpy()
            if preds.ndim == 0:
                preds = [preds.item()]
                y_batch = [y_batch.item()]
            y_pred_list.extend(preds)
            y_true.extend(y_batch.numpy())
            
    return f1_score(y_true, y_pred_list, zero_division=0), precision_score(y_true, y_pred_list, zero_division=0), recall_score(y_true, y_pred_list, zero_division=0)