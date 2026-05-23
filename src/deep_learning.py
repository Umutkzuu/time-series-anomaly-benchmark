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

# ==========================================
# 1. VERİ HAZIRLIĞI (SLIDING WINDOW)
# ==========================================
def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    
    for i in range(len(X) - seq_length):
        seq = X[i:i+seq_length]
        # Hedef etiket: Pencerenin içindeki maksimum anomali (1) değeri
        label = 1 if np.max(y[i:i+seq_length]) == 1 else 0
        sequences.append(seq)
        labels.append(label)
        
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

# ==========================================
# 2. MODEL MİMARİLERİ (1D-CNN & GRU)
# ==========================================
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_features, seq_length):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Flatten sonrası boyut
        flattened_size = 32 * (seq_length // 2)
        self.fc1 = nn.Linear(flattened_size, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Features, Seq)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

class TimeSeriesGRU(nn.Module):
    def __init__(self, input_features, hidden_size=64, num_layers=2):
        super(TimeSeriesGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_features, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==========================================
# 3. EĞİTİM VE TEST MOTORU (TRAINING ENGINE)
# ==========================================
def train_model(model, train_loader, epochs=50, lr=0.001):
    model = model.to(DEVICE)
    
    # Sınıf Dengesizliğini (Class Imbalance) çözmek için Pozitif Sınıfın cezasını 10 kat artırıyoruz
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Model {DEVICE} üzerinde eğitiliyor...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Kayıp (Loss): {total_loss/len(train_loader):.4f}")
            
    return model

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            
            probs = torch.sigmoid(outputs.squeeze())
            # Eşik değerini 0.5'ten 0.3'e çekerek modelin daha rahat "1" demesini sağlıyoruz
            preds = (probs > 0.3).int().cpu().numpy()
            
            if preds.ndim == 0:
                preds = [preds.item()]
                y_batch = [y_batch.item()]
            
            y_pred_list.extend(preds)
            y_true.extend(y_batch.numpy())
            
    f1 = f1_score(y_true, y_pred_list, zero_division=0)
    prec = precision_score(y_true, y_pred_list, zero_division=0)
    rec = recall_score(y_true, y_pred_list, zero_division=0)
    
    return f1, prec, rec