import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import ttest_ind # RUBRİK: İSTATİSTİKSEL TEST İÇİN
import warnings

import data_loader
import preprocessing
import automata
import deep_learning

warnings.filterwarnings('ignore')

def compress_labels(y, window_size):
    y_clean = (y == 1).astype(int)
    n = len(y_clean)
    remainder = n % window_size
    if remainder != 0:
        y_clean = y_clean[:-remainder]
    return y_clean.reshape(-1, window_size).max(axis=1) 

def run_pipeline():
    print("1. Veriler Çekiliyor ve Temizleniyor...\n")
    cfg = data_loader.load_config()
    batadal_df = data_loader.load_batadal_data(cfg)
    res_batadal = preprocessing.prep_batadal(batadal_df, cfg)
    
    X_train_pca = res_batadal["pca"][0]
    X_test_pca = res_batadal["pca"][2]
    
    y_train_clean = (res_batadal["y"][0] == 1).astype(int)
    y_test_clean = (res_batadal["y"][2] == 1).astype(int)
    
    print("="*60)
    print(" BÖLÜM 1: OLASILIKSAL OTOMATA (EXPLAINABILITY) ")
    print("="*60)
    w_size = 3
    a_size = 8
    
    y_test_compressed = compress_labels(y_test_clean, w_size)
    
    paa_train = automata.apply_paa(X_train_pca, window_size=w_size)
    paa_test = automata.apply_paa(X_test_pca, window_size=w_size)
    
    sax_train = automata.apply_sax(paa_train, alphabet_size=a_size)
    sax_test = automata.apply_sax(paa_test, alphabet_size=a_size)
    
    t_matrix = automata.build_transition_matrix(sax_train, alphabet_size=a_size)
    
    train_scores = automata.get_sequence_scores(sax_train, t_matrix, a_size)
    threshold = np.percentile(train_scores, 95)
    
    test_scores = automata.get_sequence_scores(sax_test, t_matrix, a_size)
    y_pred_automata = (test_scores > threshold).astype(int)
    
    print(f"Otomata F1-Score  : {f1_score(y_test_compressed, y_pred_automata, zero_division=0):.4f}")
    
    # --- YENİ: İSTATİSTİKSEL ANALİZ (T-TEST) ---
    print("\n--- İSTATİSTİKSEL ANALİZ (T-TEST) ---")
    scores_normal = test_scores[y_test_compressed == 0]
    scores_anomaly = test_scores[y_test_compressed == 1]
    
    # Welch's T-Test (Varyansların eşit olmadığını varsayar)
    t_stat, p_value = ttest_ind(scores_normal, scores_anomaly, equal_var=False)
    print(f"T-Statistic Değeri : {t_stat:.4f}")
    print(f"P-Value Değeri     : {p_value:.4e}")
    if p_value < 0.05:
        print("SONUÇ: P-Value < 0.05 olduğu için modelimizin Normal ve Anomali durumlarını birbirinden matematiksel olarak AYIRABİLDİĞİ kanıtlanmıştır!")
    else:
        print("SONUÇ: İstatistiksel olarak anlamlı bir fark bulunamadı.")

    # --- YENİ: AÇIKLANABİLİRLİK (EXPLAINABILITY) RAPORU ---
    print("\n--- AÇIKLANABİLİRLİK (EXPLAINABILITY) RAPORU ---")
    print(f"Sistem Eşiği (Threshold): {threshold:.4f}\n")
    
    # İlk 3 Anomali tespitini raporla
    anomali_sayaci = 0
    for idx, (score, pred, true_label) in enumerate(zip(test_scores, y_pred_automata, y_test_compressed)):
        if pred == 1 and true_label == 1: # True Positive yakaladık
            anomali_sayaci += 1
            # O anki kelime kalıbını (pattern) çıkar (Geçmiş 3 harf)
            start_idx = max(0, idx - 3)
            pattern = "".join(sax_test[start_idx:idx+1])
            confidence = automata.calculate_confidence(score, threshold)
            
            print(f"Örnek {anomali_sayaci} (Zaman Adımı: {idx}):")
            print(f"  Gözlemlenen Pattern : '{pattern}'")
            print(f"  Path Probability    : {score:.4f}")
            print(f"  Model Kararı        : SİBER SALDIRI (Anomali)")
            print(f"  Güven Skoru         : %{confidence:.2f} Emin")
            print(f"  Gerçek Durum        : Doğru Tespit (True Positive)\n")
            
            if anomali_sayaci >= 3:
                break

    print("="*60)
    print(" BÖLÜM 2: DERİN ÖĞRENME (BLACK-BOX MODELS) ")
    print("="*60)
    
    seq_length = 5
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    X_train_dl, y_train_dl = deep_learning.create_sequences(X_train_pca, y_train_clean, seq_length)
    X_test_dl, y_test_dl = deep_learning.create_sequences(X_test_pca, y_test_clean, seq_length)
    
    train_loader = DataLoader(TensorDataset(X_train_dl, y_train_dl), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_dl, y_test_dl), batch_size=batch_size, shuffle=False)
    
    print("[ 1D-CNN EĞİTİMİ (Özet) ]")
    cnn_model = deep_learning.TimeSeriesCNN(input_features=1, seq_length=seq_length)
    cnn_model = deep_learning.train_model(cnn_model, train_loader, epochs=epochs, lr=lr)
    cnn_f1, cnn_prec, cnn_rec = deep_learning.evaluate_model(cnn_model, test_loader)
    print(f"> 1D-CNN F1-Score : {cnn_f1:.4f}\n")

    print("[ GRU EĞİTİMİ (Özet) ]")
    gru_model = deep_learning.TimeSeriesGRU(input_features=1, hidden_size=64, num_layers=2)
    gru_model = deep_learning.train_model(gru_model, train_loader, epochs=epochs, lr=lr)
    gru_f1, gru_prec, gru_rec = deep_learning.evaluate_model(gru_model, test_loader)
    print(f"> GRU F1-Score    : {gru_f1:.4f}\n")
    
    print("="*60)
    print(" PROJE BAŞARIYLA TAMAMLANDI! ")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()