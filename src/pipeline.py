import numpy as np
import json
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import automata
import deep_learning

def compress_labels(y, window_size):
    y_clean = (y == 1).astype(int)
    remainder = len(y_clean) % window_size
    if remainder != 0: y_clean = y_clean[:-remainder]
    return y_clean.reshape(-1, window_size).max(axis=1)

def run_automata_pipeline(dataset_name, X_train_pca, X_test_pca, y_train, y_test, cfg):
    """Olasılıksal Otomata boru hattı ve Açıklanabilirlik Raporu (JSON) üretimi."""
    a_cfg = cfg["automata_params"]
    w_size, a_size = a_cfg["window_size"], a_cfg["alphabet_size"]
    
    y_test_compressed = compress_labels((y_test == 1).astype(int), w_size)
    
    paa_train = automata.apply_paa(X_train_pca, w_size)
    paa_test = automata.apply_paa(X_test_pca, w_size)
    
    sax_train = automata.apply_sax(paa_train, a_size)
    sax_test = automata.apply_sax(paa_test, a_size)
    
    t_matrix = automata.build_transition_matrix(sax_train, a_size)
    train_scores = automata.get_sequence_scores(sax_train, t_matrix, a_size)
    threshold = np.percentile(train_scores, a_cfg["threshold_percentile"])
    
    test_scores = automata.get_sequence_scores(sax_test, t_matrix, a_size)
    y_pred = (test_scores > threshold).astype(int)
    
    # Unseen (Görülmemiş) Pattern Analizi
    train_patterns = set(automata.extract_patterns(sax_train, a_cfg["pattern_length"]))
    unseen_data = automata.evaluate_unseen_patterns(sax_train, sax_test, a_cfg["pattern_length"])
    unseen_dict = {p[0]: p[1] for p in unseen_data} # {pattern: mapped_to}
    
    # Rubrik Gereksinimi: JSON Açıklanabilirlik Raporu Üretimi
    json_reports = []
    for idx, (score, pred) in enumerate(zip(test_scores, y_pred)):
        if pred == 1:
            start_idx = max(0, idx - a_cfg["pattern_length"] + 1)
            pattern = "".join(sax_test[start_idx:idx+1])
            status = "unseen" if pattern not in train_patterns else "seen"
            mapped_to = unseen_dict.get(pattern, pattern)
            
            report = {
                "time_step": idx,
                "state": "".join(sax_test[max(0, start_idx-1):start_idx]),
                "pattern": pattern,
                "status": status,
                "mapped_to": mapped_to,
                "probability": float(np.exp(-score)), # Score negatif logaritmaydı, olasılığa (0-1) geri çeviriyoruz
                "decision": "anomaly",
                "confidence": automata.calculate_confidence(score, threshold)
            }
            json_reports.append(report)

    return {
        "F1": f1_score(y_test_compressed, y_pred, zero_division=0),
        "Precision": precision_score(y_test_compressed, y_pred, zero_division=0),
        "Recall": recall_score(y_test_compressed, y_pred, zero_division=0),
        "Reports": json_reports[:3] # Raporu şişirmemek için ilk 3 anomalinin analizini döndür
    }

def run_dl_pipeline(model_class, X_train, y_train, X_val, y_val, X_test, y_test, cfg):
    """Derin Öğrenme boru hattı (Seed ortalamaları ve Early Stopping entegreli)."""
    dl_cfg = cfg["dl_params"]
    seq_length, batch_size = dl_cfg["seq_length"], dl_cfg["batch_size"]
    
    X_train_dl, y_train_dl = deep_learning.create_sequences(X_train, y_train, seq_length)
    X_val_dl, y_val_dl = deep_learning.create_sequences(X_val, y_val, seq_length)
    X_test_dl, y_test_dl = deep_learning.create_sequences(X_test, y_test, seq_length)
    
    train_loader = DataLoader(TensorDataset(X_train_dl, y_train_dl), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_dl, y_val_dl), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_dl, y_test_dl), batch_size=batch_size, shuffle=False)
    
    f1_scores = []
    
    # Rubrik Gereksinimi: 5 farklı Seed ile deney tekrarı
    for seed in dl_cfg["random_seeds"]:
        deep_learning.set_seed(seed)
        
        if model_class == deep_learning.TimeSeriesCNN:
            model = model_class(input_features=1, seq_length=seq_length, hidden_size=dl_cfg["hidden_size"])
        else:
            model = model_class(input_features=1, hidden_size=dl_cfg["hidden_size"], num_layers=dl_cfg["num_layers"])
            
        model = deep_learning.train_model(model, train_loader, val_loader, cfg)
        f1, _, _ = deep_learning.evaluate_model(model, test_loader)
        f1_scores.append(f1)
        
    return {"F1_Avg": np.mean(f1_scores), "F1_Std": np.std(f1_scores)}

def run_parameter_grid(X_train_pca, X_test_pca, y_train, y_test, cfg):
    """Rubrik Gereksinimi: Window Size ve Alphabet Size grid analizi."""
    results = []
    for w in cfg["automata_params"]["window_size_grid"]:
        for a in cfg["automata_params"]["alphabet_size_grid"]:
            temp_cfg = copy.deepcopy(cfg)
            temp_cfg["automata_params"]["window_size"] = w
            temp_cfg["automata_params"]["alphabet_size"] = a
            res = run_automata_pipeline("GRID", X_train_pca, X_test_pca, y_train, y_test, temp_cfg)
            results.append({"w_size": w, "a_size": a, "F1": res["F1"]})
    return results

def run_noise_experiment(X_test_pca, cfg):
    """Rubrik Gereksinimi: Gaussian Noise senaryosu."""
    noise = np.random.normal(0, cfg["experiment_params"]["noise_std"], size=X_test_pca.shape)
    return X_test_pca + noise