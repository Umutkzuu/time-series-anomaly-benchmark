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
    train_patterns = set(automata.extract_patterns(sax_train, a_cfg["pattern_length"]))
    unseen_data = automata.evaluate_unseen_patterns(sax_train, sax_test, a_cfg["pattern_length"])
    unseen_dict = {p[0]: (p[1], p[2]) for p in unseen_data}
    json_reports = []
    for idx, (score, pred) in enumerate(zip(test_scores, y_pred)):
        if pred == 1:
            start_idx = max(0, idx - a_cfg["pattern_length"] + 1)
            sax_window = sax_test[start_idx:idx+1]
            pattern = "".join(sax_window)
            status = "unseen" if pattern not in train_patterns else "seen"
            mapped_to, lev_dist = unseen_dict.get(pattern, (pattern, 0))
            transitions, path_prob = automata.get_transitions_and_path_prob(
                sax_window, t_matrix, a_size
            )
            report = {
                "time_step": idx,
                "state": "".join(sax_test[max(0, start_idx-1):start_idx]),
                "pattern": pattern,
                "status": status,
                "mapped_to": mapped_to,
                "levenshtein_distance": lev_dist,
                "transitions": transitions,
                "path_probability": path_prob,
                "decision": "anomaly",
                "confidence": automata.calculate_confidence(score, threshold)
            }
            json_reports.append(report)
    return {
        "F1": f1_score(y_test_compressed, y_pred, zero_division=0),
        "Precision": precision_score(y_test_compressed, y_pred, zero_division=0),
        "Recall": recall_score(y_test_compressed, y_pred, zero_division=0),
        "Reports": json_reports[:3],
        "y_true": y_test_compressed,
        "y_pred": y_pred,
        "y_scores": test_scores[:len(y_test_compressed)],
        "transition_matrix": t_matrix,
        "alphabet_size": a_size,
    }

def run_dl_pipeline(model_class, X_train, y_train, X_val, y_val, X_test, y_test, cfg):
    dl_cfg = cfg["dl_params"]
    seq_length, batch_size = dl_cfg["seq_length"], dl_cfg["batch_size"]
    input_features = X_train.shape[1]
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight_val = float(neg_count / (pos_count + 1e-9))
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["dl_params"]["pos_weight"] = pos_weight_val
    X_train_dl, y_train_dl = deep_learning.create_sequences(X_train, y_train, seq_length)
    X_val_dl, y_val_dl = deep_learning.create_sequences(X_val, y_val, seq_length)
    X_test_dl, y_test_dl = deep_learning.create_sequences(X_test, y_test, seq_length)
    train_loader = DataLoader(TensorDataset(X_train_dl, y_train_dl), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_dl, y_val_dl), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_dl, y_test_dl), batch_size=batch_size, shuffle=False)
    f1_scores = []
    for seed in dl_cfg["random_seeds"]:
        deep_learning.set_seed(seed)
        if model_class == deep_learning.TimeSeriesCNN:
            model = model_class(input_features=input_features, seq_length=seq_length, hidden_size=dl_cfg["hidden_size"])
        else:
            model = model_class(input_features=input_features, hidden_size=dl_cfg["hidden_size"], num_layers=dl_cfg["num_layers"])
        model = deep_learning.train_model(model, train_loader, val_loader, cfg_copy)
        f1, _, _ = deep_learning.evaluate_model(model, test_loader)
        f1_scores.append(f1)
    return {"F1_Avg": np.mean(f1_scores), "F1_Std": np.std(f1_scores), "F1_Seeds": f1_scores}

def run_parameter_grid(X_train_pca, X_test_pca, y_train, y_test, cfg):
    results = []
    for w in cfg["automata_params"]["window_size_grid"]:
        for a in cfg["automata_params"]["alphabet_size_grid"]:
            temp_cfg = copy.deepcopy(cfg)
            temp_cfg["automata_params"]["window_size"] = w
            temp_cfg["automata_params"]["alphabet_size"] = a
            res = run_automata_pipeline("GRID", X_train_pca, X_test_pca, y_train, y_test, temp_cfg)
            results.append({"w_size": w, "a_size": a, "F1": res["F1"]})
    return results

def run_noise_experiment(X_train_pca, X_test_pca, y_train, y_test, cfg):
    noise = np.random.normal(0, cfg["experiment_params"]["noise_std"], size=X_test_pca.shape)
    X_test_noisy = X_test_pca + noise
    return run_automata_pipeline("NOISE", X_train_pca, X_test_noisy, y_train, y_test, cfg)


def run_skab_pipeline(skab_res, cfg):
    """
    SKAB veri seti için GroupKFold tabanlı pipeline.
    Her fold'da hem otomata hem de DL modelleri çalıştırılır.
    Sonuçlar fold ortalaması ve standart sapması olarak raporlanır.
    """
    from sklearn.model_selection import GroupKFold
    import preprocessing

    X_scaled  = np.vstack([skab_res["scaled"][0], skab_res["scaled"][1], skab_res["scaled"][2]])
    X_pca     = np.vstack([skab_res["pca"][0],    skab_res["pca"][1],    skab_res["pca"][2]])
    y_all     = np.concatenate([skab_res["y"][0], skab_res["y"][1],      skab_res["y"][2]])
    groups    = skab_res["groups"]

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = {
        "automata": [],
        "cnn":      [],
        "gru":      [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y_all, groups)):
        print(f"  [Fold {fold_idx + 1}/{n_splits}]", end=" ", flush=True)

        # --- Veri bölme ---
        X_tr_sc, X_te_sc = X_scaled[train_idx], X_scaled[test_idx]
        X_tr_pca, X_te_pca = X_pca[train_idx],  X_pca[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Validation: test'in ilk %20'si
        val_cut = max(1, int(len(X_te_sc) * 0.2))
        X_val_sc, X_test_sc = X_te_sc[:val_cut], X_te_sc[val_cut:]
        y_val,    y_test_f  = y_te[:val_cut],    y_te[val_cut:]

        if len(y_test_f) == 0:
            print("atlandı (test boş)")
            continue

        # --- Otomata ---
        auto_res = run_automata_pipeline("SKAB_FOLD", X_tr_pca, X_te_pca, y_tr, y_te, cfg)
        fold_results["automata"].append(auto_res["F1"])
        print(f"Automata F1={auto_res['F1']:.4f}", end=" | ", flush=True)

        # --- DL modelleri ---
        y_tr_dl  = (y_tr     == 1).astype(int)
        y_val_dl = (y_val    == 1).astype(int)
        y_te_dl  = (y_test_f == 1).astype(int)

        cnn_res = run_dl_pipeline(
            deep_learning.TimeSeriesCNN,
            X_tr_sc, y_tr_dl, X_val_sc, y_val_dl, X_test_sc, y_te_dl, cfg
        )
        fold_results["cnn"].append(cnn_res["F1_Avg"])

        gru_res = run_dl_pipeline(
            deep_learning.TimeSeriesGRU,
            X_tr_sc, y_tr_dl, X_val_sc, y_val_dl, X_test_sc, y_te_dl, cfg
        )
        fold_results["gru"].append(gru_res["F1_Avg"])
        print(f"CNN F1={cnn_res['F1_Avg']:.4f} | GRU F1={gru_res['F1_Avg']:.4f}")

    def _stats(lst):
        arr = np.array(lst) if lst else np.array([0.0])
        return float(np.mean(arr)), float(np.std(arr)), lst

    auto_avg, auto_std, auto_folds = _stats(fold_results["automata"])
    cnn_avg,  cnn_std,  cnn_folds  = _stats(fold_results["cnn"])
    gru_avg,  gru_std,  gru_folds  = _stats(fold_results["gru"])

    return {
        "n_folds": n_splits,
        "automata": {"F1_Avg": auto_avg, "F1_Std": auto_std, "folds": auto_folds},
        "cnn":      {"F1_Avg": cnn_avg,  "F1_Std": cnn_std,  "folds": cnn_folds},
        "gru":      {"F1_Avg": gru_avg,  "F1_Std": gru_std,  "folds": gru_folds},
    }