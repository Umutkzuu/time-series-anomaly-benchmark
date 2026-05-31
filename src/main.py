import json
import data_loader
import preprocessing
import pipeline
import deep_learning
import warnings
import visualize

warnings.filterwarnings('ignore')

def main():
    print("1. Konfigürasyon ve Veriler Yükleniyor...\n")
    cfg = data_loader.load_config()
        
    batadal_df = data_loader.load_batadal_data(cfg)
    res = preprocessing.prep_batadal(batadal_df, cfg)
    
    X_train_pca = res["pca"][0]
    X_test_pca = res["pca"][2]
    
    X_train_sc = res["scaled"][0]
    X_val_sc   = res["scaled"][1]
    X_test_sc  = res["scaled"][2]
    
    y_train = res["y"][0]
    y_val = res["y"][1]
    y_test = res["y"][2]
    
    print("\n--- VERİ SETİ ANALİZİ (CLASS IMBALANCE) ---")
    anomali_orani = (y_train == 1).sum() / len(y_train)
    print(f"Train anomali oranı: {anomali_orani:.4f}")
    print(f"Normal/Anomali oranı: {(1-anomali_orani)/(anomali_orani+1e-9):.1f}:1")
    print(f"Train anomali sayısı: {(y_train == 1).sum()} / {len(y_train)}\n")
    
    print("="*60)
    print(" BÖLÜM 1: OLASILIKSAL OTOMATA (EXPLAINABILITY) ")
    print("="*60)
    auto_res = pipeline.run_automata_pipeline("BATADAL", X_train_pca, X_test_pca, y_train, y_test, cfg)
    print(f"Otomata F1-Score: {auto_res['F1']:.4f}")
    
    print("\n--- JSON Açıklanabilirlik Raporları (İlk 3 Anomali) ---")
    for report in auto_res['Reports']:
        print(json.dumps(report, indent=4, ensure_ascii=False))
        
    print("\n" + "="*60)
    print(" BÖLÜM 2: DERİN ÖĞRENME (5 SEED ORTALAMASI - ÇOK BOYUTLU VERİ) ")
    print("="*60)
    y_train_dl = (y_train == 1).astype(int)
    y_val_dl = (y_val == 1).astype(int)
    y_test_dl = (y_test == 1).astype(int)
    
    print("[ 1D-CNN EĞİTİMİ BAŞLIYOR (Early Stopping & Seed Tekrarı) ]")
    cnn_res = pipeline.run_dl_pipeline(deep_learning.TimeSeriesCNN, X_train_sc, y_train_dl, X_val_sc, y_val_dl, X_test_sc, y_test_dl, cfg)
    print(f"> 1D-CNN Ortalama F1: {cnn_res['F1_Avg']:.4f} (±{cnn_res['F1_Std']:.4f})\n")
    
    print("[ GRU EĞİTİMİ BAŞLIYOR (Early Stopping & Seed Tekrarı) ]")
    gru_res = pipeline.run_dl_pipeline(deep_learning.TimeSeriesGRU, X_train_sc, y_train_dl, X_val_sc, y_val_dl, X_test_sc, y_test_dl, cfg)
    print(f"> GRU Ortalama F1   : {gru_res['F1_Avg']:.4f} (±{gru_res['F1_Std']:.4f})")

    print("\n" + "="*60)
    print(" BÖLÜM 3: DENEYLER VE ANALİZLER (RUBRİK ZORUNLULUKLARI) ")
    print("="*60)
    print("1. Parametre Grid Search Analizi (Window & Alphabet Size)...")
    grid_results = pipeline.run_parameter_grid(X_train_pca, X_test_pca, y_train, y_test, cfg)
    best_grid = max(grid_results, key=lambda x: x["F1"])
    print(f"-> En İyi Parametreler: Window = {best_grid['w_size']}, Alphabet = {best_grid['a_size']} | F1: {best_grid['F1']:.4f}")
    
    print("\n2. Gürültülü (Gaussian Noise) Veri Senaryosu...")
    noisy_res = pipeline.run_noise_experiment(X_train_pca, X_test_pca, y_train, y_test, cfg)
    print(f"-> Gürültülü Ortamda Otomata F1-Score: {noisy_res['F1']:.4f}")

    print("\n" + "="*60)
    print(" BÖLÜM 4: SKAB VERİ SETİ - GroupKFold DEĞERLENDİRMESİ ")
    print("="*60)
    skab_df = data_loader.load_skab_data(cfg)
    skab_res = preprocessing.prep_skab(skab_df, cfg)

    anomali_skab = (skab_res["y"][0] == 1).sum() / len(skab_res["y"][0])
    print(f"SKAB Train anomali oranı: {anomali_skab:.4f}")
    print(f"SKAB Normal/Anomali oranı: {(1 - anomali_skab) / (anomali_skab + 1e-9):.1f}:1\n")

    print("[ SKAB GroupKFold Pipeline Başlıyor... ]")
    skab_pipeline_res = pipeline.run_skab_pipeline(skab_res, cfg)

    n = skab_pipeline_res["n_folds"]
    auto = skab_pipeline_res["automata"]
    cnn  = skab_pipeline_res["cnn"]
    gru  = skab_pipeline_res["gru"]

    print(f"\n{'Model':<12} {'Fold Ort. F1':>14} {'Std':>8}  {'Fold Sonuçları'}")
    print("-" * 60)
    fold_str = lambda lst: "  ".join([f"{v:.4f}" for v in lst])
    print(f"{'Automata':<12} {auto['F1_Avg']:>14.4f} {auto['F1_Std']:>8.4f}  {fold_str(auto['folds'])}")
    print(f"{'1D-CNN':<12}  {cnn['F1_Avg']:>13.4f} {cnn['F1_Std']:>8.4f}  {fold_str(cnn['folds'])}")
    print(f"{'GRU':<12}  {gru['F1_Avg']:>13.4f} {gru['F1_Std']:>8.4f}  {fold_str(gru['folds'])}")

    print("\n" + "="*60)
    print(" BÖLÜM 5: İSTATİSTİKSEL ANALİZ (Wilcoxon Signed-Rank) ")
    print("="*60)
    from scipy.stats import wilcoxon

    cnn_seeds = cnn_res["F1_Seeds"]
    gru_seeds = gru_res["F1_Seeds"]

    print(f"{'Model':<12} {'Seed F1 Listesi':<45} {'Ort':>6}")
    print("-" * 70)
    print(f"{'1D-CNN':<12} {str([round(v,4) for v in cnn_seeds]):<45} {cnn_res['F1_Avg']:.4f}")
    print(f"{'GRU':<12} {str([round(v,4) for v in gru_seeds]):<45} {gru_res['F1_Avg']:.4f}")

    print("\n[BATADAL] CNN vs GRU — Wilcoxon Signed-Rank Testi:")
    try:
        if len(set(x - y for x, y in zip(cnn_seeds, gru_seeds))) == 1:
            print("  -> Tüm farklar aynı, Wilcoxon uygulanamaz (sıfır varyans).")
        else:
            stat, p = wilcoxon(cnn_seeds, gru_seeds)
            print(f"  Statistic = {stat:.4f}, p-value = {p:.4f}")
            if p < 0.05:
                better = "1D-CNN" if cnn_res["F1_Avg"] > gru_res["F1_Avg"] else "GRU"
                print(f"  -> p < 0.05: Fark istatistiksel olarak ANLAMLI. {better} daha iyi.")
            else:
                print("  -> p >= 0.05: Fark istatistiksel olarak ANLAMSIZ (modeller benzer).")
    except Exception as e:
        print(f"  -> Wilcoxon uygulanamadı: {e}")

    skab_cnn_folds  = skab_pipeline_res["cnn"]["folds"]
    skab_gru_folds  = skab_pipeline_res["gru"]["folds"]
    skab_auto_folds = skab_pipeline_res["automata"]["folds"]

    print("\n[SKAB] CNN vs GRU — Wilcoxon Signed-Rank Testi (fold bazlı):")
    try:
        if len(skab_cnn_folds) < 2 or len(set(x - y for x, y in zip(skab_cnn_folds, skab_gru_folds))) == 1:
            print("  -> Yetersiz fold veya sıfır varyans, test uygulanamadı.")
        else:
            stat, p = wilcoxon(skab_cnn_folds, skab_gru_folds)
            print(f"  Statistic = {stat:.4f}, p-value = {p:.4f}")
            if p < 0.05:
                better = "1D-CNN" if skab_pipeline_res["cnn"]["F1_Avg"] > skab_pipeline_res["gru"]["F1_Avg"] else "GRU"
                print(f"  -> p < 0.05: Fark istatistiksel olarak ANLAMLI. {better} daha iyi.")
            else:
                print("  -> p >= 0.05: Fark istatistiksel olarak ANLAMSIZ.")
    except Exception as e:
        print(f"  -> Wilcoxon uygulanamadı: {e}")

    print("\n[SKAB] DL (CNN ort.) vs Automata — Wilcoxon Signed-Rank Testi (fold bazlı):")
    try:
        if len(skab_auto_folds) < 2 or len(set(x - y for x, y in zip(skab_cnn_folds, skab_auto_folds))) == 1:
            print("  -> Yetersiz fold veya sıfır varyans, test uygulanamadı.")
        else:
            stat, p = wilcoxon(skab_cnn_folds, skab_auto_folds)
            print(f"  Statistic = {stat:.4f}, p-value = {p:.4f}")
            if p < 0.05:
                print("  -> p < 0.05: DL ve Automata arasındaki fark istatistiksel olarak ANLAMLI.")
            else:
                print("  -> p >= 0.05: Fark istatistiksel olarak ANLAMSIZ.")
    except Exception as e:
        print(f"  -> Wilcoxon uygulanamadı: {e}")

    print("\n" + "="*60)
    print(" BÖLÜM 6: GÖRSELLEŞTİRMELER ")
    print("="*60)

    import automata as automata_mod
    import copy

    a_cfg = cfg["automata_params"]
    w_size, a_size = a_cfg["window_size"], a_cfg["alphabet_size"]

    print("Confusion Matrix üretiliyor (BATADAL - Automata)...")
    visualize.plot_confusion_matrix(
        auto_res["y_true"], auto_res["y_pred"],
        dataset_name="BATADAL", model_name="Automata"
    )

    print("ROC & Precision-Recall eğrisi üretiliyor (BATADAL - Automata)...")
    visualize.plot_roc_pr(
        auto_res["y_true"], auto_res["y_scores"],
        dataset_name="BATADAL", model_name="Automata"
    )

    print("Transition heatmap üretiliyor (BATADAL)...")
    visualize.plot_transition_heatmap(
        auto_res["transition_matrix"], auto_res["alphabet_size"],
        dataset_name="BATADAL"
    )

    print("State diagram üretiliyor (BATADAL)...")
    visualize.plot_state_diagram(
        auto_res["transition_matrix"], auto_res["alphabet_size"],
        dataset_name="BATADAL"
    )

    print("Parametre duyarlılık grafikleri üretiliyor (BATADAL)...")
    visualize.plot_parameter_sensitivity(grid_results, dataset_name="BATADAL")

    print("Model karşılaştırma grafikleri üretiliyor...")
    batadal_comparison = {
        "Automata": {"F1": auto_res["F1"], "Std": 0.0},
        "1D-CNN":   {"F1": cnn_res["F1_Avg"], "Std": cnn_res["F1_Std"]},
        "GRU":      {"F1": gru_res["F1_Avg"], "Std": gru_res["F1_Std"]},
    }
    visualize.plot_model_comparison(batadal_comparison, dataset_name="BATADAL")

    skab_comparison = {
        "Automata": {"F1": skab_pipeline_res["automata"]["F1_Avg"], "Std": skab_pipeline_res["automata"]["F1_Std"]},
        "1D-CNN":   {"F1": skab_pipeline_res["cnn"]["F1_Avg"],      "Std": skab_pipeline_res["cnn"]["F1_Std"]},
        "GRU":      {"F1": skab_pipeline_res["gru"]["F1_Avg"],      "Std": skab_pipeline_res["gru"]["F1_Std"]},
    }
    visualize.plot_model_comparison(skab_comparison, dataset_name="SKAB")

    print("\nTüm görseller outputs/figures/ klasörüne kaydedildi.")

    print("\n" + "="*60)
    print(" PROJE BORU HATTI BAŞARIYLA TAMAMLANDI! ")
    print("="*60)

if __name__ == "__main__":
    main()