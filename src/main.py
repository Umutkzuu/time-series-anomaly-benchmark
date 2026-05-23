import json
import data_loader
import preprocessing
import pipeline
import deep_learning
import warnings

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
    
    # --- TEŞHİS: Sınıf Dengesizliği Analizi ---
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
    print(" PROJE BORU HATTI BAŞARIYLA TAMAMLANDI! ")
    print("="*60)

if __name__ == "__main__":
    main()