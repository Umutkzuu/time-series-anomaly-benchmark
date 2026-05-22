import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import data_loader
import preprocessing
import automata

def compress_labels(y, window_size):
    """Orijinal etiketleri PAA pencere boyutuna uyarlar (Max Pooling)."""
    # BATADAL'daki -999 gibi anlamsız etiketleri yoksay, sadece 1'leri tut!
    y_clean = (y == 1).astype(int)
    
    n = len(y_clean)
    remainder = n % window_size
    if remainder != 0:
        y_clean = y_clean[:-remainder]
    y_reshaped = y_clean.reshape(-1, window_size)
    
    # Pencere içinde bir tane bile 1 (Anomali) varsa, pencere 1 olur
    return y_reshaped.max(axis=1) 

def run_pipeline():
    print("1. Veriler Çekiliyor ve Temizleniyor...")
    cfg = data_loader.load_config()
    batadal_df = data_loader.load_batadal_data(cfg)
    res_batadal = preprocessing.prep_batadal(batadal_df, cfg)
    
    X_train_pca = res_batadal["pca"][0]
    X_test_pca = res_batadal["pca"][2]
    
    # Orijinal etiketleri (Saldırı bayraklarını) alıyoruz
    y_test = res_batadal["y"][2]
    
    # --- YENİ PARAMETRELER (Hassasiyeti Artırıyoruz) ---
    w_size = 3   # 5 saat yerine 3 saatlik pencereler (Detay kaybı azalır)
    a_size = 8   # Alfabe boyutu büyütüldü (Ufak sensör değişimleri farklı harf olur)
    
    # Orijinal etiketleri PAA ile aynı boyuta sıkıştır
    y_test_compressed = compress_labels(y_test, w_size)
    
    print("2. PAA, SAX ve Markov Zinciri Çalıştırılıyor...")
    paa_train = automata.apply_paa(X_train_pca, window_size=w_size)
    paa_test = automata.apply_paa(X_test_pca, window_size=w_size)
    
    sax_train = automata.apply_sax(paa_train, alphabet_size=a_size)
    sax_test = automata.apply_sax(paa_test, alphabet_size=a_size)
    
    t_matrix = automata.build_transition_matrix(sax_train, alphabet_size=a_size)
    
    # --- EŞİK (THRESHOLD) BELİRLEME ---
    print("3. Eşik Değeri (Threshold) Hesaplanıyor...")
    train_scores = automata.get_sequence_scores(sax_train, t_matrix, a_size)
    
    # EŞİK DÜŞÜRÜLDÜ: %99 yerine %95. (Daha kolay alarm çalacak)
    threshold = np.percentile(train_scores, 95)
    
    test_scores = automata.get_sequence_scores(sax_test, t_matrix, a_size)
    y_pred = (test_scores > threshold).astype(int)
    
    print("\n" + "="*45)
    print("    BATADAL OTOMATA DEĞERLENDİRME SONUÇLARI")
    print("="*45)
    print(f"Sistem Eşik Değeri (Threshold) : {threshold:.4f}")
    print(f"Gerçek Anomali Sayısı          : {sum(y_test_compressed)}")
    print(f"Modelin Bulduğu Anomali Sayısı : {sum(y_pred)}")
    print("-" * 45)
    
    print("F1-Score  : {:.4f}".format(f1_score(y_test_compressed, y_pred, zero_division=0)))
    print("Precision : {:.4f}".format(precision_score(y_test_compressed, y_pred, zero_division=0)))
    print("Recall    : {:.4f}".format(recall_score(y_test_compressed, y_pred, zero_division=0)))
    print("="*45)

if __name__ == "__main__":
    run_pipeline()