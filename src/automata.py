import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def apply_paa(ts, window_size):
    """Sürekli zaman serisini parçalara bölerek ortalamasını alır ve sıkıştırır."""
    ts = ts.flatten()
    n = len(ts)
    remainder = n % window_size
    if remainder != 0:
        ts = ts[:-remainder]
    ts_reshaped = ts.reshape(-1, window_size)
    return ts_reshaped.mean(axis=1)

def apply_sax(paa_ts, alphabet_size):
    """Sıkıştırılmış değerleri Normal Dağılıma göre harflere çevirir."""
    alphabet = [chr(97 + i) for i in range(alphabet_size)]
    breakpoints = norm.ppf(np.linspace(1./alphabet_size, 1 - 1./alphabet_size, alphabet_size - 1))
    
    sax_symbols = []
    for val in paa_ts:
        idx = np.searchsorted(breakpoints, val)
        sax_symbols.append(alphabet[idx])
    return sax_symbols

# --- YENİ EKLENEN: OLASILIKSAL OTOMATA (MARKOV ZİNCİRİ) ---

def build_transition_matrix(sax_sequence, alphabet_size):
    """
    Eğitim verisindeki harf dizilimini (örn: a->b) sayarak Geçiş Matrisi oluşturur.
    """
    # Sıfıra bölme hatasını ve "sonsuz anomali" bug'ını önlemek için matrisi 
    # 0 yerine çok küçük bir sayı (epsilon) ile başlatıyoruz (Laplace Smoothing)
    epsilon = 1e-6
    matrix = np.full((alphabet_size, alphabet_size), epsilon)
    
    char_to_idx = {chr(97 + i): i for i in range(alphabet_size)}
    
    # Dizideki ardışık harf çiftlerini say
    for i in range(len(sax_sequence) - 1):
        current_state = char_to_idx[sax_sequence[i]]
        next_state = char_to_idx[sax_sequence[i+1]]
        matrix[current_state, next_state] += 1
        
    # Satırları toplayıp olasılıklara (0 ile 1 arasına) çevir
    row_sums = matrix.sum(axis=1, keepdims=True)
    transition_probs = matrix / row_sums
    
    return transition_probs

def calculate_anomaly_score(sax_sequence, transition_matrix, alphabet_size):
    """
    Test verisindeki harf geçişlerinin olasılığına bakar.
    Matriste hiç görülmemiş/nadir bir geçiş yapılmışsa anomali skoru yükselir.
    """
    char_to_idx = {chr(97 + i): i for i in range(alphabet_size)}
    scores = []
    
    for i in range(len(sax_sequence) - 1):
        current_state = char_to_idx[sax_sequence[i]]
        next_state = char_to_idx[sax_sequence[i+1]]
        
        # Bu geçişin olasılığı nedir?
        prob = transition_matrix[current_state, next_state]
        
        # Olasılık ne kadar düşükse, anomali skoru o kadar yüksek olur (-log(P))
        score = -np.log(prob)
        scores.append(score)
        
    # Toplam dizinin ortalama anomali skorunu döndür
    return np.mean(scores) if scores else 0.0

def get_sequence_scores(sax_sequence, transition_matrix, alphabet_size):
    """
    Tüm dizinin ortalamasını almak yerine, her bir harf geçişi için
    ayrı ayrı anomali skoru üretip bir dizi (array) olarak döndürür.
    """
    char_to_idx = {chr(97 + i): i for i in range(alphabet_size)}
    # İlk harfe giden bir geçiş olmadığı için onun skorunu 0 kabul ediyoruz
    scores = [0.0] 
    
    for i in range(len(sax_sequence) - 1):
        current_state = char_to_idx[sax_sequence[i]]
        next_state = char_to_idx[sax_sequence[i+1]]
        
        prob = transition_matrix[current_state, next_state]
        scores.append(-np.log(prob))
        
    return np.array(scores)


if __name__ == "__main__":
    print("--- OLASILIKSAL OTOMATA TESTİ ---\n")
    
    a_size = 3  # Alfabemiz: a, b, c
    
    # 1. Eğitim Aşaması: Sistem hep a->b, b->c, c->a akışını görüyor
    train_sax = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
    print(f"Eğitim Dizisi: {train_sax}")
    
    t_matrix = build_transition_matrix(train_sax, alphabet_size=a_size)
    print("\nÖğrenilen Geçiş Matrisi (Satırlar: a,b,c -> Sütunlar: a,b,c):")
    print(np.round(t_matrix, 2))
    
    # 2. Test Aşaması (Normal Akış)
    test_normal = ['a', 'b', 'c', 'a', 'b']
    score_normal = calculate_anomaly_score(test_normal, t_matrix, a_size)
    print(f"\nNormal Test Dizisi Skoru : {score_normal:.4f} (Beklenen: Çok Düşük)")
    
    # 3. Test Aşaması (Anormal Akış - Siber Saldırı)
    # Birdenbire kaotik geçişler oluyor: a->c, c->c, c->b
    test_anomaly = ['a', 'c', 'c', 'b', 'a']
    score_anomaly = calculate_anomaly_score(test_anomaly, t_matrix, a_size)
    print(f"Anormal Test Dizisi Skoru: {score_anomaly:.4f} (Beklenen: ÇOK YÜKSEK!)")