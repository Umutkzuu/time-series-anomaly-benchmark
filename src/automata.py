import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def apply_paa(ts, window_size):
    ts = ts.flatten()
    remainder = len(ts) % window_size
    if remainder != 0: ts = ts[:-remainder]
    return ts.reshape(-1, window_size).mean(axis=1)

def apply_sax(paa_ts, alphabet_size):
    alphabet = [chr(97 + i) for i in range(alphabet_size)]
    breakpoints = norm.ppf(np.linspace(1./alphabet_size, 1 - 1./alphabet_size, alphabet_size - 1))
    return [alphabet[np.searchsorted(breakpoints, val)] for val in paa_ts]

def build_transition_matrix(sax_sequence, alphabet_size):
    epsilon = 1e-6
    matrix = np.full((alphabet_size, alphabet_size), epsilon)
    char_to_idx = {chr(97 + i): i for i in range(alphabet_size)}
    for i in range(len(sax_sequence) - 1):
        matrix[char_to_idx[sax_sequence[i]], char_to_idx[sax_sequence[i+1]]] += 1
    return matrix / matrix.sum(axis=1, keepdims=True)

def get_sequence_scores(sax_sequence, transition_matrix, alphabet_size):
    char_to_idx = {chr(97 + i): i for i in range(alphabet_size)}
    scores = [0.0]
    for i in range(len(sax_sequence) - 1):
        prob = transition_matrix[char_to_idx[sax_sequence[i]], char_to_idx[sax_sequence[i+1]]]
        scores.append(-np.log(prob))
    return np.array(scores)

# --- RUBRİK GÜNCELLEMESİ: LEVENSHTEIN (UNSEEN DATA) YÖNETİMİ ---

def levenshtein_distance(s1, s2):
    """İki SAX kelimesi (pattern) arasındaki harf değişim mesafesini hesaplar."""
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def extract_patterns(sax_seq, length):
    """SAX dizisinden n-uzunluğunda kelimeler (pattern) çıkarır."""
    return ["".join(sax_seq[i:i+length]) for i in range(len(sax_seq) - length + 1)]

def evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3):
    """
    Test verisindeki kelimeleri eğitim verisiyle kıyaslar. Unseen (görülmemiş) 
    kelimelerin en yakın bilinen kelimeye olan Levenshtein mesafesini döndürür.
    """
    train_patterns = set(extract_patterns(train_seq, pattern_length))
    test_patterns = extract_patterns(test_seq, pattern_length)
    
    unseen_distances = []
    for pattern in test_patterns:
        if pattern not in train_patterns:
            # En yakın bilinen kelimeye olan uzaklığı bul
            min_dist = min(levenshtein_distance(pattern, tp) for tp in train_patterns)
            unseen_distances.append((pattern, min_dist))
    return unseen_distances

def calculate_confidence(score, threshold):
    """
    Anomali skorunun eşik değerine (threshold) olan uzaklığına bakarak
    %50 ile %99.9 arasında bir güven skoru (confidence) üretir.
    """
    if score >= threshold:
        confidence = 50 + ((score - threshold) / (threshold + 1e-6)) * 50
    else:
        confidence = 50 + ((threshold - score) / (threshold + 1e-6)) * 50
        
    return min(99.99, confidence) 


if __name__ == "__main__":
    print("--- LEVENSHTEIN BİRİM TESTİ (UNIT TEST) ---")
    
    # Eğitim setinde sistem sadece "aba", "bab", "abc" kelimelerini gördü.
    train_dummy = ['a', 'b', 'a', 'b', 'c'] 
    
    # Test sırasında siber saldırı oldu ve "adc" diye tamamen yabancı (Unseen) bir kelime geldi.
    test_dummy = ['a', 'd', 'c'] 
    
    print(f"Eğitim Setindeki Kelimeler: {set(extract_patterns(train_dummy, 3))}")
    print(f"Test Setine Gelen Şüpheli Kelime: {''.join(test_dummy)}")
    
    distances = evaluate_unseen_patterns(train_dummy, test_dummy, pattern_length=3)
    
    for pattern, dist in distances:
        print(f"Unseen Pattern Tespit Edildi: '{pattern}' | Bilinen en yakın kelimeye Levenshtein Uzaklığı: {dist}")
        if dist > 1:
            print("=> SONUÇ: Yüksek Risk (Büyük Morfolojik Sıçrama, Sisteme Saldırı Olabilir!)")