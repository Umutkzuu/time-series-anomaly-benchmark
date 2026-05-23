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

def levenshtein_distance(s1, s2):
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
    return ["".join(sax_seq[i:i+length]) for i in range(len(sax_seq) - length + 1)]

def evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3):
    train_patterns = list(set(extract_patterns(train_seq, pattern_length)))
    test_patterns = extract_patterns(test_seq, pattern_length)
    
    if not train_patterns: return []
    
    unseen_distances = []
    for pattern in test_patterns:
        if pattern not in train_patterns:
            distances = [(tp, levenshtein_distance(pattern, tp)) for tp in train_patterns]
            nearest_pattern, min_dist = min(distances, key=lambda x: x[1])
            unseen_distances.append((pattern, nearest_pattern, min_dist))
    return unseen_distances

def calculate_confidence(score, threshold):
    if score >= threshold:
        confidence = 50 + ((score - threshold) / (threshold + 1e-6)) * 50
    else:
        confidence = 50 + ((threshold - score) / (threshold + 1e-6)) * 50
    return min(99.99, confidence)