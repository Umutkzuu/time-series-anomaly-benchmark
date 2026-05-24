"""
Birim Testler: automata.py modülü
Kapsam: levenshtein_distance, evaluate_unseen_patterns,
        build_transition_matrix, get_transitions_and_path_prob
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import automata


class TestLevenshteinDistance(unittest.TestCase):

    def test_identical_strings(self):
        """Aynı iki string arasındaki mesafe 0 olmalı."""
        self.assertEqual(automata.levenshtein_distance("abc", "abc"), 0)

    def test_empty_vs_nonempty(self):
        """Boş string ile dolu string arasındaki mesafe, dolu stringin uzunluğu kadar olmalı."""
        self.assertEqual(automata.levenshtein_distance("", "abc"), 3)
        self.assertEqual(automata.levenshtein_distance("abc", ""), 3)

    def test_both_empty(self):
        """İki boş string arasındaki mesafe 0 olmalı."""
        self.assertEqual(automata.levenshtein_distance("", ""), 0)

    def test_single_substitution(self):
        """Tek karakter değişimi mesafesi 1 olmalı."""
        self.assertEqual(automata.levenshtein_distance("abc", "aXc"), 1)

    def test_single_insertion(self):
        """Tek karakter ekleme mesafesi 1 olmalı."""
        self.assertEqual(automata.levenshtein_distance("ab", "abc"), 1)

    def test_single_deletion(self):
        """Tek karakter silme mesafesi 1 olmalı."""
        self.assertEqual(automata.levenshtein_distance("abc", "ab"), 1)

    def test_completely_different(self):
        """Tamamen farklı iki string: mesafe = max uzunluk."""
        self.assertEqual(automata.levenshtein_distance("abc", "xyz"), 3)

    def test_symmetry(self):
        """Levenshtein mesafesi simetriktir: d(a,b) == d(b,a)."""
        self.assertEqual(
            automata.levenshtein_distance("aab", "bba"),
            automata.levenshtein_distance("bba", "aab")
        )

    def test_sax_patterns(self):
        """SAX pattern'larına özgü gerçekçi test durumları."""
        # 1 karakter farkı
        self.assertEqual(automata.levenshtein_distance("dae", "gae"), 1)
        self.assertEqual(automata.levenshtein_distance("aed", "acd"), 1)
        # 2 karakter farkı
        self.assertEqual(automata.levenshtein_distance("abc", "xyz"), 3)
        self.assertEqual(automata.levenshtein_distance("aab", "bbb"), 2)


class TestUnseenPatterns(unittest.TestCase):

    def test_all_patterns_seen(self):
        """Test pattern'ları eğitimde görüldüyse unseen listesi boş olmalı."""
        train_seq = list("abcabc")
        test_seq  = list("abcabc")
        result = automata.evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3)
        self.assertEqual(result, [])

    def test_all_patterns_unseen(self):
        """Hiç örtüşmeyen test pattern'ları tamamen unseen sayılmalı."""
        train_seq = list("aaabbb")
        test_seq  = list("cccddd")
        result = automata.evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3)
        self.assertGreater(len(result), 0)

    def test_unseen_returns_nearest(self):
        """Her unseen pattern için en yakın eğitim pattern'ı dönmeli."""
        train_seq = list("abcabc")
        test_seq  = list("abcabd")   # 'abd' unseen
        result = automata.evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3)
        patterns_found = [r[0] for r in result]
        self.assertIn("abd", patterns_found)

    def test_unseen_levenshtein_distance_positive(self):
        """Unseen pattern'ların Levenshtein mesafesi >= 1 olmalı."""
        train_seq = list("abcabc")
        test_seq  = list("xyzxyz")
        result = automata.evaluate_unseen_patterns(train_seq, test_seq, pattern_length=3)
        for _, _, dist in result:
            self.assertGreaterEqual(dist, 1)

    def test_empty_train(self):
        """Eğitim sequence boşsa sonuç boş liste olmalı."""
        result = automata.evaluate_unseen_patterns([], list("abc"), pattern_length=3)
        self.assertEqual(result, [])

    def test_short_sequence(self):
        """Pattern uzunluğundan kısa sequence'da pattern çıkarılamamalı."""
        result = automata.evaluate_unseen_patterns(list("ab"), list("ab"), pattern_length=3)
        self.assertEqual(result, [])


class TestTransitionMatrix(unittest.TestCase):

    def test_rows_sum_to_one(self):
        """Her satırın toplamı 1 (normalize) olmalı."""
        seq = list("abcabcabc")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)

    def test_no_zero_entries(self):
        """Epsilon smoothing nedeniyle sıfır eleman olmamalı."""
        seq = list("aaa")
        matrix = automata.build_transition_matrix(seq, alphabet_size=4)
        self.assertTrue(np.all(matrix > 0))

    def test_known_transition(self):
        """a→b geçişi baskın olan sequence'de P(a→b) yüksek olmalı."""
        seq = list("ababababab")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        # a=0, b=1
        self.assertGreater(matrix[0, 1], 0.5)


class TestTransitionsAndPathProb(unittest.TestCase):

    def test_transitions_count(self):
        """N uzunluklu pencerede N-1 geçiş olmalı."""
        seq = list("abcabc")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        window = list("abc")
        transitions, _ = automata.get_transitions_and_path_prob(window, matrix, 3)
        self.assertEqual(len(transitions), 2)  # a→b, b→c

    def test_path_probability_range(self):
        """Path probability (0, 1] aralığında olmalı."""
        seq = list("abcabc")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        window = list("abc")
        _, path_prob = automata.get_transitions_and_path_prob(window, matrix, 3)
        self.assertGreater(path_prob, 0)
        self.assertLessEqual(path_prob, 1)

    def test_path_probability_is_product(self):
        """Path probability, bireysel geçiş olasılıklarının çarpımına eşit olmalı."""
        seq = list("abcabc")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        window = list("abc")
        transitions, path_prob = automata.get_transitions_and_path_prob(window, matrix, 3)
        expected = 1.0
        for t in transitions:
            expected *= t["probability"]
        self.assertAlmostEqual(path_prob, round(expected, 10), places=8)

    def test_single_char_window(self):
        """Tek karakterlik pencerede geçiş olmamalı, path_prob=1.0 olmalı."""
        seq = list("abcabc")
        matrix = automata.build_transition_matrix(seq, alphabet_size=3)
        transitions, path_prob = automata.get_transitions_and_path_prob(["a"], matrix, 3)
        self.assertEqual(len(transitions), 0)
        self.assertEqual(path_prob, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)