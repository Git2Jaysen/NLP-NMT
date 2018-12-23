# coding: utf-8

import math
from collections import defaultdict

def _get_ngrams(sentences, max_order):
    """Computing all candidate n-grams where n in [1, max_order].

    Args:
        segment: a 2-D list, true or pred tokens of sentences.
        max_order: int, the max order in BELU equation.

    Returns:
        A list of dicts containing all the candidates and their counts with
        window size from [1, max_order].
    """
    ngrams = []
    for n in range(1, max_order+1):
        candidates = defaultdict(int)
        for sentence in sentences:
            start = 0
            while start <= len(sentence)-n:
                window = sentence[start: start+n]
                candidates[' '.join([str(s) for s in window])] += 1
                start += 1
        ngrams.append(candidates)
    return ngrams

def _BP(r, c):
    """Caculating brevity penalty score.

    Args:
        r: int, true number of words.
        c: int, pred number of words.

    Returns:
        A float scalar.
    """
    return math.exp(1-r/c) if c <= r else 1

def _Pn(refer_candidates, trans_candidates):
    """Caculating Pn in BELU score equation.

    Args:
        refer_candidates: a dict, all the candidate ngrams of references.
        trans_candidates: a dict, all the candidate ngrams of translations.

    Returns:
        A float scalar denoting smoothing Pn.
    """
    refer_count, trans_count = 0, 0
    for word, count in refer_candidates.items():
        refer_count += count
    for word, count in trans_candidates.items():
        trans_count += min(count, refer_candidates[word])
    return (trans_count + 1.) / (refer_count + 1.)

def BELU(references, translations, max_order=4):
    """BELU score for evaluating translation results.

    Args:
        references: a 2-D list, true tokens of the target language.
        translations: a 2-D list, pred tokens of the target language.
        max_order: int, the max order in BELU equation.

    Returns:
        A float score in [0, 1].
    """
    refer_ngrams = _get_ngrams(references, max_order)
    trans_ngrams = _get_ngrams(translations, max_order)
    bp, ep = _BP(len(refer_ngrams), len(trans_ngrams)), 0
    for i in range(max_order):
        ep += (1. / max_order) * math.log(_Pn(refer_ngrams[i], trans_ngrams[i]))
    return bp * math.exp(ep)
