"""
WORD2VEC - SKIP-GRAM (from scratch, pure NumPy)
================================================

This file is a line-by-line implementation of the pseudocode:

    WORD2VEC-SKIPGRAM(Corpus, c, d, eta, T)

    1. Build vocabulary V from the corpus
    2. Initialize input embeddings E randomly
    3. Initialize output weights W randomly
    4. Generate training pairs:
         For each sentence in Corpus:
           For each word w at position i:
             For each context position j within window c:
               If i != j:
                 Add pair (w, context_word)
    5. For epoch = 1 to T:
         Shuffle training pairs
         For each pair (center, context):
             # Forward pass
             h = E[center]
             scores = h x W
             probabilities = Softmax(scores)

             # Compute loss
             loss = -log(probabilities[context])

             # Backpropagation
             error = probabilities
             error[context] -= 1

             gradient_W = h^T x error
             gradient_h = W x error

             # Update parameters
             W = W - eta * gradient_W
             E[center] = E[center] - eta * gradient_h
    6. Return E

Everything below is self-contained: no external explanation needed, every
step is commented with WHAT is happening and WHY, so you can read this file
top to bottom and fully understand Skip-gram word2vec.

Notation used throughout (matches the pseudocode):
    Corpus : list of sentences, each sentence is a list of words (tokens)
    c      : window size (how many words to the left/right count as context)
    d      : embedding dimension (size of each word vector)
    eta    : learning rate (step size for gradient descent)
    T      : number of training epochs
    V      : vocabulary size (number of unique words)
    E      : input embedding matrix, shape (V, d) -> one row per word
    W      : output weight matrix, shape (d, V) -> used to score every
             vocabulary word against a given center word's embedding
"""

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Build vocabulary V from the corpus
# ---------------------------------------------------------------------------
def build_vocabulary(corpus):
    """
    Scans every sentence/word in the corpus and assigns each UNIQUE word
    an integer id. Neural networks cannot operate on raw strings, so this
    id is what we will actually use to index into the embedding matrix E.

    Returns:
        word2idx: dict mapping word -> integer id   (e.g. "cat" -> 3)
        idx2word: list mapping integer id -> word    (inverse of word2idx)
    """
    vocabulary = set()
    for sentence in corpus:
        for word in sentence:
            vocabulary.add(word)

    # sorted() makes the id assignment deterministic/reproducible across runs
    idx2word = sorted(vocabulary)
    print(idx2word)
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    print(word2idx)
    return word2idx, idx2word


# ---------------------------------------------------------------------------
# Step 4: Generate training pairs (center word, context word)
# ---------------------------------------------------------------------------
def generate_training_pairs(corpus, word2idx, window_size):
    """
    Skip-gram training pairs are built by sliding a window over each
    sentence. For every word w at position i, every OTHER word that falls
    within `window_size` positions to the left or right is treated as a
    "context word" that word w should be able to predict.

    Example: sentence = ["the", "cat", "sat", "on", "the", "mat"], c = 2
             center word "sat" (position i=2) pairs with:
               "the" (i-2), "cat" (i-1), "on" (i+1), "the" (i+2)
             -> pairs: (sat, the), (sat, cat), (sat, on), (sat, the)

    Returns:
        pairs: list of (center_id, context_id) integer tuples
    """
    pairs = []
    for sentence in corpus:
        sentence_len = len(sentence)
        for i, w in enumerate(sentence):
            center_id = word2idx[w]

            # j ranges over every position within the window around i,
            # clipped to the sentence boundaries so we never index out of range
            window_start = max(0, i - window_size)
            window_end = min(sentence_len, i + window_size + 1)  # +1: range() end is exclusive

            for j in range(window_start, window_end):
                if i != j:  # "If i != j" from the pseudocode: skip the center word itself
                    context_word = sentence[j]
                    context_id = word2idx[context_word]
                    pairs.append((center_id, context_id))
    return pairs


# ---------------------------------------------------------------------------
# Softmax: turns raw scores into a probability distribution over the vocabulary
# ---------------------------------------------------------------------------
def softmax(scores):
    """
    probabilities[k] = exp(scores[k]) / sum_over_all_j(exp(scores[j]))

    We subtract max(scores) before exponentiating purely for NUMERICAL
    STABILITY (prevents exp() from overflowing for large scores). This is
    mathematically a no-op: it does not change the resulting probabilities,
    because the max cancels out in the ratio of exponentials.
    """
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


# ---------------------------------------------------------------------------
# Steps 2, 3, 5, 6: the main training routine
# ---------------------------------------------------------------------------
def word2vec_skipgram(corpus, c, d, eta, T, seed=42, verbose=True):
    """
    Trains Skip-gram word2vec embeddings from scratch using plain gradient
    descent (one training pair processed at a time, i.e. SGD with batch
    size = 1), exactly as described in the pseudocode.

    Parameters:
        corpus : list[list[str]] - tokenized sentences
        c      : int   - context window size
        d      : int   - embedding dimensionality
        eta    : float - learning rate
        T      : int   - number of epochs (full passes over all pairs)
        seed   : int   - RNG seed, so results are reproducible
        verbose: bool  - print average loss per epoch

    Returns:
        E        : np.ndarray of shape (V, d) - the trained word embeddings
        word2idx : dict mapping word -> row index in E
        idx2word : list mapping row index -> word
    """
    rng = np.random.default_rng(seed)

    # ---- 1. Build vocabulary V from the corpus -----------------------
    word2idx, idx2word = build_vocabulary(corpus)
    V = len(idx2word)

    # ---- 2. Initialize input embeddings E randomly --------------------
    # E has one row per vocabulary word; E[word_id] is that word's embedding.
    # Small random values (scaled by 0.01) keep initial scores/gradients small,
    # which helps training start smoothly instead of saturating softmax.
    E = rng.normal(loc=0.0, scale=0.01, size=(V, d))

    # ---- 3. Initialize output weights W randomly -----------------------
    # W maps a center word's embedding (size d) to a score for every
    # vocabulary word (size V), so W has shape (d, V).
    W = rng.normal(loc=0.0, scale=0.01, size=(d, V))

    # ---- 4. Generate training pairs ------------------------------------
    pairs = generate_training_pairs(corpus, word2idx, window_size=c)

    # ---- 5. Training loop ------------------------------------------------
    for epoch in range(1, T + 1):
        # "Shuffle training pairs": prevents the network from learning
        # spurious patterns tied to sentence order, and improves SGD convergence.
        shuffled_pairs = pairs.copy()
        rng.shuffle(shuffled_pairs)

        total_loss = 0.0

        for center, context in shuffled_pairs:
            # ----------------- Forward pass -----------------
            h = E[center]                 # h: (d,)   the center word's current embedding
            scores = h @ W                # scores: (V,)  similarity of h to every word's output vector
            probabilities = softmax(scores)  # probabilities: (V,) distribution over the vocabulary

            # ----------------- Compute loss ------------------
            # Cross-entropy loss: how "surprised" the model is that the true
            # context word got probability probabilities[context].
            # Add a tiny epsilon to avoid log(0) if a probability underflows to 0.
            loss = -np.log(probabilities[context] + 1e-10)
            total_loss += loss

            # ----------------- Backpropagation ----------------
            # For softmax + cross-entropy, the gradient of the loss w.r.t.
            # the raw `scores` simplifies beautifully to:
            #     error = probabilities - one_hot(context)
            # i.e. the whole probability vector, except we subtract 1 at the
            # position of the true context word.
            error = probabilities.copy()  # error: (V,)
            error[context] -= 1

            # d(loss)/dW = outer product of h and error -> shape (d, V), matches W
            gradient_W = np.outer(h, error)

            # d(loss)/dh = W applied to error -> shape (d,), matches h / E[center]
            gradient_h = W @ error

            # ----------------- Update parameters ----------------
            W = W - eta * gradient_W
            E[center] = E[center] - eta * gradient_h

        if verbose:
            avg_loss = total_loss / len(shuffled_pairs)
            print(f"Epoch {epoch:3d}/{T} | avg loss: {avg_loss:.4f}")

    # ---- 6. Return E ------------------------------------------------------
    return E, word2idx, idx2word


# ---------------------------------------------------------------------------
# Helper: inspect the learned embeddings via cosine similarity
# ---------------------------------------------------------------------------
def most_similar(word, E, word2idx, idx2word, top_n=5):
    """
    Finds the `top_n` words whose embeddings are closest (by cosine
    similarity) to the embedding of `word`. This is the standard way to
    sanity-check that word2vec has learned something meaningful: words
    used in similar contexts should end up with similar vectors.
    """
    if word not in word2idx:
        raise ValueError(f"'{word}' is not in the vocabulary.")

    word_vec = E[word2idx[word]]
    word_norm = np.linalg.norm(word_vec) + 1e-10

    similarities = []
    for other_word, idx in word2idx.items():
        if other_word == word:
            continue
        other_vec = E[idx]
        other_norm = np.linalg.norm(other_vec) + 1e-10
        cosine_sim = np.dot(word_vec, other_vec) / (word_norm * other_norm)
        similarities.append((other_word, cosine_sim))

    similarities.sort(key=lambda pair: pair[1], reverse=True)
    return similarities[:top_n]


# ---------------------------------------------------------------------------
# Demo / example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A tiny toy corpus: each sentence is a list of already-tokenized words.
    # In a real project you would lowercase, strip punctuation, and tokenize
    # raw text to produce this structure.
    corpus = [
        ["the", "king", "rules", "the", "kingdom"],
        ["the", "queen", "rules", "the", "kingdom"],
        ["the", "king", "and", "the", "queen", "love", "each", "other"],
        ["the", "man", "walks", "the", "dog"],
        ["the", "woman", "walks", "the", "dog"],
        ["the", "man", "and", "the", "woman", "love", "each", "other"],
        ["paris", "is", "the", "capital", "of", "france"],
        ["berlin", "is", "the", "capital", "of", "germany"],
        ["the", "king", "lives", "in", "paris"],
        ["the", "queen", "lives", "in", "berlin"],
    ]

    # Hyperparameters, named exactly as in the pseudocode:
    c = 2       # window size
    d = 10      # embedding dimension
    eta = 0.05  # learning rate
    T = 200     # epochs

    # ---- Build vocabulary + generate training pairs, just to display them ----
    # (word2vec_skipgram() below regenerates these internally before training;
    # we rebuild them here purely so we can print the (center, context) word
    # pairs in human-readable form and see exactly what step 4 produces.)
    _word2idx, _idx2word = build_vocabulary(corpus)
    _pairs = generate_training_pairs(corpus, _word2idx, window_size=c)

    print(f"Generated {len(_pairs)} training pairs (window size c={c}).")
    print("First 20 pairs (center, context):")
    for center_id, context_id in _pairs[:20]:
        print(f"  ({_idx2word[center_id]}, {_idx2word[context_id]})")
    print("...\n")

    E, word2idx, idx2word = word2vec_skipgram(corpus, c=c, d=d, eta=eta, T=T)

    print("\nVocabulary size:", len(idx2word))
    print("Embedding matrix shape:", E.shape)

    for query_word in ["king", "queen", "paris"]:
        print(f"\nWords most similar to '{query_word}':")
        for word, score in most_similar(query_word, E, word2idx, idx2word, top_n=5):
            print(f"  {word:10s}  cosine_sim = {score:.4f}")
