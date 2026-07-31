"""app.dl - Phase 5 (Deep Learning), Milestone 1.

The neural upgrade to the Phase 4 intent classifier:

    neural_classifier:  IntentMLP (TF-IDF -> hidden/ReLU/dropout ->
                        softmax) behind the IntentClassifier API.
    train:              Backpropagation training loop with early
                        stopping, plus an honest head-to-head against
                        the Phase 4 logistic regression baseline.
"""
