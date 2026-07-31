"""app.ml - Phase 4 (Machine Learning), Milestone 1.

Supervised intent classification as the ML module. The milestone goal is
the MODULE (dataset + classifier + training/evaluation + persisted
artifact) plus tests; wiring it into the Assistant's reply routing is
Milestone 2.

Modules:
    dataset:            Labeled intent examples + data-quality checks.
    intent_classifier:  TF-IDF + logistic-regression pipeline wrapper.
    train:              Training script: split, evaluate, persist.
"""
