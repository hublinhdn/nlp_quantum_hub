"""Classical baselines cho QNLP comparison.

Modules:
    common     — load qnlp_* CSV + compute metrics (acc/F1/AUC)
    lr         — Logistic Regression + TF-IDF
    bilstm     — BiLSTM 1-layer PyTorch
    glove      — GloVe pretrained vector loader (optional)
    lr_glove   — Logistic Regression + averaged GloVe (optional)
"""
