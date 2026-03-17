# 🎬 IMDB Sentiment Analysis 
## (BiLSTM vs BERT)

A comparative NLP project that builds and evaluates two sentiment classifiers on the IMDB movie review dataset, deployed as a live side-by-side demo on gradio.

**🔗 Live Demo:** [huggingface.co/spaces/yxnmei/imdb-sentiment-analyzer](https://huggingface.co/spaces/yxnmei/imdb-sentiment-analyzer)

---

## 📊 Results

| Model | Accuracy | ROC-AUC | Neg F1 | Pos F1 |
|-------|----------|---------|--------|--------|
| Bidirectional LSTM | 80.5% | 0.886 | 0.78 | 0.82 |
| BERT (fine-tuned) | 91.2% | 0.968 | 0.91 | 0.91 |

BERT achieves ~11% higher accuracy by leveraging transfer learning from pretraining on 3.3 billion words, requiring only 3 epochs of fine-tuning compared to the LSTM's training from scratch.

---

## 🧠 Models

### Bidirectional LSTM
- Trained from scratch on 25,000 IMDB reviews
- Architecture: `Embedding → BiLSTM (2 layers) → Dropout → Linear → Sigmoid`
- Custom vocabulary of 20,000 tokens with padding
- Weights hosted at [yxnmei/imdb-lstm-sentiment](https://huggingface.co/yxnmei/imdb-lstm-sentiment)

### BERT (fine-tuned)
- `bert-base-uncased` fine-tuned on a 5,000-sample subset
- Architecture: `BERT → [CLS] token → Dropout → Linear → Sigmoid`
- WordPiece tokenization with max length 256
- Weights hosted at [yxnmei/imdb-bert-sentiment](https://huggingface.co/yxnmei/imdb-bert-sentiment)

---

## 📁 Project Structure

```
imdb-sentiment-analyzer/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis & text cleaning
│   ├── 02_lstm_model.ipynb       # BiLSTM training & evaluation
│   └── 03_bert_finetune.ipynb    # BERT fine-tuning & comparison
├── app.py                        # Gradio demo (HuggingFace Spaces)
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch |
| NLP | HuggingFace Transformers |
| Models | `bert-base-uncased`, custom BiLSTM |
| Dataset | [IMDB (HuggingFace Datasets)](https://huggingface.co/datasets/imdb) |
| Evaluation | scikit-learn (F1, ROC-AUC, confusion matrix) |
| Demo | Gradio |
| Hosting | HuggingFace Spaces & Hub |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yxnmei/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebooks in order**
```
notebooks/01_eda.ipynb          → generates train_clean.csv, test_clean.csv
notebooks/02_lstm_model.ipynb   → trains BiLSTM, saves models/lstm_best.pt
notebooks/03_bert_finetune.ipynb → fine-tunes BERT, pushes to HuggingFace Hub
```

**4. Launch the Gradio app locally**
```bash
pip install gradio
python app.py
```

---

## 💡 Key Takeaways

- **Transfer learning wins** — BERT achieves 91% accuracy in 3 epochs vs LSTM's 80% in 5 epochs, purely because of pretraining
- **WordPiece tokenization** handles out-of-vocabulary words better than simple word splitting — BERT almost never hits a truly unknown token
- **Trade-offs matter** — BERT is 20× larger (~110M vs ~5.4M parameters), making BiLSTM a better fit for latency-sensitive or resource-constrained applications
- **Windows compatibility** — `num_workers=0` required in PyTorch DataLoaders on Windows due to multiprocessing constraints