import gradio as gr
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download
import json
import re

# ─── Config ───────────────────────────────────────────────────────────────────
BERT_REPO = "yxnmei/imdb-bert-sentiment"
LSTM_REPO = "yxnmei/imdb-lstm-sentiment"
DEVICE    = torch.device("cpu")

LSTM_CONFIG = {
    "max_len"    : 512,
    "vocab_size" : 20000,
    "embed_dim"  : 128,
    "hidden_dim" : 256,
    "num_layers" : 2,
    "dropout"    : 0.3,
}

BERT_MAX_LEN = 256

# ─── Model Definitions ────────────────────────────────────────────────────────

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        _, (hidden, _)  = self.lstm(embedded)
        hidden          = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden          = self.dropout(hidden)
        return torch.sigmoid(self.fc(hidden)).squeeze(1)


class BERTSentimentClassifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.bert       = BertModel.from_pretrained(BERT_REPO)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls        = self.dropout(out.pooler_output)
        return torch.sigmoid(self.classifier(cls)).squeeze(1)


# ─── Load Models ──────────────────────────────────────────────────────────────

def load_models():
    print("Loading LSTM model...")
    # Download weights + vocab from Hub
    lstm_weights_path = hf_hub_download(repo_id=LSTM_REPO, filename="lstm_best.pt")
    vocab_path        = hf_hub_download(repo_id=LSTM_REPO, filename="vocab.json")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    lstm_model = BiLSTMSentiment(
        vocab_size  = LSTM_CONFIG["vocab_size"],
        embed_dim   = LSTM_CONFIG["embed_dim"],
        hidden_dim  = LSTM_CONFIG["hidden_dim"],
        num_layers  = LSTM_CONFIG["num_layers"],
        dropout     = LSTM_CONFIG["dropout"]
    )
    lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=DEVICE))
    lstm_model.eval()

    print("Loading BERT model...")
    # Load classifier head
    head_path  = hf_hub_download(repo_id=BERT_REPO, filename="classifier_head.pt")
    head_ckpt  = torch.load(head_path, map_location=DEVICE)

    bert_model = BERTSentimentClassifier(dropout=head_ckpt["dropout"])
    bert_model.classifier.load_state_dict(head_ckpt["classifier_state_dict"])
    bert_model.eval()

    tokenizer = BertTokenizer.from_pretrained(BERT_REPO)

    print("✅ Both models loaded!")
    return lstm_model, bert_model, tokenizer, vocab


# Load once at startup
lstm_model, bert_model, tokenizer, vocab = load_models()


# ─── Helper: Text Cleaning ────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


# ─── Prediction Functions ─────────────────────────────────────────────────────

def predict_lstm(text):
    cleaned = clean_text(text)
    tokens  = [vocab.get(w, 1) for w in cleaned.split()]
    tokens  = tokens[:LSTM_CONFIG["max_len"]]
    tokens  = tokens + [0] * (LSTM_CONFIG["max_len"] - len(tokens))

    x = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        prob = lstm_model(x).item()

    label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, round(conf, 4)


def predict_bert(text):
    encoding = tokenizer(
        text,
        max_length=BERT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        prob = bert_model(
            encoding["input_ids"],
            encoding["attention_mask"]
        ).item()

    label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, round(conf, 4)


# ─── Gradio Predict Function ──────────────────────────────────────────────────

def predict(review):
    if not review.strip():
        return "—", "—", "—", "—"

    lstm_label, lstm_conf = predict_lstm(review)
    bert_label, bert_conf = predict_bert(review)

    lstm_conf_str = f"{lstm_conf * 100:.1f}%"
    bert_conf_str = f"{bert_conf * 100:.1f}%"

    return lstm_label, lstm_conf_str, bert_label, bert_conf_str


# ─── Gradio UI ────────────────────────────────────────────────────────────────

examples = [
    ["This movie was absolutely brilliant! The acting was superb and I was on the edge of my seat the entire time."],
    ["Terrible waste of time. The plot made no sense and the characters were completely flat."],
    ["It was okay. Some good moments but overall a bit disappointing."],
    ["One of the best films I've seen in years. A masterpiece!"],
    ["Bruh it was mid."],
    ["Im hooked!"],
]

with gr.Blocks(theme=gr.themes.Soft(), title="🎬 Sentiment Analyzer") as demo:

    gr.Markdown(
        """
        # 🎬 Movie Review Sentiment Analyzer
        ### BiLSTM vs BERT — Side-by-Side Comparison
        Type or paste a movie review below and see how both models classify its sentiment.
        """
    )

    with gr.Row():
        review_input = gr.Textbox(
            label="Movie Review",
            placeholder="e.g. This movie was absolutely brilliant...",
            lines=4,
        )

    analyze_btn = gr.Button("Analyze Sentiment 🔍", variant="primary")

    gr.Markdown("### Results")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🔵 BiLSTM")
            lstm_label_out = gr.Textbox(label="Sentiment", interactive=False)
            lstm_conf_out  = gr.Textbox(label="Confidence", interactive=False)

        with gr.Column():
            gr.Markdown("#### 🟣 BERT")
            bert_label_out = gr.Textbox(label="Sentiment", interactive=False)
            bert_conf_out  = gr.Textbox(label="Confidence", interactive=False)

    analyze_btn.click(
        fn=predict,
        inputs=review_input,
        outputs=[lstm_label_out, lstm_conf_out, bert_label_out, bert_conf_out]
    )

    gr.Examples(
        examples=examples,
        inputs=review_input,
        label="Try these examples 👇"
    )

    gr.Markdown(
        """
        ---
        **Models:**
        - **BiLSTM** — Bidirectional LSTM trained from scratch on IMDB (25k reviews) · ~80% accuracy
        - **BERT** — `bert-base-uncased` fine-tuned on IMDB (5k subset) · ~91% accuracy

        **Dataset:** [IMDB Movie Reviews](https://huggingface.co/datasets/imdb) · 50,000 reviews
        """
    )

demo.launch()