# 📰 AI News Headline Generator

An end-to-end Natural Language Processing (NLP) web application that generates concise headlines from long-form news articles. Built from scratch using a **Sequence-to-Sequence (Seq2Seq)** architecture with **Bahdanau Attention**, trained on a dataset of 300,000+ news samples.

## 🚀 Features
* **Abstractive Summarization:** Generates new sentences rather than just extracting existing ones.
* **Attention Mechanism:** Uses Bahdanau (Additive) Attention to focus on relevant parts of the source text during generation.
* **Custom Inference Engine:** Implements a recursive decoder loop (Greedy Search) for autoregressive text generation.
* **Web Interface:** Clean, responsive UI built with **Flask** and **HTML/CSS** for real-time interaction.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (LSTM, GRU, Custom Layers).
* **Backend:** Python, Flask.
* **Frontend:** HTML5, CSS3.
* **Data Processing:** Pandas, NumPy, NLTK.
* **Hardware Optimization:** Mixed Precision Training (FP16) for RTX 3050 GPU.

## 📂 Project Structure
```text
Headline_Generator/
│
├── app.py                   # Main Flask application & Inference Logic
├── requirements.txt         # Dependencies
├── README.md                # Project Documentation
│
├── models/                  # Trained AI artifacts
│   ├── encoder_model.keras  # The Reader (Bidirectional GRU)
│   ├── decoder_model.keras  # The Writer (LSTM + Attention)
│   └── tokenizer.pkl        # Word-to-Integer Dictionary
│
├── static/
│   └── style.css            # Styling for the web interface
│
└── templates/
    └── index.html           # Frontend HTML structure

```

## 🧠 Model Architecture

The core of this project is a **Seq2Seq** model designed to map long sequences (articles) to short sequences (headlines).

1. **Encoder:** A Bidirectional GRU layer that processes the input text and creates a "Context Vector."
2. **Attention Layer:** A custom-built `BahdanauAttention` layer that allows the Decoder to "look back" at specific words in the article at every step of generation.
3. **Decoder:** An LSTM layer that predicts the headline one word at a time, conditioned on the previous word and the Attention context.

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ritee17/Headline_Generator
cd Headline_Generator

```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install flask tensorflow pandas numpy

```

### 4. Run the Application

```bash
python app.py

```

You will see a message: `Running on http://127.0.0.1:5000`. Open this link in your browser.

## 🚧 Engineering Challenges Solved

* **OOM (Out of Memory) Errors:** The initial model with a 30,000-word vocabulary exceeded the 4GB VRAM limit of the RTX 3050. This was solved by implementing **Mixed Precision (FP16)** training and optimizing batch sizes.
* **Inference Loop:** Unlike training (which uses Teacher Forcing), real-world prediction requires the model to feed its own output back as input. I engineered a custom `decode_sequence` function to handle this recursive state management.
* **Custom Layer Serialization:** Flask cannot natively load custom Keras layers. I implemented a wrapper to inject the `BahdanauAttention` class definition during the model loading process.


## 👤 Author

**Ritee**

* **Role:** AI Engineer & Full-Stack Developer
* **Focus:** Deep Learning, NLP, System Architecture
