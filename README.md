# 📝 Next Word Prediction System using LSTM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An **end-to-end Natural Language Processing (NLP) deep learning application** designed to predict the next word in a sequence of text using a multi-layer **Long Short-Term Memory (LSTM)** Recurrent Neural Network built with **TensorFlow / Keras**.

The project features tokenization, sequence padding, dynamic custom-object deserialization for cross-version compatibility, and an interactive **Streamlit web dashboard** for real-time text completion.

---

## 📌 Project Overview

* **Sequence Modeling Engine:** Uses a stacked LSTM network trained on textual sequence data to model context and predict the next token based on word index probability mappings (`Softmax` activation).
* **Robust Text Preprocessing:** Leverages `Keras Tokenizer` and `pad_sequences` (pre-padding) to ensure input texts match the training context window length.
* **Deserialization Compatibility:** Custom `GlorotUniform` handling in `app.py` ensures seamless model loading across varying TensorFlow / Keras versions.
* **Interactive Web UI:** Built with Streamlit featuring dynamic card displays, custom CSS styling, and sidebar metadata readouts.

---

## 🛠️ Key Features & Tech Stack

| Domain | Tools / Technologies |
| :--- | :--- |
| **Language** | **Python 3.10+** |
| **Deep Learning** | **TensorFlow / Keras**, **LSTM**, **Recurrent Neural Networks (RNN)** |
| **NLP & Data Processing** | **Tokenizer**, **Sequence Padding**, **NumPy**, **Pickle** |
| **Deployment & UI** | **Streamlit**, **Custom CSS** |
| **Development Environment** | **VS Code**, **Jupyter Notebooks (`.ipynb`)**, **Python `.venv`** |

---

## ⚙️ Model Architecture & Pipeline Specs

| Attribute | Specification |
| :--- | :--- |
| **Task Type** | Multi-class Token Classification (Next Word Prediction) |
| **Embedding Layer** | Dense vector representation for input tokens |
| **Recurrent Layers** | Stacked LSTM Layers (256 units & 128 units) with Dropout (0.2) |
| **Output Layer** | Dense layer with `Softmax` activation (Vocabulary Size dimension) |
| **Loss Function** | `categorical_crossentropy` |
| **Optimizer** | Adam |

---

## 📁 Repository Structure

```text
├── .venv/                   # Virtual environment directory
├── experiments.ipynb        # Data preprocessing, Tokenizer fit, & LSTM model training
├── app.py                   # Modern Streamlit web application
├── next_word_lstm.keras     # Trained TensorFlow/Keras LSTM model
├── tokenizer.pickle         # Serialized Keras Tokenizer object
├── requirements.txt         # Environment dependency manifest
└── README.md                # Project documentation

```
---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone [https://github.com/atejeendra-ba/LSTM-Next-Word-Prediction.git](https://github.com/atejeendra-ba/LSTM-Next-Word-Prediction.git)
cd LSTM-Next-Word-Prediction
```
### **2. Set Up Virtual Environment**

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```
#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **4. Launch the Streamlit App**
```bash
streamlit run app.py
```
### **5. Explicit Virtual Environment Launch (Windows cmd):**
```bash
..\.venv\Scripts\python.exe -m streamlit run app.py
```

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 👤 Author

**A Tejeendra**
* **GitHub:** [@ATejeendra](https://github.com/atejeendra-ba)
* **LinkedIn:** [A Tejeendra](https://www.linkedin.com/in/a-tejeendra/)

## 🙏 Acknowledgements

* **TensorFlow & Keras Documentation**
