import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.utils import register_keras_serializable

# Page Configuration
st.set_page_config(
    page_title="Next Word Predictor | LSTM",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    /* Card Container Styling */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    /* Subtle Divider Accent */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, rgba(59, 130, 246, 0) 100%);
        margin: 20px 0;
        border-radius: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# Register custom GlorotUniform to safely strip unexpected fields ('input_axes', 'output_axes')
@register_keras_serializable(package="CustomInitializers")
class CompatibleGlorotUniform(GlorotUniform):
    def __init__(self, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return super().from_config(config)

# Resolve absolute paths relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'next_word_lstm.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pickle')

# Load Model & Tokenizer Safely
@st.cache_resource
def load_assets():
    model = load_model(
        MODEL_PATH,
        custom_objects={
            'GlorotUniform': CompatibleGlorotUniform,
            'CompatibleGlorotUniform': CompatibleGlorotUniform
        },
        compile=False
    )
    
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

# Load resources with loading spinner
with st.spinner("Initializing Deep Learning Engine..."):
    model, tokenizer = load_assets()

# Prediction Function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    text_cleaned = text.lower()
    token_list = tokenizer.texts_to_sequences([text_cleaned])[0]
    
    if not token_list:
        return None
        
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-max_sequence_len:]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Sidebar Information ---
with st.sidebar:
    st.header("🧠 Model Architecture")
    st.markdown("""
    This app leverages a **Recurrent Neural Network (RNN)** with **LSTM (Long Short-Term Memory)** layers trained on textual sequence data.
    """)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Model Metadata Readout
    vocab_size = len(tokenizer.word_index) + 1 if hasattr(tokenizer, 'word_index') else "N/A"
    seq_len = model.input_shape[1] if hasattr(model, 'input_shape') else "N/A"
    
    st.markdown(f"**Vocabulary Size:** `{vocab_size}` words")
    st.markdown(f"**Max Sequence Length:** `{seq_len}` tokens")
    st.markdown("**Status:** `Ready` 🟢")

# --- Main App Interface ---
st.title("🤖 Next Word Prediction")
st.caption("Deep Learning Sequence Generator built with TensorFlow & Streamlit")

st.markdown("""
Enter a sequence of words below, and the trained LSTM model will predict the most probable word to complete your phrase.
""")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# User Input Form
with st.form("prediction_form"):
    input_text = st.text_input(
        "Enter Starting Text / Phrase:",
        value="Mar. Horatio saies, 'tis but our",
        placeholder="Type a phrase here...",
        help="The model relies on your input tokens to infer the single next token."
    )
    
    submit_button = st.form_submit_button("✨ Predict Next Word", use_container_width=True)

# Run Inference
if submit_button:
    if input_text.strip():
        max_sequence_len = model.input_shape[1]
        
        with st.spinner("Analyzing context and generating token..."):
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        
        if next_word:
            # Highlighted Prediction Display
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Next Word</div>
                    <div class="metric-value">{next_word}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show completed phrase in expander
            full_text = f"{input_text.strip()} **{next_word}**"
            st.info(f"**Completed Sequence:** {full_text}")
        else:
            st.warning("⚠️ Could not predict the next word. The input vocabulary might not match the trained dataset.")
    else:
        st.warning("⚠️ Please enter a text sequence first.")