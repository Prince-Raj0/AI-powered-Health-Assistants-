import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
import os

# 1. Set NLTK data path (CRITICAL for Streamlit)
NLTK_DATA_PATH = os.path.join(st.cache_resource.get_cache_dir(), "nltk_data")

if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

nltk.data.path.append(NLTK_DATA_PATH)

# 2. Download NLTK data (with spinner and correct path)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)


st.title("AI-Powered Health Assistant")

user_input = st.text_area("Enter your health query:", height=150)

if st.button("Submit"):
    if user_input:
        try:
            # Preprocessing
            text = user_input.lower()
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]
            processed_input = " ".join(filtered_tokens)

            messages = [{"role": "user", "content": processed_input}]

            # Model Loading and Generation (with try-except)
            try:
                pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True) # Or a suitable model
                response = pipe(messages, max_new_tokens=200) # Adjust max_new_tokens

                st.write("AI Assistant's Response:")

                if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
                    st.write(response[0]['generated_text'])
                elif isinstance(response, str):
                    st.write(response)
                else:
                    st.write(response)  # For debugging

            except Exception as model_err:
                st.error(f"Error with the language model: {model_err}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")



def preprocess_text(text):  # Example of more advanced preprocessing (not used directly, but available)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]
    # Add stemming or lemmatization here if needed
    return " ".join(filtered_tokens)
