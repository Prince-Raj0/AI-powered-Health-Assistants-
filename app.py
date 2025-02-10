import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

st.title("AI-Powered Health Assistant")

user_input = st.text_area("Enter your health query:", height=150)

if st.button("Submit"):
    if user_input:
        try:
            # Preprocessing
            text = user_input.lower()
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]  # Keep alphanumeric words
            processed_input = " ".join(filtered_tokens)

            messages = [{"role": "user", "content": processed_input}]

            # Model Loading and Generation (using a try-except for the pipeline)
            try:
                pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)
                response = pipe(messages, max_new_tokens=200)

                st.write("AI Assistant's Response:")

                if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
                    st.write(response[0]['generated_text'])
                elif isinstance(response, str):
                    st.write(response)
                else:
                    st.write(response)  # For debugging

            except Exception as model_err:
                st.error(f"Error with the language model: {model_err}")  # More specific error message

        except Exception as e:  # Catch any preprocessing or other errors
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")



def preprocess_text(text):  # Example of more advanced preprocessing (not used directly, but available)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]
    # Add stemming or lemmatization here if needed (e.g., Porter Stemmer, WordNet Lemmatizer)
    return " ".join(filtered_tokens)
