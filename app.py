import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import os
import tempfile

# 1. NLTK Data Path (using tempfile)
try:
    NLTK_DATA_PATH = os.path.join(tempfile.gettempdir(), "nltk_data")
except:
    NLTK_DATA_PATH = ".nltk_data"

if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)

nltk.data.path.append(NLTK_DATA_PATH)

# 2. Download NLTK data (using the correct resource name and path)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH)


# 3. Load Mistral model (outside the Streamlit execution flow for caching)
@st.cache_resource  # Cache the model loading
def load_mistral_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.bfloat16) # Add device_map and torch_dtype
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.bfloat16)  # Create pipeline here
    return pipe  # Return the pipeline

mistral_pipe = load_mistral_model()  # Load model and create pipeline


st.title("AI-Powered Health Assistant")

user_input = st.text_area("Enter your health query:", height=150)

if st.button("Submit"):
    if user_input:
        try:
            # Preprocessing
            try:  # Error handling for preprocessing
                text = user_input.lower()
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]
                processed_input = " ".join(filtered_tokens)
            except Exception as preprocessing_err:
                st.error(f"Error during preprocessing: {preprocessing_err}")
                processed_input = ""  # Set to empty to avoid further errors

            messages = [{"role": "user", "content": processed_input}]

            # Model Generation (using the pre-loaded pipeline)
            try:
                response = mistral_pipe(messages, max_new_tokens=200)

                st.write("AI Assistant's Response:")

                if isinstance(response, list) and len(response) > 0 and 'generated_text' in response[0]:
                    st.write(response[0]['generated_text'])
                elif isinstance(response, str):
                    st.write(response)
                else:
                    st.write(response)  # For debugging

            except Exception as model_err:
                st.error(f"Error with the language model: {model_err}")

        except Exception as e: # Catches errors *outside* of preprocessing and model
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")



def preprocess_text(text):  # Example of more advanced preprocessing (not used directly, but available)
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if not w in stop_words and w.isalnum()]
        # Add stemming or lemmatization here if needed
        return " ".join(filtered_tokens)
    except Exception as preprocess_error:
        st.error(f"Error in preprocess_text function: {preprocess_error}")
        return "" # Return empty string in case of error

