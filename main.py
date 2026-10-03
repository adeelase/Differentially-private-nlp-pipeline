import re
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Check CUDA Availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clean_text(text: str) -> list:
    """Cleans, tokenizes, removes stopwords, and lemmatizes input text."""
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    tokens = word_tokenize(text)
    return [
        lemmatizer.lemmatize(w) for w in tokens 
        if w not in stop_words and len(w) > 1
    ]


def generate_laplacian_noise_vector(dim: int = 50, sensitivity: float = 1.0, epsilon: float = 25.0) -> np.ndarray:
    """Generates a d-dimensional continuous Multivariate Laplacian Noise vector."""
    gaussian_samples = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(gaussian_samples)
    direction = gaussian_samples / norm if norm != 0 else gaussian_samples
    
    scale = sensitivity / epsilon
    magnitude = np.random.gamma(shape=dim, scale=scale)
    
    return direction * magnitude


def replace_word(word: str, model: KeyedVectors, epsilon: float = 25.0, sensitivity: float = 1.0, top_k: int = 5) -> str:
    """Applies LDP vector perturbation and retrieves top-k stochastic nearest neighbor."""
    if word not in model:
        return word  # OOV Fallback
    
    original_vec = model[word]
    noise = generate_laplacian_noise_vector(dim=model.vector_size, sensitivity=sensitivity, epsilon=epsilon)
    noisy_vec = original_vec + noise
    
    try:
        similar_words = model.most_similar(positive=[noisy_vec], topn=top_k)
        candidates = [candidate for candidate, sim in similar_words]
        return np.random.choice(candidates)
    except Exception:
        return word  # Exception Fallback


def obfuscate_tokens(tokens: list, model: KeyedVectors, epsilon: float = 25.0) -> str:
    """Processes list of tokens through the LDP obfuscation pipeline."""
    obfuscated = [replace_word(token, model, epsilon=epsilon) for token in tokens]
    return " ".join(obfuscated)


def process_pipeline_batched(df: pd.DataFrame, text_col: str, model: KeyedVectors, 
                             batch_size: int = 100, output_csv: str = "privatized_output.csv"):
    """CUDA-aware batched pipeline processor streaming output to CSV."""
    total_rows = len(df)
    first_batch = True
    
    # Cast embeddings to GPU if available
    embedding_matrix = torch.tensor(model.vectors, dtype=torch.float32).to(device)
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        batch_df['clean_tokens'] = batch_df[text_col].apply(clean_text)
        batch_df['obfuscated_text'] = batch_df['clean_tokens'].apply(
            lambda tokens: obfuscate_tokens(tokens, model)
        )
        
        export_df = batch_df.drop(columns=['clean_tokens'])
        
        if first_batch:
            export_df.to_csv(output_csv, mode='w', index=False, header=True)
            first_batch = False
        else:
            export_df.to_csv(output_csv, mode='a', index=False, header=False)
            
    print(f"Dataset successfully privatized and saved to {output_csv}")