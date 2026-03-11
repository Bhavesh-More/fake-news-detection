import pandas as pd
import numpy as np
import os
import re

#  EMOTIONAL / SENSATIONAL WORDS LIST
EMOTIONAL_WORDS = [
    "shocking", "unbelievable", "breaking", "urgent", "exposed",
    "hidden", "secret", "truth", "conspiracy", "hoax", "alert",
    "warning", "banned", "censored", "leaked", "scandal", "fraud",
    "bombshell", "outrage", "horrifying", "disgusting", "corrupt"
]

#  FEATURE EXTRACTION FUNCTIONS

def is_emotional(text):
    """Check if text contains emotional/sensational words."""
    text_lower = text.lower()
    for word in EMOTIONAL_WORDS:
        if word in text_lower:
            return 1
    return 0

def has_excessive_caps(text):
    """Check if text has excessive capital letters (more than 30% caps)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0
    caps = [c for c in letters if c.isupper()]
    return 1 if (len(caps) / len(letters)) > 0.3 else 0

def has_numbers(text):
    """Check if text contains numbers or statistics."""
    return 1 if re.search(r'\d+', text) else 0

def is_short_article(text):
    """Check if article is very short (less than 50 words)."""
    words = text.split()
    return 1 if len(words) < 50 else 0

def extract_features(text):
    """
    Given raw text, return a dictionary of features.
    This is used both during training and during live user inference.
    """
    return {
        'is_emotional'  : is_emotional(text),
        'title_caps'    : has_excessive_caps(text),
        'has_numbers'   : has_numbers(text),
        'short_article' : is_short_article(text),
    }

#  DATASET LOADING AND PROCESSING

def load_and_process(fake_path, real_path, output_path):
    """
    Load Fake.csv and True.csv, extract features from each article,
    and save the final feature matrix to features.csv.
    """
    print("Loading dataset")

    # Load raw CSVs
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Add labels: 1 = Fake, 0 = Real
    fake_df['label'] = 1
    real_df['label'] = 0

    # Combine into one dataset
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Use title + text combined for feature extraction
    # If 'text' column is missing or empty, fall back to 'title' only
    if 'text' in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    else:
        df['content'] = df['title'].fillna('')

    print(f"Total articles loaded: {len(df)}")
    print("Extracting features")

    # Extract features for every article
    features = df['content'].apply(extract_features)
    features_df = pd.DataFrame(features.tolist())

    # Add label column
    features_df['label'] = df['label'].values

    # Save processed features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"  Features saved to: {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Fake articles : {features_df['label'].sum()}")
    print(f"  Real articles : {(features_df['label'] == 0).sum()}")
    print()

    return features_df
