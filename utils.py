import re

def clean_text(text):
    """
    Cleans the input text for model prediction.
    Performs basic preprocessing:
    - Lowercase conversion
    - Remove punctuation and special characters
    - Remove numbers
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
