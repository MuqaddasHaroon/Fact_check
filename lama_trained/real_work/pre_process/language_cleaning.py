from langdetect import detect
from langdetect.detector_factory import DetectorFactory
import stanza
import spacy
import re
import emoji
import subprocess
from transformers import MarianMTModel, MarianTokenizer

# Fix random behavior in langdetect
DetectorFactory.seed = 0

# Define language support
spacy_model_mapping = {
    "eng": "en_core_web_sm",  # English
    "fra": "fr_core_news_sm",  # French
    "deu": "de_core_news_sm",  # German
    "spa": "es_core_news_sm",  # Spanish
    "ita": "it_core_news_sm",  # Italian
    "por": "pt_core_news_sm",  # Portuguese
    "zho": "zh_core_web_sm",  # Chinese
}

stanza_supported_languages = [
    "ar", "hi", "ur", "th", "ms", "tgl", "ceb", "sw", "zu"
]  # Supported Stanza languages

# Global dictionary to store initialized Stanza pipelines
stanza_pipelines = {}

def get_stanza_pipeline(lang):
    """Get or initialize a Stanza pipeline for a given language."""
    if lang in stanza_pipelines:
        return stanza_pipelines[lang]

    print(f"Initializing Stanza pipeline for language: {lang}")
    # Try to download the pipeline with `tokenize` as the only processor if others are unavailable
    try:
        stanza.download(lang, processors="tokenize")  # Minimal processors
        nlp = stanza.Pipeline(lang, processors="tokenize", use_gpu=True)
    except Exception as e:
        print(f"Failed to initialize Stanza pipeline for {lang}: {e}")
        nlp = None

    stanza_pipelines[lang] = nlp
    return nlp

def spacy_cleaning_pipeline(text, lang):
    """
    Clean text using spaCy, removing stopwords, punctuation, and unwanted tokens.
    """
    if lang not in spacy_model_mapping:
        print(f"No spaCy model available for language: {lang}")
        return text

    model = spacy_model_mapping[lang]
    nlp = spacy.load(model)

    # Process the text with spaCy
    doc = nlp(text)

    # Retain alphabetic tokens and URLs, remove stopwords/punctuation
    cleaned_tokens = [
        token.text for token in doc
        if not token.is_stop and (token.is_alpha or token.like_url)
    ]

    return " ".join(cleaned_tokens)

def regex_cleaning_pipeline(text):
    """Fallback cleaning for unsupported languages."""
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    text = emoji.demojize(text)  # Replace emojis with descriptive text
    text = re.sub(r':\w+:', '', text)  # Remove emoji placeholders
    return text


def translate_to_english(text, translator_tokenizer, translator_model ):
    """Translate text to English using MarianMT."""
    try:
        inputs = translator_tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        translated = translator_model.generate(**inputs)
        return translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error translating text: {text}. Error: {e}")
        return text  
    
import unicodedata
import re

def clean_text_unicode(text):
    # Normalize Unicode characters (e.g., small caps to normal text)
    normalized_text = unicodedata.normalize('NFKD', text)

    # Remove non-ASCII characters (optional, if you want to keep only plain text)
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('utf-8')

    # Remove unnecessary symbols, bullet points, and excess spaces
    cleaned_text = re.sub(r'[^\w\s.,!?]', '', ascii_text)  # Keep letters, numbers, basic punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove multiple spaces

    return cleaned_text

def process_text(text):
    """Clean text using language-aware tools."""
    if not isinstance(text, str) or not text.strip():
        return text  # Return as-is for invalid text

    cleaned_text = text  # Default to original text
    lang = "unknown"
    try:
        # Detect language
        lang = detect(text)

        # Use spaCy if supported
        if lang in spacy_model_mapping:
            model = spacy_model_mapping[lang]
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            cleaned_text = spacy_cleaning_pipeline(text, lang)
        elif lang in stanza_supported_languages:
            nlp = get_stanza_pipeline(lang)
            if nlp:
                doc = nlp(text)
                cleaned_text = " ".join(
                    [word.text for sentence in doc.sentences for word in sentence.words]
                )
                cleaned_text = regex_cleaning_pipeline(cleaned_text)
            else:
                cleaned_text = regex_cleaning_pipeline(text)
        else:
            cleaned_text = regex_cleaning_pipeline(text)

    except Exception as e:
        cleaned_text = clean_text_unicode(text)

    return cleaned_text



def process_text_pair(original, translated):
    text = original 

    """Clean text using language-aware tools."""
    if not isinstance(text, str) or not text.strip():
        return text  # Return as-is for invalid text

    cleaned_text = ""
    lang = "unknown"
    try:
        # Detect language
        lang = detect(text)

        # Use spaCy if supported
        if lang in spacy_model_mapping:
            model = spacy_model_mapping[lang]
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            cleaned_text = spacy_cleaning_pipeline(text, lang)
        elif lang in stanza_supported_languages:
            nlp = get_stanza_pipeline(lang)
            if nlp:
                doc = nlp(text)
                cleaned_text = " ".join(
                    [word.text for sentence in doc.sentences for word in sentence.words]
                )
                cleaned_text = regex_cleaning_pipeline(cleaned_text)
            else:
                cleaned_text = regex_cleaning_pipeline(text)
        else:
            cleaned_text = regex_cleaning_pipeline(text)

        return cleaned_text

    except Exception as e:
        text = translated
        cleaned_text = regex_cleaning_pipeline(text)
        return cleaned_text