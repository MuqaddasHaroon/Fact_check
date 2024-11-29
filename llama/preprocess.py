import re
import emoji
import pandas as pd
import ast
import pandas as pd
import re
from langdetect import detect
from collections import Counter
from sentencepiece import SentencePieceProcessor
import unicodedata

class TextPreprocessor:
    @staticmethod
    def remove_urls(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'http\S+', '', text)

    @staticmethod
    def remove_emojis(text):
        if not isinstance(text, str):
            return text
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def replace_whitespaces(text):
        if not isinstance(text, str):
            return text
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess(self, text):
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.replace_whitespaces(text)
        return text

# Move these functions outside of the class

def replace_text_with_ocr(df, text_column='text', ocr_column='ocr'):
    df[text_column] = df.apply(
        lambda row: row[ocr_column] if pd.isna(row[text_column]) or row[text_column].strip() == '' else row[text_column],
        axis=1
    )
    return df


def safe_literal_eval(val):
    try:
        if isinstance(val, str):
            parsed_val = ast.literal_eval(val)
            if isinstance(parsed_val, list):
                for item in parsed_val:
                    if isinstance(item, tuple) and len(item) == 3:
                        return item  # Return the first valid tuple
            elif isinstance(parsed_val, tuple) and len(parsed_val) == 3:
                return parsed_val
        return val
    except (ValueError, SyntaxError) as e:
        print(f"Literal eval failed for {val}: {e}")
        return (None, None, None)


def split_text_column(row, row_name):
    try:
        parsed = safe_literal_eval(row[row_name])
        if isinstance(parsed, tuple) and len(parsed) == 3:
            first_text = parsed[0].strip() if isinstance(parsed[0], str) else None
            second_text = parsed[1].strip() if isinstance(parsed[1], str) else None

            # Extract the language with the highest confidence
            if isinstance(parsed[2], list) and all(isinstance(item, tuple) for item in parsed[2]):
                lang_conf = max(parsed[2], key=lambda x: x[1] if len(x) == 2 else 0)
            else:
                lang_conf = (None, None)

            lang = lang_conf[0]
            confidence = lang_conf[1]
            return pd.Series([first_text, second_text, lang, confidence])
    except Exception as e:
        print(f"Error processing row: {row[row_name]} -> {e}")
    return pd.Series([None, None, None, None])

def remove_repeated_chars(text, threshold=0.4):
    """Remove texts with a high ratio of repeated characters."""
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())
    repeated_ratio = max(char_counts.values()) / total_chars if total_chars > 0 else 0
    return 

def remove_repeated_words(text):
    """
    Remove consecutive repeated words from the text.
    """
    if not isinstance(text, str) or text is None:  # Ensure the input is a valid string
        return text  # Return as is if not a string or None

    words = text.split()  # Split the string into words
    result = [words[0]] if words else []  # Initialize the result with the first word if exists

    for word in words[1:]:
        if word != result[-1]:  # Only add word if it's not a duplicate of the last one
            result.append(word)

    return ' '.join(result)  # Join the words back into a string


def remove_special_chars(text, threshold=0.3):
    """Remove texts with a high ratio of special characters."""
    if not isinstance(text, str):  # Handle None or non-string values
        return text  # Return as-is if not a string

    special_chars = sum(not char.isalnum() for char in text)
    total_chars = len(text)
    special_ratio = special_chars / total_chars if total_chars > 0 else 0
    return text if special_ratio < threshold else None


def remove_short_chunks(text, min_words=3):
    """Remove texts with insufficient word count."""
    return text if len(text.split()) >= min_words else None

def deduplicate_sentences(df, text_column):
    """Deduplicate rows based on text content."""
    df[text_column] = df[text_column].str.strip().str.lower()
    return df.drop_duplicates(subset=[text_column])

def preprocess_text(series):
    """Apply all preprocessing steps to the text column."""
  
    series = series.dropna()

    
    series = series.apply(lambda x: remove_repeated_chars(x) if isinstance(x, str) else x)
    series = series.apply(lambda x: remove_repeated_words(x) if isinstance(x, str) else x)
    series = series.apply(lambda x: remove_special_chars(x) if isinstance(x, str) else x)
    series = series.apply(lambda x: remove_short_chunks(x) if isinstance(x, str) else x)


    series = series.dropna()


    series = series.drop_duplicates()

    return series
