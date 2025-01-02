from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Remove non-alphabetic tokens
    cleaned_tokens = [word for word in filtered_tokens if word.isalpha()]

    return " ".join(cleaned_tokens)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

def is_english_sentence(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
