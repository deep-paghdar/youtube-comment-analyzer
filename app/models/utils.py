from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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