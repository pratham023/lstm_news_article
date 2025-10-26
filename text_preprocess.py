import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.corpus.__dir__() else set()
lemmatizer = WordNetLemmatizer()

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+|www\S+', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = s.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Apply cleaning (may be slow on large datasets)
df['clean_text'] = df['text'].map(clean_text)
df['text_len'] = df['clean_text'].map(lambda x: len(x.split()))

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

X = df['clean_text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
print('Train:', len(X_train), 'Test:', len(X_test))