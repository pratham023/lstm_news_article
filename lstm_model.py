import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

MAX_VOCAB = 20000
MAX_LEN = 60
EMBED_DIM = 100
BATCH_SIZE = 128
EPOCHS = 10

#I have re-defined X_train, X_test, y_train, y_test & Label encoder so this code can run independently

df_train = pd.read_json("/content/train_data.json")
X_train = df_train["headline"].tolist()
y_train = df_train["category"].tolist()


le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_train_cat = to_categorical(y_train_enc, num_classes=len(le.classes_))

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')

with open("/content/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
mc = ModelCheckpoint('/content/lstm_best.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train_pad, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc]
)

model.save("/content/lstm_model.h5")


test_path = "/content/test_data_modified.json"
df_test = pd.read_json(test_path)
X_test = df_test["headline"].tolist()

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

y_pred_probs = model.predict(X_test_pad, batch_size=BATCH_SIZE, verbose=1)
y_pred_indices = np.argmax(y_pred_probs, axis=1)
y_pred_labels = le.inverse_transform(y_pred_indices)

with open("/content/lstm_predictions.txt", "w", encoding="utf-8") as f:
    for label in y_pred_labels:
        f.write(str(label) + "\n")