import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from google.colab import drive

path = "/content/train_data.json"

data = []
try:
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

except Exception as e:
    print('Error loading file:', e)
    df = pd.DataFrame() 


print('Columns:', df.columns.tolist())

if 'headline' in df.columns and 'short_description' in df.columns:
    df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
else:
    df['text'] = ''

cols = ['category','text']
for c in ['headline','short_description','authors','link','date']:
    if c in df.columns:
        cols.append(c)

df = df[[c for c in cols if c in df.columns]]

if 'category' in df.columns:
    df['category'] = df['category'].astype(str)
else:
    df['category'] = ''


print('Loaded', len(df), 'rows')