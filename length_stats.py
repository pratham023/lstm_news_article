import matplotlib.pyplot as plt

print('Category counts:')
print(df['category'].value_counts().head(20))

plt.figure(figsize=(10,4))
df['category'].value_counts().nlargest(20).plot(kind='bar')
plt.title('Top categories')
plt.show()

plt.figure(figsize=(8,4))
df['text_len'].hist(bins=50)
plt.title('Text length distribution (words)')
plt.show()