import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
import pickle

nltk.download("stopwords")
pd.options.plotting.backend = "plotly"

# Загрузка датасетов
df1 = pd.read_csv('./input/datasets/Twitter_Data.csv')
df1.head()

df2 = pd.read_csv('./input/datasets/apple-twitter-sentiment-texts.csv')
df2 = df2.rename(columns={'text': 'clean_text', 'sentiment': 'category'})
df2['category'] = df2['category'].map({-1: -1.0, 0: 0.0, 1: 1.0})
df2.head()

df3 = pd.read_csv('./input/datasets/finalSentimentdata2.csv')
df3 = df3.rename(columns={'text': 'clean_text', 'sentiment': 'category'})
df3['category'] = df3['category'].map({'sad': -1.0, 'anger': -1.0, 'fear': -1.0, 'joy': 1.0})
df3 = df3.drop(['Unnamed: 0'], axis=1)
df3.head()

df4 = pd.read_csv('./input/datasets/Tweets.csv')
df4 = df4.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df4['category'] = df4['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df4 = df4[['category', 'clean_text']]
df4.head()

# Объединение
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

df.isnull().sum()

df.dropna(axis=0, inplace=True)

df.shape

df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

df.head()

# Data Preprocessing

def tweet_to_words(tweet):
    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words

print("\nOriginal tweet ->", df['clean_text'][0])
print("\nProcessed tweet ->", tweet_to_words(df['clean_text'][0]))

# Apply data processing to each tweet
print("\n Tweets preprocessing started..")
X = list(map(tweet_to_words, df['clean_text']))
print("\n Tweets preprocessing complete")

# Encode target labels
le = LabelEncoder()
Y = le.fit_transform(df['category'])

print(X[0])
print(Y[0])

# ### Train and test split
y = pd.get_dummies(df['category'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# ### Bag of words (BOW) feature extraction
vocabulary_size = 5000

count_vector = CountVectorizer(max_features=vocabulary_size,
                               preprocessor=lambda x: x,
                               tokenizer=lambda x: x)

# Fit the training data
X_train = count_vector.fit_transform(X_train).toarray()

# Transform testing data
X_test = count_vector.transform(X_test).toarray()

# print first 200 words/tokens
print(count_vector.get_feature_names()[0:200])

# ### Tokenizing & Padding
max_words = 5000
max_len = 50


def tokenize_pad_sequences(text):
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer


print('Before Tokenization & Padding \n', df['clean_text'][0])
X, tokenizer = tokenize_pad_sequences(df['clean_text'])
print('After Tokenization & Padding \n', X[0])

# ### Saving tokenized data
#
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train & Test Split
y = pd.get_dummies(df['category'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print('Train Set ->', X_train.shape, y_train.shape)
print('Validation Set ->', X_val.shape, y_val.shape)
print('Test Set ->', X_test.shape, y_test.shape)

def f1_score(precision, recall):
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# ## Bidirectional LSTM Using NN
vocab_size = 5000
embedding_size = 32
epochs = 20
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
# Build model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

print(model.summary())

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy', Precision(), Recall()])

# Train model
batch_size = 64
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1)


# Evaluate model on the test set
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
# Print metrics
print('')
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F1 Score  : {:.4f}'.format(f1_score(precision, recall)))

# Save the model architecture & the weights
model.save('best_model.h5')
print('Best model saved')