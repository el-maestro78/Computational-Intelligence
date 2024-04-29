from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout

df = pd.read_csv("iphi2802.csv", delimiter="\t", encoding='utf-8')
pattern = r'[.,\d\[\]-]'

# Clean the text data
for i in range(len(df)):
    sentence = (df['text'].iloc[i]).lower().split()
    for index, word in enumerate(sentence):
        clean_word = re.sub(pattern, '', word.strip())
        sentence[index] = clean_word
    clean_sent = ' '.join(sentence)
    df.loc[i, 'text'] = clean_sent


kwargs = {
    'ngram_range': (1, 1),
    'dtype': 'int',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': 'word',
    'min_df': 2,
    'max_df': 0.1,
    'max_features': 1000,
}
vectorizer = TfidfVectorizer(**kwargs)
X_tfidf = vectorizer.fit_transform(df['text'])
arr = X_tfidf.toarray()
tfidf_df = pd.DataFrame(X_tfidf.toarray())
df['Text_IDF'] = tfidf_df.values.tolist()
vocabulary = vectorizer.vocabulary_

warnings.simplefilter(action='ignore', category=FutureWarning)
df['text_num'] = pd.Series
for index, sentence in enumerate(df['Text_IDF']):
    df.at[index, 'text_num'] = sum(df['Text_IDF'].iloc[index])

df[['text_num', 'date_min', 'date_max']].hist(figsize=(10, 6), bins=20)
plt.tight_layout()
plt.show()

# Assuming df is your DataFrame
x_values = df['date_min']
y_values = df['id']
x_values_2 = df['date_max']
y_values_2 = df['id']

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x_values, y_values)
axes[0].set_xlabel('Date Min')
axes[0].set_ylabel('Id')
axes[0].set_title('Scatter Plot of Date Min')

axes[1].scatter(x_values_2, y_values_2)
axes[1].set_xlabel('Date Max')
axes[1].set_ylabel('Id')
axes[1].set_title('Scatter Plot of Date Max')
plt.show()


X_text_idf = X_tfidf
scaler = MinMaxScaler()
X_numeric = scaler.fit_transform(df[['region_main_id', 'region_sub_id', 'date_min', 'date_max']])
norm_dataset = pd.concat([df['text_num'], pd.DataFrame(X_numeric,
                                                       columns=['region_main_id', 'region_sub_id',
                                                                'date_min', 'date_max'])], axis=1)


# #df[['Text_IDF','region_main_id', 'region_sub_id', 'date_min','date_max']]
X = norm_dataset.values
zeros_array = np.zeros(len(df))
# print(X.shape)
y = zeros_array
# print(y.shape)
kfold = StratifiedKFold(n_splits=5, shuffle=True)

for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
    # X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]


#%%


@tf.function
def find_closest(true, pred):
    # date_min = norm_dataset[:, 3]  # Assuming the date column is at index 3
    row = norm_dataset.loc[true]
    date_min = norm_dataset[row, 3]
    date_max = norm_dataset[row, 4]  # Assuming the date column is at index 4

    min_diff = tf.abs(date_min - pred)
    max_diff = tf.abs(date_max - pred)

    closest = tf.where(min_diff > max_diff, date_max, date_min)

    return closest


@tf.function
def crmse(y_true, y_pred):
    #closest = find_closest(y_true, y_pred)
    #return K.sqrt(K.mean(K.square(y_pred - closest)))
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))



rmseList = []

X = X.astype('float64')
y = y.astype('float64')

for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
    # X[train_index][0][0] = np.array(X[train_index][0][0])
    # X[val_index] =  np.array(X[val_index][0][0])
    '''for idx in train_index:
        X[idx, train_index] = np.array(X[train_index, idx][0])
    for idx in val_index:
        X[idx] = np.array(X[idx])'''
    # X_train = np.array([X[i] for i in train_index])
    # X_val = np.array([X[i] for i in val_index])

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # X_train = X_train.astype('float64')
    # X_val = X_val.astype('float64')

    model = Sequential()
    model.add(Dense(10, activation="relu", input_dim=5))

    optimizer = SGD(learning_rate=0.01, momentum=0.2, decay=0.0, nesterov=False)
    # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[crmse])
    model.compile(loss='rmse', optimizer='sgd', metrics=['mse'])
    # Fit model
    model.fit(X_train, y_train)  #  epochs=500, batch_size=500,

    # Evaluate model
    scores = model.evaluate(X_val, y_val, verbose=0)
    rmseList.append(scores[0])
    print("Fold :", fold, " RMSE:", scores[0])

print("RMSE: ", np.mean(rmseList))
