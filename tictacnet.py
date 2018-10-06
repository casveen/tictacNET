import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
"""
# Cool ML here:
    
df=pd.read_csv('tictactoe-data.csv')
print(df.head())

#iloc: treat as numpy array
X=df.iloc[:, [list(range(18))+[-2]]]
print(X.head())

target=df.iloc[:, list(range(18,27))]

X_train, X_test, y_train, y_test=train_test_split(X,target, test_size=0.2)

#keras: wrapper, sequentail, can make layers on the go
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=X.shape[1]))
model.add(tf.keras,layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras,layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras,layers.Dropout(0.3))

#final layer, output=9, we want output as a probability, so we use softmax
model.add(tf.keras.layers.Dense(target.shape[1], activation='softmax'))

#the model migth be overfit, add dropout layer between two layers

#compile, build, the model
model.compile(omptimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_data=[X_test, y_test], )

print('accuracy_', model.evaluate(X_test, y_test))

model.save('tictacNET.h5')
"""