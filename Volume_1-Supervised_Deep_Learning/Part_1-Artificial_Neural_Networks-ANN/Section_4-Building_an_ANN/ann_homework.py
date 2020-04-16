# answer for the home work
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# script_dir = os.path.dirname(__file__)
# abs_file_path = os.path.join(script_dir, 'Churn_Modelling.csv')

# Importing the dataset
# Last row contains elements that needs to be predicted
dataset = pd.read_csv(
    '/home/mario/PycharmProjects/deep_learning_A-Z/Volume_1-Supervised_Deep_Learning/Part_1-Artificial_Neural_Networks-ANN/Section_4-Building_an_ANN/hw_Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
X = X[:, 1:]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
X_last = X[-1, :]  # Take the row that needs to be predicted
X = X[:-1, :]  # Remove predictable row
y = y[:-1]  # Remove predictable row

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.1)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting  new point
y_new_pred = classifier.predict(np.array([X_last]))
y_new_pred = (y_new_pred > 0.5)
print(y_new_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Accuracy
total_correct_pred = (1550 + 126)
total_test = 2000
print('Accuracy', total_correct_pred / total_test)
backend.clear_session()
