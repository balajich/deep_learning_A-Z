import numpy
from keras.layers import Dense
from keras.models import Sequential

# load pima indians dataset
dataset = numpy.loadtxt(
    "/home/mario/PycharmProjects/deep_learning_A-Z/frist_neural_network_with_keras/data/pima-indians-diabetes.csv",
    delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size=10, epochs=150)

scores = model.evaluate(X, Y)