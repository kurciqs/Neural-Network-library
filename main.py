import matrix as mat
import nn
import random
Matrix = mat.Matrix
NeuralNetwork = nn.NeuralNetwork

training_data = [
  [
    [0, 0],
    [0]
  ],
  [
    [1, 1],
    [0]
  ],
  [
    [0, 1],
    [1]
  ],
  [
    [1, 0],
    [1]
  ]
]

NN = NeuralNetwork([2, 2, 1], 0.5)

for i in range(50000):
  data = random.choice(training_data)
  NN.train(data[0], data[1])

print(NN.feedforward([0, 0]))
print(NN.feedforward([1, 1]))
print(NN.feedforward([1, 0]))
print(NN.feedforward([0, 1]))