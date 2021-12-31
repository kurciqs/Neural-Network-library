import matrix as mat
Matrix = mat.Matrix
inspect = mat.inspect
sys = mat.sys
import pickle
import copy
     
     
def sigmoid(x):
  return 1 / (1 + mat.math.exp(-x))

def sigmoid_prime(y):
  #return sigmoid(x) * (1 - sigmoid(x))
  return y * (1 - y)

class NeuralNetwork:
  def __init__(self, dimensions, learning_rate):
    if 0 in dimensions:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("All the layer sizes have to be larger than 0!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")  
    if type(learning_rate) != float and type(learning_rate) != int:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("The learning rate has to be a number!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

    self.learning_rate = learning_rate
    self.weights = [] 
    self.dimensions = dimensions
    self.biases = []
    for x in range(len(dimensions)-1):
      W = Matrix(dimensions[x+1], dimensions[x])
      W.randomize(1, 20)
      self.weights.append(W)
    for x in range(len(dimensions)-1):
      B = Matrix(dimensions[x+1], 1)
      B.randomize(1, 10)
      self.biases.append(B)


  def __str__(self):
    dims = "Neural network hyperparameters: \n"
    for x in range(len(self.dimensions)-1):
      dims += str(self.dimensions[x]) + ", "
    dims += str(self.dimensions[len(self.dimensions)-1]) + "\n"
    weights = "The current weights are:\n\n" 
    for x in range(len(self.weights)):
      weights += "Layer " + str(x+1) + ":\n" + str(self.weights[x]) + "\n"  
    biases = "The current biases are:\n\n" 
    for x in range(len(self.biases)):
      biases += "Layer " + str(x+1) + ":\n" + str(self.biases[x]) + "\n"
      
    #print(self.errors)
    try: 
      errors = self.errors
      errorMsg = "These are the current errors:\n\n" 
      for error in errors:
        errorMsg += str(error) + "\n"
    except:   
      errorMsg = "\n" 
    
    return dims + "\n" + weights + "\n" + biases + "\n" + errorMsg

  def feedforward(self, input_array):
    if len(input_array) != self.weights[0].cols:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Input array has to be the same size as the layer 1 size!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

    inputs = Matrix.fromArray(input_array)
    
    layerValues = [inputs]
    for x in range(len(self.weights)):      
      prev = layerValues[x]
      current = Matrix.multMat(self.weights[x], prev)
      current.add(self.biases[x])
      current.map(sigmoid)
      layerValues.append(current)

    output = layerValues[len(layerValues)-1]
    return output.toArray()


  def train(self, input_array, target_array):
    #Error prevention
    if len(target_array) != self.weights[len(self.weights)-1].rows:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Target array has to be the same size as the output layer size!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

    inputs = Matrix.fromArray(input_array)
    #initial outputs
    
    #fast feedforward algorithm + all the layer values stored in an array
    layerValues = [inputs]
    for x in range(len(self.weights)):      
      prev = layerValues[x]
      current = Matrix.multMat(self.weights[x], prev)
      current.add(self.biases[x])
      current.map(sigmoid)
      layerValues.append(current)

    outputs = layerValues[len(layerValues)-1]
    
    #error
    targets = Matrix.fromArray(target_array)
    output_errors = Matrix.subMat(targets, outputs)
    errors = [output_errors]

    #grads
    gradients = Matrix.mapToMat(outputs, sigmoid_prime)
    gradients.mult(errors[0])
    
    gradients.mult(self.learning_rate)
    
    #deltas
    last_hidden = layerValues[len(layerValues)-2]
    last_hidden_T = Matrix.trans(last_hidden)
    weight_ho_deltas = Matrix.multMat(gradients, last_hidden_T)
    
    #appliance to the last biases and last weights
    self.weights[len(self.weights)-1].add(weight_ho_deltas)
    self.biases[len(self.biases)-1].add(gradients)
    
    
    i = len(self.weights)-1
    for j in range(len(self.weights)-1):
      #current weight and errors
      who_t = Matrix.trans(self.weights[i])
      hidden_errors = Matrix.multMat(who_t, errors[j])
      errors.append(hidden_errors)

      #grads
      hidden_gradient = Matrix.mapToMat(layerValues[i], sigmoid_prime)
      hidden_gradient.mult(errors[j+1])
      hidden_gradient.mult(self.learning_rate)

      #calculated changes
      inputs_T = Matrix.trans(layerValues[i-1])
      weight_ih_deltas = Matrix.multMat(hidden_gradient, inputs_T)
      
      #Appliance
      self.weights[i-1].add(weight_ih_deltas)
      self.biases[i-1].add(hidden_gradient)
      i -= 1
      
    self.errors = errors 

    #return the last layer error for now
    return errors[0]

  def save(self, filename):
    if type(filename) != str:
      print("Filename must be a string.")
      return 0
    with open(filename + ".nn", "wb") as f:
      pickle.dump(self, f)

  @staticmethod
  def load(filename):
    try:
      with open(filename, "rb") as f:
        return pickle.load(f)
    except:
      callerLine = str(inspect.stack()[1][0])
      print("File was not found or something else broke lol!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
      return 0


  @staticmethod
  def copy(NN):
    if type(NN) == NeuralNetwork:
      return copy.deepcopy(NN)
    else:
      callerLine = str(inspect.stack()[1][0])
      print("You can only copy a NeuralNetwork with this function!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
      return 0