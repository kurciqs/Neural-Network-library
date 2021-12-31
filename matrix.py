import random
import math
import sys
import inspect
import copy

class Matrix:
  def __init__(self, rows, cols):
    #Error prevention
    if type(rows) != int or type(cols) != int:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Columns and rows have to be integers. \nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
    if rows == 0 or cols == 0:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Columns and rows have to be greater than 0. \nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

    self.rows = rows
    self.cols = cols
    self.value = [] 
    size = [self.rows, self.cols]
    self.size = tuple(size)
    for i in range(self.rows):
      row = []
      for j in range(self.cols):
        row.append(0)     
      self.value.append(row)
  
  @staticmethod
  def fromArray(arr):
    m = Matrix(len(arr), 1)
    for i in range(m.rows):
      m.value[i][0] = arr[i]

    return m

  def toArray(self):
    arr = []
    for i in range(self.rows):
      for j in range(self.cols):
        arr.append(self.value[i][j])
    
    return arr

  def randomize(self, n, m):
    for i in range(self.rows):
      for j in range(self.cols):
        self.value[i][j] += round(random.uniform(-1, 1) * n, m)
    
    return self
  
  def __str__(self):
    dims = [self.rows, self.cols] 
    dims = tuple(dims) 
    out = "Dimensions: " + str(dims) + "\n"
    for i in range(self.rows):
      row = [] 
      for j in range(self.cols):
        row.append(self.value[i][j])
      out += (str(row) + "\n").replace(",", " ").replace("]", " |").replace("[", "| ")
    return str(out)

  @staticmethod
  def subMat(a, b):
    if a.size != b.size:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Matrix dimensions must be the same when subtracting!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
    else:
      A = copy.deepcopy(a)
      B = copy.deepcopy(b)
      A.sub(B)
      return A
  
  def sub(self, other):
    if (type(other) == Matrix):
      if self.size != other.size:
        callerLine = str(inspect.stack()[1][0])
        sys.exit("Matrix dimensions must be the same when subtracting!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
      
      else:
        for i in range(self.rows):
          for j in range(self.cols):
            self.value[i][j] -=  other.value[i][j]

      return self

    elif type(other) == float or type(other) == int:
      for i in range(self.rows):
        for j in range(self.cols):
          self.value[i][j] -= other
          
      return self

    else:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Argument has to be a scalar or a matrix.\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
  
  @staticmethod
  def addMat(a, b):
    if a.size != b.size:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Matrix dimensions must be the same when adding!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
    else:
      A = copy.deepcopy(a)
      B = copy.deepcopy(b)
      A.add(B)
      return A
  
  def add(self, other):
    if (type(other) == Matrix):
      if self.size != other.size:
        callerLine = str(inspect.stack()[1][0])
        sys.exit("Matrix dimensions must be the same when adding!\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
      
      else:
        for i in range(self.rows):
          for j in range(self.cols):
            self.value[i][j] +=  other.value[i][j]

      return self

    elif type(other) == float or type(other) == int:
      for i in range(self.rows):
        for j in range(self.cols):
          self.value[i][j] += other
          
      return self

    else:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Argument has to be a scalar or a matrix.\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")


  @staticmethod
  def multMat(m1, m2):
    if type(m1) == Matrix and type(m2) == Matrix:
      if(m1.cols != m2.rows):
        callerLine = str(inspect.stack()[1][0])
        sys.exit("Number of columns must match rows! Matrix multiplication error.\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

      else:
        result = Matrix(m1.rows, m2.cols)
        A = m1.value
        B = m2.value
        for i in range(result.rows):
          for j in range(result.cols):
            rowSum = 0
            for k in range(len(A[0])):
              rowSum += A[i][k] * B[k][j]

            result.value[i][j] = rowSum

        return result

    else:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Both arguments have to be a matrix. Argument incompatability.\nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")
    

  def mult(self, other):
    if type(other) == float or type(other) == int:
      for i in range(self.rows):
        for j in range(self.cols):
          self.value[i][j] *= other
      
      return self

    if type(other) == Matrix:
      if other.size != self.size:
        callerLine = str(inspect.stack()[1][0])
        sys.exit("To perform elemetwise matrix multiplication, both matrices must be the same size. For the matrix product use Matrix.multMat(). \nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

      else:
        for i in range(self.rows):
          for j in range(self.cols):
            self.value[i][j] *= other.value[i][j]

        return self

    else:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("The argument has to be a scalar or a matrix. multiply() either performs a scalar operation on the whole matrix or performs elementwise matrix multiplication, depending on the arguments. Argument incompatability. \nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

  @staticmethod  
  def trans(A):
    if type(A) != Matrix:
      callerLine = str(inspect.stack()[1][0])
      sys.exit("Argument has to be a matrix. Argument incompatability. \nError in: " + callerLine + " --> " + __file__ + " " + str(inspect.currentframe().f_lineno) + "\n")

    result = Matrix(A.cols, A.rows)

    for i in range(A.cols):
      for j in range(A.rows):
        result.value[i][j] = A.value[j][i]

    return result

  def transpose(self):
    result = Matrix(self.cols, self.rows)
    
    for i in range(self.cols):
      for j in range(self.rows):
        result.value[i][j] = self.value[j][i]
    
    self.size = result.size
    self.rows = result.rows
    self.cols = result.cols
    self.value = result.value
    

  def map(self, fn):
    #Apply a function to every element of the matrix values
    for i in range(self.rows):
      for j in range(self.cols):
        val = self.value[i][j]
        self.value[i][j] = fn(val)

  @staticmethod
  def mapToMat(mat, fn):
    result = Matrix(mat.rows, mat.cols)
    for i in range(result.rows):
      for j in range(result.cols):
        val = mat.value[i][j]
        result.value[i][j] = fn(val)
    
    return result