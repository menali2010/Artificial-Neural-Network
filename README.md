# Artificial-Neural-Network
Method for generating time series of daily solar irradiance from one year of station data (temperature, humidity) combined with values produced by a satellite model.

#Artificial Neural Networks (ANNs) are systems composed of neurons that solve mathematical functions (both linear and nonlinear). ANNs are a tool capable of storing knowledge from training (examples) and can be used in various areas, such as prediction models, pattern recognition, and other human applications. 
#To develop an ANN, it is necessary to execute code functions that train the network. In this stage, examples are provided to the network, which initially adjusts its synaptic weights and biases randomly and gradually refines these values through various functions until it extracts the best combinations to represent the data. Subsequently, these values are fixed and used to generate solutions for new input data.
#The objective of this code is to develop a method for generating time series of daily solar irradiance from one year of field data acquisition combined with values produced by the BRASIL-SR model and used in the Brazilian Solar Energy Atlas.
#To start creating the code, it is necessary to process the data obtained from the stations and the satellite, normalizing them to the range between 0 and 1. It is also necessary to group them into three files to import this information into the algorithm. The training file (Pet2014train) should contain 66.6% of the data corresponding to one year, with approximately 20 data points for each month. The second file is the testing file (Pet2014test), which contains the remaining 33.3% of data from the year used for training. The third file is the prediction file (Pet2010prediction), which contains another full year of data.
#Initially, the data is grouped into three distinct spreadsheets used for training, testing, and prediction. Due to the storage format of the values (Excel), the pandas tool was used for importing. After the import, each set of values was saved in different data frames for later application in matrices. The values inserted into X are the input data of the system, which include satellite-estimated irradiation, surface temperature, and relative humidity. The data inserted into y are the output values, the expected result, which is the measured irradiation at the station. The last variable, xPredicted, represents the data that will be used for prediction, having the same properties as the variable X. However, the code should predict the output value these values will provide based on the training and testing steps using the variables X_train/X_test and y_train/y_test.


@author: Menali

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


xlsx60=pd.ExcelFile("Pet2014train.xlsx")
df=pd.read_excel(xlsx60, "Planilha1")
m=df.shape[0]
df1=df.radext
df2=df.ghiext
df3=df.temp
df7=df.umidade

xlsx40=pd.ExcelFile("Pet2014test.xlsx")
df4=pd.read_excel(xlsx40, "Planilha1")
df5=df4.ghiext
df6=df4.temp
df8=df4.umidade
df13=df4.radext

xlsx70=pd.ExcelFile("Pet2010predição.xlsx")
df9=pd.read_excel(xlsx70, "Planilha1")
df10=df9.ghiext
df11=df9.temp
df12=df9.umidade
df14=df9.radext

#X=irradiaçãoSatélite//temperatura//umidade

#y=irradiaçãoEstação

X_train = np.array(([df2,df3,df7]), dtype=float)
X_train=X_train.transpose()

X_test = np.array(([df5,df6,df8]), dtype=float)
X_test = X_test.transpose()

X_predict = np.array(([df10,df11,df12]), dtype=float)
X_predict = X_predict.transpose()

y_train = np.array(([df1]), dtype=float)   
y_train = y_train.transpose()

y_test = np.array(([df13]), dtype=float)
y_test = y_test.transpose()

y_predict = np.array(([df14]), dtype=float)
y_predict = y_predict.transpose()

#________________________________________________________________________________________

#After creating the variables, we can proceed with the development of the Artificial Neural Network (ANN) code. An ANN is essentially a class filled with functions that make it behave like an actual neural network. Therefore, to start the code, it is necessary to create a class (Neural_Network) to encapsulate the entire code within it, defining the properties and values used. Subsequently, the initialization function (init), also known as the constructor, should be defined. It generates the initial values of the algorithm. Within the initialization method, the weights and biases need to be defined. They are initialized randomly and adjusted later.
#Since it involves matrix multiplication, the weights and biases must be randomly generated in the appropriate size. The initial weights (W1) that multiply the input layer should have the number of rows equal to the number of variables used in X (predictors), and the number of columns should be determined through testing. The best results are often achieved with a value of 4. The final weights (W2) have the number of rows equal to 4 and the number of columns equal to the number of output variables, which is usually 1.
#The bias, as it represents values added after the multiplication and summation of weights with input data, initially contains (B1) a shape with the number of rows equal to the number of output variables and the number of columns equal to 4. Subsequently, (B2) contains the number of rows and columns equal to the number of output variables.
#The learning rate (learning_rate) is defined empirically by testing which value works best for the neural network.

n_predictors = X_train.shape[1]
n_hiddenneurons = 4
n_outputs = 1
learning_rate = 0.05

class Neural_Network(object):   
    def __init__(self):            
    
        self.inputSize =  n_predictors
        self.outputSize = n_outputs
        self.hiddenSize = n_hiddenneurons

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
        self.B1 = np.random.randn(self.outputSize, self.hiddenSize)
        self.B2 = np.random.randn(self.outputSize, self.outputSize)
       
#________________________________________________________________________________________

#The next step is to define the signal propagation, which involves propagating the signal from the input layer of the neural network towards the output layer. In this step, matrix multiplication is performed using the previously developed weights. The input values are multiplied by the initial weights (W1), and the resulting matrix is then added to the initial biases (B1). Subsequently, the resulting matrix is passed through a sigmoid function (which will be defined shortly), resulting in the hidden layer of the ANN. The values obtained in this layer are multiplied by the final weights (W2), and the resulting matrix is added to the final bias (B2). The result obtained in this last step is passed through the sigmoid function, giving rise to the values of the output variable.

    def forward(self, X):
        self.x = np.dot(X, self.W1) + self.B1     
        self.x2 = self.sigmoid(self.x)
        self.x3 = np.dot(self.x2, self.W2) + self.B2   
        o_rna = self.sigmoid(self.x3)
        return o_rna
        
#________________________________________________________________________________________

#The next step is to define the activation function (previously used) and also the derivative of the activation function (sigmoidPrime), which is used to adjust the weights and biases, consequently reducing the error.

    def sigmoid(self, s):
    #activation function
        return 1/(1+np.exp(-s))
  
    def sigmoidPrime(self, s):
    #derivative of the sigmoid
        return s * (1 - s)
        
#________________________________________________________________________________________

#After defining the forward propagation function, the next step is to define the backward propagation function, also known as backpropagation. This method is responsible for propagating the signal from the output (o_train) back to the input. It incrementally adjusts the weights and biases, leading to improved output values.
#First, we calculate the sample difference between the expected output value (y_train) and the output value obtained from the code (o_train). Then, we apply the derivative of the sigmoid function to the output value and multiply it by the previous step's result (o_error). This result corresponds to the internal activity level of the neuron between the hidden layer and the output layer.
#To determine how much the final weights (W2) contributed to the error, we perform a matrix multiplication between the previous step's result (o_delta) and the transpose of the old W2. The result (x2_error) is then multiplied by the values of the hidden layer (x2) that have been passed through the derivative of the activation function. This result corresponds to the internal activity level of the neuron between the input layer and the hidden layer.
#To adjust W1, we add the previous initial weight with the matrix multiplication of the transposed input variables and the result from the previous step (x2_delta), multiplied by a predefined learning rate of the neural network. The adjustment of W2 is done similarly, using the previous final weights and adding the matrix multiplication of the hidden layer (x2) with o_delta, multiplied by the learning rate.
#To adjust the initial bias (B1), we sum the results obtained from the column-wise summation of x2_delta, multiplied by the learning rate, and add the initial biases (B1). Lastly, we adjust the final bias (B2) in a similar manner as B1, summing the results obtained from the column-wise summation of the o_delta, multiplied by the learning rate, and adding the final bias.
#The learning rate is crucial for gradually and slowly minimizing the error, aiming to achieve optimal performance.

    def backward(self, X_train, y_train, o_train):
        self.o_error = y_train - o_train # erro na saida
        self.o_delta = self.o_error*self.sigmoidPrime(o_train) 

        self.x2_error = self.o_delta.dot(self.W2.T) 
        self.x2_delta = self.x2_error*self.sigmoidPrime(self.x2) 

        self.W1 += X_train.T.dot(self.x2_delta)*learning_rate #ajustando o weight1
        self.W2 += self.x2.T.dot(self.o_delta)*learning_rate #ajustando weight2
        self.B1 += np.sum(self.x2_delta, axis=0)*learning_rate #ajustando bias1
        self.B2 += np.sum(self.o_delta, axis=0)*learning_rate #ajustando bias2

#________________________________________________________________________________________

#After defining the propagation and backpropagation algorithms, the training function (train) is defined. In this function, all the parameters are applied to train the neural network. The step of saving the weight and bias values is crucial because, after obtaining the best weights during training, they can be saved and used for testing and prediction purposes.

    def train(self, X_train, y_train):
        o_train = self.forward(X_train)
        self.backward(X_train, y_train, o_train)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
        np.savetxt("b1.txt", self.B1, fmt="%s")
        np.savetxt("b2.txt", self.B2, fmt="%s")

#________________________________________________________________________________________

#The next step is to define some stopping criteria for training the neural network. A neural network is designed to achieve its best performance, and therefore it is necessary to define a stopping algorithm that recognizes the best error at each epoch and determines if it is feasible to continue training. The algorithm compares the errors obtained during the testing phase, and if the current error is lower than the previous one, the network continues training.
#If the last obtained error value is greater than the previous one, the network receives a stop signal. At this point, the best set of weights and biases is stored and applied to the variable X_predict, resulting in the output values.

NN = Neural_Network()
erro_min = 9999
status = True
count = 0   #Count the number of training epochs.
errors_train = []
errors_test = []

#Training and defining its error.
for i in range (1000):
    NN.train(X_train, y_train)
    erro_train = np.mean(np.square(y_train - NN.forward(X_train)))
print ("Erro de treinamento: ", erro_train)
   
while status:
        
    for i in range (1000):
        NN.train(X_train, y_train)
        erro_train = np.mean(np.square(y_train - NN.forward(X_train)))
    print ("Erro de treinamento: ", erro_train)
    errors_train.append(erro_train)
    
    erro = np.mean(np.square(y_test - NN.forward(X_test)))
    errors_test.append(erro)
    
    if erro<erro_min:
        erro_min = erro
        print ("________________________", status)
        print ("Épocas de treinamento: ", count)
        print ("Erro do teste: ", erro)
        print ("________")
        
    else:
        status = False
        NN.saveWeights()   #Store the best weights.
        print ("________________________", status)
        print ("Épocas de treinamento: ", count)
        print ("Erro do teste: ", erro)
        print ()
        print ("______")
        
    #Check if the error varies very little during training.        
    if ((count > 1) and (abs(errors_train[count-1]-erro_train) < 1.0e-06)):
        status = False
        print("very low gradient")
        NN.saveWeights()  
        
    if count == 2000:
        status = False
        NN.saveWeights()
    print ("Erro do Teste: ", erro)
    print ("______")
    print ()
    count += 1    
 
       
y_prev = NN.forward(X_predict)
erro_previsão = np.mean(np.square(y_predict - NN.forward(X_predict)))
print ("Erro de previsão: ", erro_previsão)

#________________________________________________________________________________________

#During the normalization of solar irradiation values, they were divided by their corresponding daily extraterrestrial irradiation value (as this is the maximum value that could reach the Earth's surface). Thus, after obtaining the output values, they are extracted to a new spreadsheet manually and multiplied by their corresponding daily extraterrestrial irradiation values to obtain the actual final solar irradiation values.
#For a better analysis of the neural network's performance, graphs are used to measure the errors obtained during each epoch of training. The first set of graphs examines the mean squared error and its decrease during training, both for the training phase and the testing phase.

figu=plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.ylabel('MSE')
plt.xlabel('Periods')
plt.title("Mean Squared Error Train vs. Periods")
plt.tight_layout()
plt.plot(errors_train)

plt.subplot(1, 2, 2)
plt.ylabel('MSE')
plt.xlabel('Periods')
plt.title("Mean Squared Error Test vs. Periods")
plt.tight_layout()
plt.plot(errors_test)
plt.show()
figu.savefig("erros.png", dpi=300)  # salva gráficos

#Output a scatter plot estimates versus observations.

fig=plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 3)
plt.ylabel('RNA Validation')
plt.xlabel('Observations')
plt.title("Scatter Plot")

plt.scatter(y_predict, y_prev, c="g")
zz=np.linspace(0,1,2) 
plt.plot(zz,zz,'k--') # identity line
plt.xlim(0,1)
plt.ylim(0,1)

plt.subplot(2, 3, 2)
plt.ylabel('RNA Test')
plt.xlabel('Observations')
plt.title("Scatter Plot")
plt.scatter(NN.forward(X_test), y_test, c="b")
plt.plot(zz,zz,'k--') # identity line
plt.xlim(0,1)
plt.ylim(0,1)

plt.subplot(2, 3, 1)
plt.ylabel('RNA Training')
plt.xlabel('Observations')
plt.title("Scatter Plot")
plt.scatter(NN.forward(X_train), y_train, c="r")
plt.plot(zz,zz,'k--') # identity line
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
fig.savefig("scatter.png", dpi=300) 
