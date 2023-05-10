# -*- coding: utf-8 -*-

"""

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


n_predictors = X_train.shape[1]

n_hiddenneurons = 4

n_outputs = 1

learning_rate = 0.05


class Neural_Network(object):   
    def __init__(self):            #ao utilizar self, podemos utilizar os atributos da classe
    
        self.inputSize =  n_predictors
        self.outputSize = n_outputs
        self.hiddenSize = n_hiddenneurons

    #weights, valores entre 0 e 1
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
        self.B1 = np.random.randn(self.outputSize, self.hiddenSize)
        self.B2 = np.random.randn(self.outputSize, self.outputSize)
    
    def forward(self, X):
    #forward propagation através da rede
        self.x = np.dot(X, self.W1) + self.B1     
        self.x2 = self.sigmoid(self.x)
        self.x3 = np.dot(self.x2, self.W2) + self.B2   
        o_rna = self.sigmoid(self.x3)
        return o_rna
    
    def sigmoid(self, s):
    # função de ativação
        return 1/(1+np.exp(-s))
  
    def sigmoidPrime(self, s):
    #derivada da sigmoide
        return s * (1 - s)

    def backward(self, X_train, y_train, o_train):
    # backward propagation atraves da rede
        self.o_error = y_train - o_train # erro na saida
        self.o_delta = self.o_error*self.sigmoidPrime(o_train) 

        self.x2_error = self.o_delta.dot(self.W2.T) 
        self.x2_delta = self.x2_error*self.sigmoidPrime(self.x2) 

        self.W1 += X_train.T.dot(self.x2_delta)*learning_rate #ajustando o peso1
        self.W2 += self.x2.T.dot(self.o_delta)*learning_rate #ajustando peso2
        self.B1 += np.sum(self.x2_delta, axis=0)*learning_rate #ajustando bias1
        self.B2 += np.sum(self.o_delta, axis=0)*learning_rate #ajustando bias2
    
    def train(self, X_train, y_train):
        o_train = self.forward(X_train)
        self.backward(X_train, y_train, o_train)

    def saveWeights(self): #salvar valores dos pesos em um documento txt
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
        np.savetxt("b1.txt", self.B1, fmt="%s")
        np.savetxt("b2.txt", self.B2, fmt="%s")

       
NN = Neural_Network()

erro_min = 9999

status = True

count = 0   #conta o número de épocas do treinamento

errors_train = []

errors_test = []

#treinamento e definição do erro do mesmo
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
        NN.saveWeights()   #armazena os melhores pesos
        print ("________________________", status)
        print ("Épocas de treinamento: ", count)
        print ("Erro do teste: ", erro)
        print ()
        print ("______")
        
    #Checa se o erro varia muito pouco no treinamento        
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
        
# Output a graph of loss metrics over periods.

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

# Output a scatter plot estimates versus observations.

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
fig.savefig("scatter.png", dpi=300) # salva gráficos
