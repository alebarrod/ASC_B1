import numpy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

#Static
PI = 3.14159265358979
_ROUND_ = 6
_PROB_CROSSOVER_ = 0.2
_PROB_MUTATE_ = 0.1

N = 50   #Numero de individuos
G = 800  #Numero de iteraciones
T = int(N*0.3)   #Numero de vecinos
if T == 0: T = 1

Z = [1.0, 1.0] #Valor de referencia
#////////////////////////////////////////////////////////////////////////////////
#Valores de ZDT3
nreal = 30
nbin = 0
ncon = 0
nobj = 2
p = 8  #Numero de vectores peso y dimension

class Subproblema:
    def __init__(self, peso, vecinos):
        self.peso = peso
        self.vecinos = vecinos
        self.individuo = None
        self.fitness = 9999999999999
    
    def setSolucion(self, individuo, fitness):
        if fitness < self.fitness:
            print(self.fitness, " <-- ", self.individuo, " << ", individuo, " --> ", fitness)
            self.individuo = copy.deepcopy(individuo)
            self.fitness = fitness

def subproblemaPorPeso(vecindad, peso):

    for element in vecindad:
        if element.peso == peso:
            return element
    
    return None


def ZDT3(xreal):
    tmp = 0.0
    obj = [0.0,0.0]
    obj[0] = xreal[0]
    for i in range(1,p):
        tmp += xreal[i]
    g = 1 + ((9 * tmp)/(p-1))
    h = 1 - math.sqrt(xreal[0]/g) - (xreal[0]/g) * math.sin(10*PI*xreal[0])
    obj[1] = g * h
    return obj
#////////////////////////////////////////////////////////////////////////////////

def gte(peso, f1, f2):
    return max(peso[0] * abs(f1 - Z[0]), peso[1] * abs(f2 - Z[1]))
    

def checkZ(f1, f2):

    if f1 < Z[0]:
        Z[0] = f1
    if f2 < Z[1]:
        Z[1] = f2


def inializacionPoblacion():
    individuos = list()

    for i in range(0,N):
        aux = list()

        for j in range(0, p):
            aux.append(round(random.random(), _ROUND_))
        individuos.append(aux)
    
    return individuos

#Generacion de vectores repartidos uniformemente
def inializacionPesos():
    vectores = list()

    for i in numpy.arange(0,1,1/N):
        x = round(i, _ROUND_)
        y = round(1 - x, _ROUND_)
        vectores.append(tuple((x,y)))
    return vectores

def distanciaEuclidea(vector1, vector2):
    return numpy.sqrt((vector2[0] - vector1[0])**2 + (vector2[1] - vector1[1])**2)


def test(listaPesos, origen):
    new = list()

    for elemento in listaPesos:
        distancia = distanciaEuclidea(origen, elemento)
        tupla = (elemento, distancia)
        new.append(tupla)
    
    return copy.deepcopy(new)

def vecinosIterativo(listaPesos, individuo):
    res = list()
    pesos = list()

    pesos = test(listaPesos, individuo)
    cantidad = 0

    while cantidad < T:
        minimo = 999999999
        minimoVecino = tuple()

        for vecino in pesos:
            if vecino[1] < minimo:
                minimo = vecino[1]
                minimoVecino = copy.deepcopy(vecino[0])
                pesos.remove((minimoVecino, minimo))
                res.append(vecino[0])
                cantidad = cantidad + 1
    
    return res
        
        
def inicializacionVecinos(listaVectores):
    mapaDistancias = list()

    for j,vector in enumerate(listaVectores):
        listaDistancias = list()
        for i,element in enumerate(listaVectores):

            if i < len(listaVectores):
                listaDistancias.append(listaVectores[i])
        mapaDistancias.append(listaDistancias)
    return mapaDistancias

#crossover
def crossoverInd(individuo1, individuo2):
    individuo1res = list()
    individuo2res = list()

    crossover = False
    for i in range(0, p):
        if random.random() < _PROB_CROSSOVER_:
            crossover = not crossover
        
        if crossover:
            individuo1res.append(individuo2[i])
            individuo2res.append(individuo1[i])
        else:
            individuo1res.append(individuo1[i])
            individuo2res.append(individuo2[i])
    
    return individuo1res, individuo2res
            
def crossoverPop(population):
    newPop = list()

    longitud = len(population)

    for i in range(0, longitud, 2):
        individuo1, individuo2 = crossoverInd(population[i], population[i + 1])
        newPop.append(individuo1)
        newPop.append(individuo2)
    
    if longitud % 2 == 1:
        newPop.append(population[-1])
    
    return newPop

#mutaciones
def mutatateInd(individuo):
    newInd = list()
    for gen in individuo:
        if random.random() < _PROB_MUTATE_:
            newInd.append(round(random.random(), _ROUND_))
        else:
            newInd.append(gen)
    
    return newInd

def mutatatePop(population):
    newPop = list()

    for individuo in population:
        newPop.append(mutatateInd(individuo))
    
    return newPop

def evolucion(population):
    pop = mutatatePop(population)
    random.shuffle(pop)
    pop = crossoverPop(pop)
    pop = mutatatePop(pop)

    return pop

def plotGraph(subproblemas):
    x = list()
    y = list()
    x1 = list()
    y1 = list()

    file = open("PF_ZDT3.dat", "r")
    for line in file:
        s,t = line.split("\t")
        x1.append(float(s))
        y1.append(float(t.replace("\n","")))
    colors1 = ("red")

    for subproblema in subproblemas:
        f1,f2 = ZDT3(subproblema.individuo)
        x.append(f1)
        y.append(f2)
    colors = ("blue")

    # Plot
    plt.scatter(x1, y1, c=colors1)
    plt.scatter(x, y, c=colors)
    plt.title('ZDT3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    listaSubproblemas = list()
    listaVectores = inializacionPesos()

    poblacion = inializacionPoblacion()

    for peso in listaVectores:
        listaVecinos = vecinosIterativo(listaVectores,peso)
        listaSubproblemas.append(Subproblema(peso, listaVecinos))

    for i,superproblema in enumerate(listaSubproblemas):
        superproblema.individuo = poblacion[i]
    i = 0
    #plotGraph(listaSubproblemas)

    for subproblema in listaSubproblemas:
        i = i + 1
        print("N: ", i)
        for individuo in poblacion: #Cuando acabemos 
            for repeticion in range(0, G):
                sample = random.sample(subproblema.vecinos, k = 3)  #3 elementos aleatorios para el crossover

                individuosEvolucion = list()
                individuosEvolucion.append(individuo)

                for peso in sample: #Obtener el individuo que tiene casa subproblema 
                    objetivo = subproblemaPorPeso(listaSubproblemas, peso).individuo
                    individuosEvolucion.append(copy.deepcopy(objetivo))

                
                resultadoEvolucion = evolucion(individuosEvolucion)
                hijo = random.choice(resultadoEvolucion)

                f1,f2 = ZDT3(hijo)

                checkZ(f1, f2)

                for vecino in subproblema.vecinos:
                    subproblemaVecino = subproblemaPorPeso(listaSubproblemas, vecino)
                    fitness = gte(subproblemaVecino.peso, f1, f2)
                    subproblemaVecino.setSolucion(hijo, fitness)
                #Mutar individuo con 3 individuos de los vecinos --> obtenemos un hijo
                #Comprobamos el hijo con los T vecinos
           
    plotGraph(listaSubproblemas)


###########################################

main()
