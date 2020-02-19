# Abstract Bat Class
import numpy as np
from numpy.random import seed
from numpy.random import rand
from Objective import Objective

# from copy import deepcopy


class Bat:

    def __init__(self, numOfFeature, featureRange,
                 f_min, f_max, w1, w2, w3, A_base,
                 alpha, r_base, gama, epsilon,
                 objectiveFunction):

        seed(int(rand(1)[0]*100))

        self.x = []
        for i in range(numOfFeature):
            self.x.append(featureRange[i][0] + ((featureRange[i][1] - featureRange[i][0]) * rand(1)[0]))
        self.x = np.array(self.x)
        self.x_new = self.x
        self.xBest = self.x
        self.x_history = []

        self.v = np.zeros(numOfFeature)
        self.v_new = np.zeros(numOfFeature)
        self.v_history = []

        self.A_base = A_base
        self.A = 0
        self.A_history = []

        self.r = r_base
        self.r_new = 0
        self.r_history = []

        self.fitnessValue = 999999999
        self.fitnessValue_new = 999999999
        self.fitnessValue_history = []

        self.objectiveValues = []
        self.objectiveValues_new = []
        self.objectiveValues_history = []

        self.generation = 0
        self.f_min = f_min
        self.f_max = f_max
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.alpha = alpha
        self.gama = gama
        self.epsilon = epsilon
        self.objectiveFunction = objectiveFunction

        self.neighbors = []

    def update_v_BA(self, p, vType=0, xType=0):
        X = 0
        V = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new
        if vType == 0:
            V = self.v
        else:
            V = self.v_new

        f = self.calc_f()
        Xdif = X - p
        self.v_new = V + (Xdif * f)
        return True

    def update_v_CBA(self, p, vType=0, xType=0):
        X = 0
        V = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new
        if vType == 0:
            V = self.v
        else:
            V = self.v_new

        f = self.calc_f()
        Xdif = p - X
        Vsum = np.zeros(len(V))
        for item in self.neighbors:
            Vsum = Vsum + (item.v - V)
        w2 = 0 + (2 * self.w2 * rand(1)[0])
        w3 = 0 + (2 * self.w3 * rand(1)[0])

        self.v_new = (self.w1 * V) + (w2 * Vsum) + (w3 * Xdif * f)

        return True

    def update_x_byV(self, xType=0):
        X = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new

        self.x_new = X + self.v_new
        return True

    def update_x_byX(self, p, xType=0):
        X = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new

        k = rand(1)[0]
        self.x_new = X + (k * (p - X))
        return True

    def update_A(self):
        self.A = self.alpha * pow(self.A_base, self.generation)

    def update_r(self):
        self.r_new = self.r * (1 - np.exp(-1 * self.gama * self.generation))

    def randomWalk(self, xType=0):
        X = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new

        self.x_new = X + (self.epsilon * self.A)
        return True

    def walkAroundBest(self, p):
        k = rand(1)[0]
        self.x_new = self.xBest + (k * (p - self.xBest))
        return True

    def calc_f(self):
        beta = rand(1)[0]
        f = self.f_min + ((self.f_max - self.f_min) * beta)
        f = f/(self.f_max - self.f_min)
        return f

    def calc_ObjectiveValues(self,xType=0):
        X = 0
        if xType == 0:
            X = self.x
        else:
            X = self.x_new

        self.objectiveValues_new = []
        for item in self.objectiveFunction:
            fInput = []  # feature input
            for index in item.parameterIndex:
                fInput.append(X[index])
            self.objectiveValues_new.append(item.function(fInput))

    def calc_Fitness(self,allBats,oType=0):
        fitVal = 0
        if oType == 0:
            for i in range(len(self.objectiveFunction)):
                # allBats.sort(key=lambda x: x.objectiveValues[i])
                newAllBats = sorted(allBats, key=lambda x: x.objectiveValues[i])
                mmin = newAllBats[0].objectiveValues[i]
                mmax = newAllBats[-1].objectiveValues[i]
                fitVal = fitVal + ((self.objectiveValues[i] - mmin) / (mmax - mmin)) #normal
        else:
            for i in range(len(self.objectiveFunction)):
                # allBats.sort(key=lambda x: x.objectiveValues_new[i])
                newAllBats = sorted(allBats, key=lambda x: x.objectiveValues_new[i])
                mmin = newAllBats[0].objectiveValues_new[i]
                mmax = newAllBats[-1].objectiveValues_new[i]
                fitVal = fitVal + ((self.objectiveValues_new[i] - mmin) / (mmax - mmin))

        self.fitnessValue_new = fitVal
        return True

    def dominanceCheck(self, otherBat):
        dominance = True
        for i in range(len(self.objectiveValues)):
            if self.objectiveValues[i] >= otherBat.objectiveValues[i]:
                dominance = False
                break
        return dominance

    def copy(self, otherBat):
        self.x_new = otherBat.x_new
        self.v_new = otherBat.v_new
        self.objectiveValues_new = otherBat.objectiveValues_new
        self.fitnessValue_new = otherBat.fitnessValue_new
        return True

    def findNeighbors(self, selfIndex, graph, allBats):
        for i in range(len(graph)):
            if graph[selfIndex][i] == 1:
                self.neighbors.append(allBats[i])
        return True

    def submitIteration(self):

        self.x_history.append(self.x)
        self.x = self.x_new

        self.v_history.append(self.v)
        self.v = self.v_new

        self.objectiveValues_history.append(self.objectiveValues)
        self.objectiveValues = self.objectiveValues_new

        self.fitnessValue_history.append(self.fitnessValue)
        self.fitnessValue = self.fitnessValue_new

        self.r_history.append(self.r)
        self.r = self.r_new

        self.A_history.append(self.A)

        self.generation = self.generation + 1

        if self.fitnessValue_new >= self.fitnessValue:
            self.xBest = self.x_new

        return True

    def __str__(self):
        name = "X : " + str(self.x) + "\n" \
               + "fitness : " + str(self.fitnessValue) + "\n"\
               + "Objective :" + str(self.objectiveValues)
        return name


