# Abstract EA Class
import numpy as np
from numpy.random import seed
from numpy.random import rand
from Bat import Bat
from Objective import Objective
from copy import deepcopy


class SEA:
    def __init__(self, q):
        self.q = q
        self.mutationRate = 0.05

    def crossover(self, x1, x2):
        child = deepcopy(x1)
        child.x = x1.x + (x2.x-x1.x)
        return child

    def mutation(self, x):
        for i in range(len(x.x)):
            if rand(1)[0] <= self.mutationRate:
                x.x[i] = x.x[i] + rand(1)[0]
        return x

    def optimize(self, batList):
        childs = []
        batCount = len(batList)
        for i in range(batCount):
            parents = []
            for i in range(2):
                selectionIndex = np.random.randint(low=0, high=batCount-1, size=self.q)
                selectionList = []
                for index in selectionIndex:
                    selectionList.append(batList[index])
                selectionList.sort(key=lambda x: x.fitnessValue)
                parents.append(selectionList[0])
            child = self.crossover(parents[0], parents[1])
            child = self.mutation(child)
            child.calc_ObjectiveValues()
            child.calc_Fitness(batList,oType=1)
            child.objectiveValues = child.objectiveValues_new
            child.fitnessValue = child.fitnessValue_new
            childs.append(child)

        return childs
