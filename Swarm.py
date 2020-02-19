# Abstract Swarm Class
import numpy as np
from numpy.random import seed
from numpy.random import rand
from Bat import Bat
from Objective import Objective
from copy import deepcopy
from EA import SEA
import matplotlib
import matplotlib.pyplot as plt


class Swarm:

    def __init__(self, name, numberOfSwarm, objectiveFunction, numOfFeature, featureRange, numberOfIter, q):
        self.swarmName = name
        self.graph = []
        self.archive = []
        self.numberOfSwarm = numberOfSwarm
        self.archiveLimit = 50
        self.swarms = []
        self.objectiveFunction = objectiveFunction
        self.numOfFeature = numOfFeature
        self.featureRange = featureRange
        self.f_min = 0
        self.f_max = 100
        self.w1 = 1
        self.w2 = 0.8
        self.w3 = 0.8
        self.A_base = 1
        self.r_base = 0.5
        self.alpha = 0.9
        self.gama = 0.9
        self.epsilon = 0.1
        self.numberOfIter = numberOfIter
        self.EA = SEA(q)

        self.bestSwarm = 0
        self.bestX = []
        self.bestFitness = 99999999999999999
        self.best_objectiveValue = []

        self.average_archive_fitness_History = []
        self.average_archive_objectiveValue_History = []

        self.best_fitness_History = []
        self.best_objectiveValue_History = []

    def createGraph(self):
        matrix = np.zeros((self.numberOfSwarm, self.numberOfSwarm))
        for i in range(self.numberOfSwarm):
            for j in range((i + 1), self.numberOfSwarm):
                if rand(1)[0] <= 0.2:
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        return matrix

    def initialize(self):
        self.swarms = []
        for i in range(self.numberOfSwarm):
            self.swarms.append(Bat(
                numOfFeature=self.numOfFeature,
                featureRange=self.featureRange,
                f_min=self.f_min,
                f_max=self.f_max,
                w1=self.w1,
                w2=self.w2,
                w3=self.w3,
                A_base=self.A_base,
                alpha=self.alpha,
                r_base=self.r_base,
                gama=self.gama,
                epsilon=self.epsilon,
                objectiveFunction=self.objectiveFunction
            ))

        self.graph = self.createGraph()

        for i in range(len(self.swarms)):
            self.swarms[i].findNeighbors(i, self.graph, self.swarms)

        for item in self.swarms:
            item.calc_ObjectiveValues(xType=0)

        for item in self.swarms:
            item.calc_Fitness(self.swarms, oType=1)

        for item in self.swarms:
            item.objectiveValues = item.objectiveValues_new
            item.objectiveValues_new = []
            item.fitnessValue = item.fitnessValue_new
            item.fitnessValue_new = 999999999

        for item in self.swarms:
            isDominated = False
            for aItem in self.swarms:
                if aItem.dominanceCheck(item):
                    isDominated = True
                    break
            if not isDominated:
                self.archive.append(deepcopy(item))

        self.updateBest()

        return True

    def updateArchive(self, newList):
        newlistRemove = []
        archiveRemove = []
        newArchive = []
        for i in range(len(newList)):
            for j in range(len(self.archive)):
                if self.archive[j].dominanceCheck(newList[i]):
                    newlistRemove.append(i)
                    break
                elif newList[i].dominanceCheck(self.archive[j]):
                    archiveRemove.append(j)
                    break

        for i in range(len(newList)):
            if i not in newlistRemove:
                newArchive.append(deepcopy(newList[i]))

        for j in range(len(self.archive)):
            if j not in archiveRemove:
                newArchive.append(self.archive[j])

        newArchive.sort(key=lambda x: x.fitnessValue)
        if len(newArchive) > self.archiveLimit:
            newArchive = newArchive[0:self.archiveLimit]

        self.archive = newArchive

    def submitIteration(self):
        for item in self.swarms:
            item.submitIteration()
        return True

    def updateBest(self, fType=0):
        if fType == 0:
            self.swarms.sort(key=lambda x: x.fitnessValue)
            if self.swarms[0].fitnessValue <= self.bestFitness:
                self.bestSwarm = deepcopy(self.swarms[0])
                self.bestX = deepcopy(self.swarms[0].x)
                self.bestFitness = deepcopy(self.swarms[0].fitnessValue)
                self.best_objectiveValue = deepcopy(self.swarms[0].objectiveValues)
        else:
            self.swarms.sort(key=lambda x: x.fitnessValue_new)
            if self.swarms[0].fitnessValue_new <= self.bestFitness:
                self.bestSwarm = deepcopy(self.swarms[0])
                self.bestX = deepcopy(self.swarms[0].x_new)
                self.bestFitness = deepcopy(self.swarms[0].fitnessValue_new)
                self.best_objectiveValue = deepcopy(self.swarms[0].objectiveValues_new)

        self.best_fitness_History.append(deepcopy(self.bestFitness))
        self.best_objectiveValue_History.append(deepcopy(self.best_objectiveValue))

        return True

    def CBA(self):
        for item in self.swarms:
            item.update_v_BA(p=self.bestX, vType=0, xType=0)  # Basic method to update BAT
            item.update_x_byV(xType=0)
            item.update_A()
            item.update_r()
            item.calc_ObjectiveValues(xType=1)
            item.calc_Fitness(self.swarms, oType=0)
            if rand(1)[0] > item.r_new:
                item.update_v_CBA(p=self.bestX, vType=1, xType=1)
                item.update_x_byV(xType=1)
                item.calc_ObjectiveValues(xType=1)
                item.calc_Fitness(self.swarms, oType=0)
                if item.fitnessValue_new <= self.bestFitness:
                    self.updateBest(fType=1)

            item.randomWalk()
            item.calc_ObjectiveValues(xType=1)
            item.calc_Fitness(self.swarms, oType=0)
            if rand(1)[0] < item.A and item.fitnessValue_new <= self.bestFitness:
                self.updateBest(fType=1)

        return True

    def updateArchiveHistory(self):
        self.average_archive_fitness_History.append(
            np.mean(
                [item.fitnessValue for item in self.archive]
            )
        )
        tempOlist = []
        for i in range(len(self.objectiveFunction)):
            tempOlist.append(
                np.mean(
                    [item.objectiveValues[i] for item in self.archive]
                )
            )
        self.average_archive_objectiveValue_History.append(tempOlist)

    def optimize(self):
        for i in range(self.numberOfIter):
            print("iteration : " + str(i))
            self.CBA()
            self.submitIteration()
            self.updateBest()
            self.updateArchive(self.swarms)
            EA_output = self.EA.optimize(self.swarms)
            self.updateArchive(EA_output)
            self.updateArchiveHistory() # to draw charts

        return True

    def chartHistories(self):

        ###Avg Fitness
        t = range(len(self.average_archive_fitness_History))
        s = self.average_archive_fitness_History

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='time', ylabel='avg Fitness',
               title='average Fitness over time')
        ax.grid()

        fig.savefig("Fitness_AVG.png")

        #best Fitness
        plt.clf()
        t = range(len(self.best_fitness_History))
        s = self.best_fitness_History

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='time', ylabel='best Fitness',
               title='best Fitness over time')
        ax.grid()

        fig.savefig("Fitness_Best.png")

        #AVG Objective
        for i in range(len(self.objectiveFunction)):
            plt.clf()

            t = range(len(self.average_archive_objectiveValue_History))
            s = np.array(self.average_archive_objectiveValue_History)[:,i]

            fig, ax = plt.subplots()
            ax.plot(t, s)

            ax.set(xlabel='time', ylabel='avg ' + self.objectiveFunction[i].name,
                   title='average ' + self.objectiveFunction[i].name + ' over time')
            ax.grid()

            fig.savefig(self.swarmName + "_" + self.objectiveFunction[i].name + "_AVG.png")

        #Best Objective
        for i in range(len(self.objectiveFunction)):
            plt.clf()
            t = range(len(self.best_objectiveValue_History))
            s = np.array(self.best_objectiveValue_History)[:, i]

            fig, ax = plt.subplots()
            ax.plot(t, s)

            ax.set(xlabel='time', ylabel='Best ' + self.objectiveFunction[i].name,
                   title='Best ' + self.objectiveFunction[i].name + ' over time')
            ax.grid()

            fig.savefig(self.swarmName + "_" + self.objectiveFunction[i].name + "_Best.png")

        return True
