from Objective import Objective
from Swarm import Swarm


def X2(data):
    return data[0] ** 2


def X2_reverse(data):
    return 1 / (data[0] ** 2)


def X4(data):
    return data[0] ** 4


if __name__ == "__main__":
    name = "Power2"
    numberOfSwarm = 50

    firstObjective = Objective("X2", [0], X2)
    secondObjective = Objective("X2_reverse", [1], X2_reverse)
    objectiveFunction = [firstObjective, secondObjective]

    numOfFeature = 2
    featureRange = (
        (-100, 100),
        (-100, 100),
    )
    numberOfIter = 150
    q = 2

    swarm = Swarm(
        name=name,
        numberOfSwarm=numberOfSwarm,
        objectiveFunction=objectiveFunction,
        numOfFeature=numOfFeature,
        featureRange=featureRange,
        numberOfIter=numberOfIter,
        q=q
    )

    swarm.initialize()
    swarm.optimize()
    swarm.chartHistories()
    print(swarm.archive[0])
