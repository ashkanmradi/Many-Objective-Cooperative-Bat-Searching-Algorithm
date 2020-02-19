# Abstract Objective Class


class Objective:
    def __init__(self,name, parameterIndex=[], function=lambda: 0):
        self.name = name
        self.parameterIndex = parameterIndex
        self.function = function
