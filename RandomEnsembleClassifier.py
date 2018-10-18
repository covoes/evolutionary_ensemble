import random
import numpy as np
import operator

class RandomEnsembleClassifier:
    def __init__(self, algorithms, population_size = 100, random_state = None):
        random.seed(random_state)
        self.population_size = population_size
        self.population = []
        for i in range(0, population_size):
            algorithm = random.choice(list(algorithms.keys()))
            params = {}
            for hyperparameter, h_range in algorithms[algorithm].items():
                if isinstance(h_range[0], str):
                    params[hyperparameter] = random.choice(h_range)
                elif isinstance(h_range[0], float):
                    params[hyperparameter] = random.uniform(h_range[0], h_range[1]+1)
                else:
                    params[hyperparameter] = random.randint(h_range[0], h_range[1]+1)
            self.population.append(algorithm(**params))


    def fit(self, X, y):
        for classifier in self.population:
            classifier.fit(X, y)

    def predict(self, X):
        predictions = np.empty((self.population_size, len(X)))
        y = np.empty(len(X))
        for classifier in range(0, self.population_size):
            predictions[classifier] = self.population[classifier].predict(X)
        for i in range(0, len(X)):
            pred = {}
            for j in range(0, self.population_size):
                if predictions[j][i] in pred:
                    pred[predictions[j][i]] += 1
                else:
                    pred[predictions[j][i]]  = 1
            y[i] = max(pred.items(), key=operator.itemgetter(1))[0]
        return y
