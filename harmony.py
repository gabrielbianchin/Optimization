import numpy as np 
from scipy.optimize import rosen

class Harmony:
	def __init__(self, population_size=1000, number_weights=2, max_it=100, hm=100, hmcr=0.90, par=0.10):
		self.population_size = population_size
		self.MAX_IT = max_it
		self.HM = hm
		self.HMCR = hmcr
		self.PAR = par
		self.number_weights = number_weights

	def generate_population(self):
		self.population = np.random.uniform(low=0.0, high=10.0, size=(self.population_size, self.number_weights))
		self.get_best()

	def fitness(self, individual):
		return rosen(individual)

	def get_best(self):
		individual_fitness = []
		for individual in self.population:
			individual_fitness.append([self.fitness(individual), individual])
		individual_fitness.sort(key=lambda x:x[0])
		self.population = [x[1] for x in individual_fitness[:self.HM]]
		self.best_fitness = self.fitness(self.population[0])
		self.best_weights = self.population[0]

	def iteration(self):
		for _ in range(self.population_size - self.HM):
			new_individual = []
			for index in range(self.number_weights):
				values = []
				for individual in self.population:
					values.append(individual[index])
				a = float(np.random.choice(values, size=1))
				b = float(np.random.uniform(low=0.0, high=10.0, size=1))
				new_weight = float(np.random.choice([a, b], size=1, p=[self.HMCR, 1-self.HMCR]))
				if new_weight == b:
					new_weight = float(np.random.choice([b, b-0.5, b+0.5], size=1, p=[1-self.PAR, self.PAR/2, self.PAR/2]))
				new_individual.append(new_weight)
			self.population.append(np.array(new_individual))
			self.get_best()

	def operate(self, verbose=1):
		self.generate_population()
		for epoch in range(self.MAX_IT):
			self.iteration()
			if verbose == 1:
				print('Epoch', epoch+1)
				print('Best Fitness:', self.best_fitness)
				print('Weights:', self.best_weights)
				print()

		