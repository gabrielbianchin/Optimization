import numpy as np 
from scipy.optimize import rosen

class SineCosine:
	def __init__(self, population_size=1000, number_weights=2, a=2, max_it=100):
		self.population_size = population_size
		self.MAX_IT = max_it
		self.a = a
		self.number_weights = number_weights
		self.best_fitness = np.inf

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
		self.population = [x[1] for x in individual_fitness[:]]
		if self.best_fitness > self.fitness(self.population[0]):
			self.best_fitness = self.fitness(self.population[0])
			self.best_weights = self.population[0]

	def iteration(self, t=1):
		r1 = self.a - t*(self.a / self.MAX_IT)
		r2 = float(np.random.uniform(low=0.0, high=2*np.pi, size=1))
		r3 = float(np.random.uniform(low=0.0, high=2.0, size=1))
		r4 = float(np.random.uniform(low=0.0, high=1.0, size=1))

		for index_individual in range(self.population_size):
			for d in range(self.number_weights):
				if r4 < 0.5:
					self.population[index_individual][d] = self.population[index_individual][d] + (r1 * np.sin(r2) * abs(r3 * self.best_weights[d] - self.population[index_individual][d]))
				else:
					self.population[index_individual][d] = self.population[index_individual][d] + (r1 * np.cos(r2) * abs(r3 * self.best_weights[d] - self.population[index_individual][d]))
		self.get_best()

	def operate(self, verbose=1):
		self.generate_population()
		for epoch in range(self.MAX_IT):
			self.iteration(epoch+1)
			if verbose == 1:
				print('Epoch', epoch+1)
				print('Best Fitness:', self.best_fitness)
				print('Weights:', self.best_weights)
				print()

		