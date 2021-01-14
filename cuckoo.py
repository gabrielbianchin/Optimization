import numpy as np 
from scipy.optimize import rosen

class Cuckoo:
	def __init__(self, population_size=1000, number_weights=2, max_it=100, lambda_value=1.5, lr=0.01, percent_reset=0.25):
		self.population_size = population_size
		self.MAX_IT = max_it
		self.lambda_value = lambda_value
		self.lr = lr
		self.number_weights = number_weights
		self.percent_reset = percent_reset

	def generate_population(self):
		self.population = np.random.uniform(low=0.0, high=10.0, size=(self.population_size, self.number_weights))
		self.fit = [np.inf for _ in range(self.population_size)]
		self.get_best()

	def fitness(self, individual):
		return rosen(individual)

	def get_best(self):
		individual_fitness = []
		for individual in self.population:
			individual_fitness.append([self.fitness(individual), individual])
		individual_fitness.sort(key=lambda x:x[0])
		self.population = [x[1] for x in individual_fitness[:]]
		self.best_fitness = self.fitness(self.population[0])
		self.best_weights = self.population[0]

	def iteration(self):
		for index_individual in range(self.population_size):
			sigma2 = 1
			sigma1 = np.power((np.random.gamma(1 + self.lambda_value) * np.sin(np.pi * self.lambda_value / 2)) / (np.random.gamma((1 + self.lambda_value) / 2) * self.lambda_value * np.power(2, (self.lambda_value - 1) / 2)), 1 / self.lambda_value)
			u = np.random.normal(0, sigma1, size=self.number_weights)
			v = np.random.normal(0, sigma2, size=self.number_weights)
			step = self.lr * (u / np.power(np.fabs(v), 1 / self.lambda_value))

			self.fit[index_individual] = self.fitness(self.population[index_individual] + step)
			
			j = np.random.randint(low=0, high=self.population_size)
			while j == index_individual:
				j = np.random.randint(low=0, high=self.population_size)

			if self.fit[j] < self.fit[index_individual]:
				self.fit[j] = self.fit[index_individual]

			index = int(self.population_size * self.percent_reset)
			self.population[self.population_size - index:] = np.random.uniform(low=0.0, high=10.0, size=(index, self.number_weights))

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

		