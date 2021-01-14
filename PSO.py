import numpy as np 
from scipy.optimize import rosen

class PSO:
	def __init__(self, population_size=1000, number_weights=2, max_it=100, w=0.5, c1=0.8, c2=0.9):
		self.population_size = population_size
		self.MAX_IT = max_it
		self.w = w
		self.c1 = c1
		self.c2 = c2
		self.number_weights = number_weights

	def generate_population(self):
		self.population = np.random.uniform(low=0.0, high=10.0, size=(self.population_size, self.number_weights))
		self.best_individual_position = self.population
		self.best_individual_fitness = [np.inf for _ in range(self.population_size)]
		self.individual_velocity = [0 for _ in range(self.population_size)]
		self.best_fitness = np.inf
		self.get_best()

	def fitness(self, individual):
		return rosen(individual)

	def get_best(self):
		for index_individual in range(self.population_size):
			fit = self.fitness(self.population[index_individual])
			if self.best_individual_fitness[index_individual] > fit:
				self.best_individual_fitness[index_individual] = fit
				self.best_individual_position[index_individual] = self.population[index_individual]
			if self.best_fitness > fit:
				self.best_fitness = fit
				self.best_weights = self.population[index_individual]

	def iteration(self):
		self.get_best()

		for index_individual in range(self.population_size):
			self.individual_velocity[index_individual] = (self.w * self.individual_velocity[index_individual]) + \
				(self.c1 * np.random.random()) * (self.best_individual_position[index_individual] - self.population[index_individual]) + \
				(self.c2 * np.random.random()) * (self.best_weights - self.population[index_individual])
			self.population[index_individual] += self.individual_velocity[index_individual]

	def operate(self, verbose=1):
		self.generate_population()
		for epoch in range(self.MAX_IT):
			self.iteration()
			if verbose == 1:
				print('Epoch', epoch+1)
				print('Best Fitness:', self.best_fitness)
				print('Weights:', self.best_weights)
				print()