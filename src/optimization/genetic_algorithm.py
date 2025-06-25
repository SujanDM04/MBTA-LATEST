import random
import numpy as np
from optimization.cost import CostFunction

class GeneticAlgorithmScheduler:
    def __init__(self, slots, geo_mean_df, penalties, train_capacity=1000, min_trains=3, 
                 population_size=50, generations=1000, mutation_rate=0.1, elite_size=5,
                 crossover_rate=0.8):
        """        
        slots: Dictionary of HillClimbingTimeSlot objects
        geo_mean_df: DataFrame with geometric mean ons for each (day_type_name, hour, direction_id)
        penalties: Dictionary of penalty values for cost calculation
        train_capacity: Maximum passengers per train
        min_trains: Minimum number of trains per hour
        population_size: Number of individuals in population
        generations: Number of iterations/generations
        mutation_rate: Probability of mutation per gene
        elite_size: Number of best individuals to keep each generation
        crossover_rate: Probability of crossover between parent pairs
        """
        self.slots = slots
        self.train_capacity = train_capacity
        self.min_trains = min_trains
        self.population_size = population_size
        self.generations = generations  
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.penalties = penalties
        
        self.slot_keys = list(self.slots.keys())
        
        # Preprocess geo_mean_df into a dict for fast lookup
        self.geo_lookup = {
            (row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] 
            for _, row in geo_mean_df.iterrows()
        }

    def compute_fitness(self, chromosome):
        """Apply chromosome to slots and compute fitness (inverse of cost)"""
        for i, key in enumerate(self.slot_keys):
            self.slots[key].current_trains = chromosome[i]
        cost = CostFunction(self).cost_fn()
        return 1 / (1 + cost)

    def initialize_population(self):
        """Create initial population with random chromosomes respecting constraints."""
        population = []
        for _ in range(self.population_size):
            individual = []
            for key in self.slot_keys:
                slot = self.slots[key]
                # Updated: use full valid range without frequency-based limits
                min_possible = self.min_trains
                max_possible = slot.max_slots
                trains = random.randint(min_possible, max_possible)
                individual.append(trains)
            population.append(individual)
        return population

    def roulette_wheel_selection(self, population, fitness_scores):
        """Standard roulette wheel selection"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i][:]
        return population[-1][:]  # fallback

    def uniform_crossover(self, parent1, parent2):
        """Standard uniform crossover"""
        child1, child2 = parent1[:], parent2[:]
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2

    def single_point_crossover(self, parent1, parent2):
        """Standard single-point crossover"""
        if len(parent1) <= 1:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2

    def mutate(self, chromosome):
        """Standard mutation - randomly change genes"""
        mutated = chromosome[:]
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                slot = self.slots[self.slot_keys[i]]
                # Updated: use full valid range without frequency-based limits
                min_possible = self.min_trains
                max_possible = slot.max_slots
                mutated[i] = random.randint(min_possible, max_possible)
        
        return mutated

    def optimize(self):
        """Main genetic algorithm optimization loop"""
        population = self.initialize_population()
        best_solution = None
        best_fitness = 0
        best_cost = float('inf')
        cost_progress = []

        # Find initial best
        for ind in population:
            fitness = self.compute_fitness(ind)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = ind[:]
                best_cost = 1 / best_fitness - 1

        print(f"GA Initial Best Cost: {best_cost:,.2f}")
        cost_progress.append(best_cost)

        for generation in range(self.generations):  
            # Evaluate population
            fitness_scores = [self.compute_fitness(ind) for ind in population]
            
            # Track best solution
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx][:]
                best_cost = 1 / best_fitness - 1

            current_best_cost = 1 / max(fitness_scores) - 1
            cost_progress.append(current_best_cost)

            # Progress reporting
            if generation % 10 == 0:
                print(f"Generation {generation:3d}: Current Best = {current_best_cost:,.2f}, Overall Best = {best_cost:,.2f}")

            # Create next generation
            new_population = []
            
            # Elitism - keep best individuals
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.roulette_wheel_selection(population, fitness_scores)
                parent2 = self.roulette_wheel_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:self.population_size]

        # Apply best solution
        for i, key in enumerate(self.slot_keys):
            self.slots[key].current_trains = best_solution[i]

        # Convert to dictionary format
        solution_dict = {self.slot_keys[i]: best_solution[i] for i in range(len(self.slot_keys))}

        print(f"GA Final Best Cost: {best_cost:,.2f}")

        return solution_dict, best_cost, cost_progress