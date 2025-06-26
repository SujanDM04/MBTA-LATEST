import random
import numpy as np

class GeneticAlgorithmScheduler:
    def __init__(self, train_scheduler, geo_mean_df, train_capacity=1000, min_trains=3, 
                 population_size=50, generations=100, mutation_rate=0.1, elite_size=5,
                 crossover_rate=0.8):
        self.train_scheduler = train_scheduler
        self.train_capacity = train_capacity
        self.min_trains = min_trains
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.penalties = {
            'constraint_violation': 1e6,
            'overload': 10,
            'underutil': 100,
            'wait_time': 0.1,
            'frequency': 100,
            'no_trains': 1e8
        }
        self.slot_keys = list(self.train_scheduler.slots.keys())
        self.geo_lookup = {
            (row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] 
            for _, row in geo_mean_df.iterrows()
        }

    def cost_function(self, chromosome):
        total_cost = 0
        for i, key in enumerate(self.slot_keys):
            trains = chromosome[i]
            slot = self.train_scheduler.slots[key]
            geo_mean_ons = self.geo_lookup.get(key, 0)
            # Constraint violations
            if trains < self.min_trains or trains > slot.max_slots:
                total_cost += self.penalties['constraint_violation']
                continue
            if trains > 0:
                load_per_train = geo_mean_ons / trains
                overload = max(0, load_per_train - self.train_capacity)
                total_cost += overload**2 * self.penalties['overload']
                underutil = max(0, self.train_capacity * 0.3 - load_per_train)
                total_cost += underutil * self.penalties['underutil']
                frequency = 60 / trains
                wait_time = frequency / 2
                total_cost += wait_time * geo_mean_ons * self.penalties['wait_time']
                if frequency < slot.min_frequency:
                    total_cost += (slot.min_frequency - frequency) * self.penalties['frequency']
                elif frequency > slot.max_frequency:
                    total_cost += (frequency - slot.max_frequency) * self.penalties['frequency']
            else:
                if geo_mean_ons > 0:
                    total_cost += self.penalties['no_trains']
        return total_cost

    def fitness(self, chromosome):
        # Lower cost is better, so fitness is inverse
        return 1 / (1 + self.cost_function(chromosome))

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            for key in self.slot_keys:
                slot = self.train_scheduler.slots[key]
                min_val = max(self.min_trains, int(60 / slot.max_frequency))
                max_val = min(slot.max_slots, int(60 / slot.min_frequency))
                individual.append(random.randint(min_val, max_val))
            population.append(individual)
        return population

    def select(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i][:]
        return population[-1][:]

    def crossover(self, parent1, parent2):
        child1, child2 = parent1[:], parent2[:]
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def mutate(self, chromosome):
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                slot = self.train_scheduler.slots[self.slot_keys[i]]
                min_val = max(self.min_trains, int(60 / slot.max_frequency))
                max_val = min(slot.max_slots, int(60 / slot.min_frequency))
                mutated[i] = random.randint(min_val, max_val)
        return mutated

    def optimize(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = -np.inf
        best_cost = float('inf')
        cost_progress = []

        for ind in population:
            fit = self.fitness(ind)
            if fit > best_fitness:
                best_fitness = fit
                best_solution = ind[:]
                best_cost = 1 / fit - 1
        cost_progress.append(best_cost)

        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            max_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_solution = population[max_idx][:]
                best_cost = 1 / best_fitness - 1
            cost_progress.append(1 / max(fitness_scores) - 1)
            if generation % 20 == 0:
                print(f"Gen {generation}: Best = {best_cost:,.0f}")
            new_population = []
            # Elitism
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            while len(new_population) < self.population_size:
                parent1 = self.select(population, fitness_scores)
                parent2 = self.select(population, fitness_scores)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
        for i, key in enumerate(self.slot_keys):
            self.train_scheduler.slots[key].current_trains = best_solution[i]
        solution_dict = {self.slot_keys[i]: best_solution[i] for i in range(len(self.slot_keys))}
        print(f"Final Cost: {best_cost:,.0f}")
        return solution_dict, best_cost, cost_progress