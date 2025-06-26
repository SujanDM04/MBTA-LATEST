"""
Genetic Algorithm Implementation for Train Scheduling Optimization

This module implements a genetic algorithm approach to optimize train scheduling
by finding the optimal number of trains for each time slot while considering
passenger demand, capacity constraints, and operational requirements.

The genetic algorithm uses:
- Population-based evolution with selection, crossover, and mutation
- Fitness-based selection using roulette wheel method
- Elitism to preserve the best solutions across generations
- Constraint-aware cost function with penalty mechanisms
"""

import random
import numpy as np


class GeneticAlgorithmScheduler:
    """
    Genetic Algorithm Scheduler for optimizing train allocation.
    
    This class implements a genetic algorithm to find optimal train schedules
    by evolving a population of potential solutions. Each solution represents
    a complete train allocation across all time slots and directions.
    
    The algorithm considers multiple constraints:
    - Minimum and maximum trains per time slot
    - Frequency constraints (time between trains)
    - Passenger capacity and demand
    - Operational efficiency metrics
    
    Attributes:
        train_scheduler: The base train scheduler with slot definitions
        train_capacity: Maximum passengers per train
        min_trains: Minimum trains required per hour
        population_size: Size of the genetic algorithm population
        generations: Number of evolution generations
        mutation_rate: Probability of mutation per gene
        elite_size: Number of best individuals to preserve
        crossover_rate: Probability of crossover between parents
        penalties: Dictionary of penalty weights for constraint violations
        slot_keys: List of time slot identifiers
        geo_lookup: Dictionary mapping slots to geometric mean passenger demand
    """
    
    def __init__(self, train_scheduler, geo_mean_df, train_capacity=1000, min_trains=3, 
                 population_size=50, generations=100, mutation_rate=0.1, elite_size=5,
                 crossover_rate=0.8):
        """
        Initialize the Genetic Algorithm Scheduler.
        
        parameters:
            train_scheduler: Base train scheduler with slot definitions
            geo_mean_df: DataFrame containing geometric mean passenger demand
            train_capacity: Maximum passengers per train (default: 1000)
            min_trains: Minimum trains required per hour (default: 3)
            population_size: Size of genetic algorithm population (default: 50)
            generations: Number of evolution generations (default: 100)
            mutation_rate: Probability of mutation per gene (default: 0.1)
            elite_size: Number of best individuals to preserve (default: 5)
            crossover_rate: Probability of crossover between parents (default: 0.8)
        """
        self.train_scheduler = train_scheduler
        self.train_capacity = train_capacity
        self.min_trains = min_trains
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        
        # Penalty weights for different constraint violations
        self.penalties = {
            'constraint_violation': 1e6,  # Heavy penalty for invalid solutions
            'overload': 10,               # Penalty for exceeding train capacity
            'underutil': 100,             # Penalty for underutilized trains
            'wait_time': 0.1,             # Penalty for passenger wait times
            'frequency': 100,             # Penalty for frequency violations
            'no_trains': 1e8              # Extreme penalty for no trains when demand exists
        }
        
        # Extract slot keys for chromosome representation
        self.slot_keys = list(self.train_scheduler.slots.keys())
        
        # Create lookup dictionary for geometric mean passenger demand
        self.geo_lookup = {
            (row['day_type_name'], row['hour'], row['direction_id']): row['geo_mean_ons'] 
            for _, row in geo_mean_df.iterrows()
        }

    def cost_function(self, chromosome):
        """
        Calculate the total cost of a chromosome (train allocation solution).
        
        The cost function evaluates the quality of a solution based on:
        - Constraint violations (min/max trains, frequency limits)
        - Passenger load distribution (overload/underutilization)
        - Wait time costs
        - Operational efficiency
        
        parameters:
            chromosome: List of train counts for each time slot
            
        Returns:
            float: Total cost of the solution (lower is better)
        """
        total_cost = 0
        
        # Evaluate each time slot in the chromosome
        for i, key in enumerate(self.slot_keys):
            trains = chromosome[i]
            slot = self.train_scheduler.slots[key]
            geo_mean_ons = self.geo_lookup.get(key, 0)
            
            # Check for constraint violations (min/max trains)
            if trains < self.min_trains or trains > slot.max_slots:
                total_cost += self.penalties['constraint_violation']
                continue
            
            if trains > 0:
                # Calculate load per train and related costs
                load_per_train = geo_mean_ons / trains
                
                # Overload penalty (quadratic penalty for exceeding capacity)
                overload = max(0, load_per_train - self.train_capacity)
                total_cost += overload**2 * self.penalties['overload']
                
                # Underutilization penalty (linear penalty for low utilization)
                underutil = max(0, self.train_capacity * 0.3 - load_per_train)
                total_cost += underutil * self.penalties['underutil']
                
                # Wait time cost (proportional to demand and frequency)
                frequency = 60 / trains
                wait_time = frequency / 2  # Average wait time
                total_cost += wait_time * geo_mean_ons * self.penalties['wait_time']
                
                # Frequency constraint penalties
                if frequency < slot.min_frequency:
                    total_cost += (slot.min_frequency - frequency) * self.penalties['frequency']
                elif frequency > slot.max_frequency:
                    total_cost += (frequency - slot.max_frequency) * self.penalties['frequency']
            else:
                # Heavy penalty for no trains when there's demand
                if geo_mean_ons > 0:
                    total_cost += self.penalties['no_trains']
                    
        return total_cost

    def fitness(self, chromosome):
        """
        Calculate fitness score for a chromosome.
        
        Fitness is the inverse of cost, so lower cost solutions have higher fitness.
        The 1/(1+cost) transformation ensures fitness is always positive and bounded.
        
        parameters:
            chromosome: List of train counts for each time slot
            
        Returns:
            float: Fitness score (higher is better)
        """
        # Lower cost is better, so fitness is inverse
        return 1 / (1 + self.cost_function(chromosome))

    def initialize_population(self):
        """
        Initialize a random population of potential solutions.
        
        Each individual in the population is a chromosome representing
        train allocations for all time slots. Initial values are constrained
        by the minimum/maximum train requirements and frequency limits.
        
        Returns:
            list: List of chromosomes (each chromosome is a list of train counts)
        """
        population = []
        
        # Generate population_size individuals
        for _ in range(self.population_size):
            individual = []
            
            # Generate train count for each time slot
            for key in self.slot_keys:
                slot = self.train_scheduler.slots[key]
                
                # Calculate valid range for this slot
                min_val = max(self.min_trains, int(60 / slot.max_frequency))
                max_val = min(slot.max_slots, int(60 / slot.min_frequency))
                
                # Randomly assign train count within valid range
                individual.append(random.randint(min_val, max_val))
                
            population.append(individual)
            
        return population

    def select(self, population, fitness_scores):
        """
        Select an individual from population using roulette wheel selection.
        
        Roulette wheel selection gives higher probability of selection
        to individuals with higher fitness scores, while still allowing
        lower fitness individuals a chance to be selected.
        
        parameters:
            population: List of chromosomes
            fitness_scores: List of fitness scores corresponding to population
            
        Returns:
            list: Selected chromosome (copy of original)
        """
        total_fitness = sum(fitness_scores)
        
        # Handle case where all fitness scores are zero
        if total_fitness == 0:
            return random.choice(population)
        
        # Roulette wheel selection
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i][:]  # Return copy of selected individual
                
        return population[-1][:]  # Fallback to last individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent chromosomes.
        
        
        parameters:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            tuple: Two child chromosomes (child1, child2)
        """
        child1, child2 = parent1[:], parent2[:]
        
        # Uniform crossover: each gene has 50% chance of being swapped
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
                
        return child1, child2

    def mutate(self, chromosome):
        """
        Apply mutation to a chromosome.
        
        
        
        parameters:
            chromosome: Chromosome to mutate
            
        Returns:
            list: Mutated chromosome (copy of original)
        """
        mutated = chromosome[:]
        
        # Apply mutation to each gene with probability mutation_rate
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                slot = self.train_scheduler.slots[self.slot_keys[i]]
                
                # Calculate valid range for this slot
                min_val = max(self.min_trains, int(60 / slot.max_frequency))
                max_val = min(slot.max_slots, int(60 / slot.min_frequency))
                
                # Assign new random value within valid range
                mutated[i] = random.randint(min_val, max_val)
                
        return mutated

    def optimize(self):
        """
        Run the genetic algorithm optimization process.
        
        
        
        
        Returns:
            tuple: (solution_dict, best_cost, cost_progress)
                - solution_dict: Dictionary mapping slots to train counts
                - best_cost: Cost of the best solution found
                - cost_progress: List of best costs for each generation
        """
        # Initialize population with random solutions
        population = self.initialize_population()
        
        # Track the best solution found so far
        best_solution = None
        best_fitness = -np.inf
        best_cost = float('inf')
        cost_progress = []

        # Evaluate initial population and find best individual
        for ind in population:
            fit = self.fitness(ind)
            if fit > best_fitness:
                best_fitness = fit
                best_solution = ind[:]
                best_cost = 1 / fit - 1  # Convert fitness back to cost
                
        cost_progress.append(best_cost)

        # Main evolution loop
        for generation in range(self.generations):
            # Calculate fitness for all individuals in current population
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Find best individual in current generation
            max_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_solution = population[max_idx][:]
                best_cost = 1 / best_fitness - 1
                
            # Track progress
            cost_progress.append(1 / max(fitness_scores) - 1)
            
            # Print progress every 20 generations
            if generation % 20 == 0:
                print(f"Gen {generation}: Best = {best_cost:,.0f}")
                
            # Create new population for next generation
            new_population = []
            
            # Elitism: preserve the best individuals
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
                
            # Generate remaining individuals through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = self.select(population, fitness_scores)
                parent2 = self.select(population, fitness_scores)
                
                # Perform crossover with probability crossover_rate
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                    
                # Apply mutation to offspring
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add offspring to new population
                new_population.extend([child1, child2])
                
            # Ensure population size is maintained
            population = new_population[:self.population_size]
            
        # Update the train scheduler with the best solution found
        for i, key in enumerate(self.slot_keys):
            self.train_scheduler.slots[key].current_trains = best_solution[i]
            
        # Convert solution to dictionary format
        solution_dict = {self.slot_keys[i]: best_solution[i] for i in range(len(self.slot_keys))}
        
        print(f"Final Cost: {best_cost:,.0f}")
        return solution_dict, best_cost, cost_progress