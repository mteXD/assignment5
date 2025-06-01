import numpy as np
import time
from opfunu.cec_based import cec2022

# Constants
MAX_ITERS = 10000

class Search:
    def __init__(self, func, lb, ub):
        self.func = func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)

class RandomLocalSearch:
    def __init__(self, search, max_iters=50000):
        self.search = search
        self.max_iters = max_iters

    def run(self):
            currBestPoint = np.random.uniform(self.search.lb, self.search.ub)
            currBestVal = self.search.func.evaluate(currBestPoint.tolist())
            #print(currBestPoint.tolist())
            for _ in range(self.max_iters):
                rand_point =  np.random.uniform(self.search.lb, self.search.ub)
                eval = self.search.func.evaluate(rand_point.tolist())
                if eval < currBestVal:
                    currBestPoint = rand_point
                    currBestVal = eval
            return currBestPoint, currBestVal

class GradientDescentLocalSearch:   
    def __init__(self, search, stepSize=1, max_iters=5000):
        self.search = search
        self.max_iters = max_iters
        self.stepSize = stepSize
    
    def __grad(self, vector):
        grad = np.zeros_like(vector)
        diff = 1e-5
        for i in range(self.search.dim):
            x_plus_diff = vector.copy()
            x_plus_diff[i] += diff*0.5
            x_minus_diff = vector.copy()
            x_minus_diff[i] -= diff*0.5
            """
            central difference method
            https://www.youtube.com/watch?v=safjLBaOWuA&ab_channel=Udacity
            odvod v točki X = (f(vector[i]+diff*0.5) - f(vector[i]-diff*0.5))/ diff
            """
            f_plus = self.search.func.evaluate(x_plus_diff.tolist())
            f_minus = self.search.func.evaluate(x_minus_diff.tolist())
            grad[i] = (f_plus - f_minus) / diff
        return grad
    
    def run(self):
        bestPoint = np.random.uniform(self.search.lb, self.search.ub)
        x = bestPoint
        bestValue = self.search.func.evaluate(bestPoint.tolist())
        f_x = bestValue
        for _ in range(0, self.max_iters, self.stepSize):
            x_next = x - self.stepSize * self.__grad(x)
            x_next = np.clip(x_next, self.search.lb, self.search.ub)
            f_next = self.search.func.evaluate(x_next.tolist())
            if(f_next < f_x):
                bestPoint = x_next
                bestValue = f_next
            x = x_next
            f_x = f_next 
        return bestPoint, bestValue

class SimulatedAnnealing:
    def __init__(self, search, max_iters=500000, initial_temp=10000.0, lambda_=0.9995, delta=0.1):
        self.search = search
        self.max_iters = max_iters
        self.initial_temp = initial_temp
        self.lambda_ = lambda_  # cooling rate
        self.delta = delta      # neighborhood perturbation range

    def _neighbor(self, s):
        neighbor = s + np.random.uniform(-self.delta, self.delta, size=self.search.dim)
        return np.clip(neighbor, self.search.lb, self.search.ub)

    def run(self):
        currentPoint = bestPoint = np.random.uniform(self.search.lb, self.search.ub)
        bestValue = self.search.func.evaluate(bestPoint.tolist())
        currentValue = bestValue
        T = self.initial_temp

        for _ in range(self.max_iters):
            neighbour = self._neighbor(currentPoint)
            neighbourValue = self.search.func.evaluate(neighbour.tolist())

            # If S' is better than best-so-far
            if neighbourValue < bestValue:
                bestPoint = neighbour
                bestValue = neighbourValue

            # If S' is better than current, accept
            if neighbourValue < currentValue:
                currentPoint = neighbour
                currentValue = neighbourValue
            else:
                # Maybe accept worse S'
                prob = np.exp(-(neighbourValue - currentValue) / T)
                if np.random.rand() < prob:
                    currentPoint = neighbour
                    currentValue = neighbourValue

            # Cool down
            T *= self.lambda_

        return bestPoint, bestValue

class BestDescentLocalSearch:
    def __init__(self, search, max_iters=MAX_ITERS, neighborhood_size=100, delta=0.1):
        self.search = search
        self.max_iters = max_iters
        self.neighborhood_size = neighborhood_size
        self.delta = delta

    def _generate_neighbors(self, current):
        """
        Generates random neighbors around the current point, clipped within bounds.
        """
        perturbations = np.random.uniform(-self.delta, self.delta, size=(self.neighborhood_size, self.search.dim))
        neighbors = current + perturbations
        neighbors = np.clip(neighbors, self.search.lb, self.search.ub)
        return neighbors

    def run(self):
        current = np.random.uniform(self.search.lb, self.search.ub)  # Random starting point
        current_val = self.search.func.evaluate(current.tolist())

        for _ in range(self.max_iters):
            neighbors = self._generate_neighbors(current)
            values = np.array([self.search.func.evaluate(n.tolist()) for n in neighbors])
            best_idx = np.argmin(values)
            best_val = values[best_idx]

            if best_val < current_val:
                current = neighbors[best_idx]
                current_val = best_val
            else:
                break  # No improvement found

        return current, current_val

class GuidedLocalSearch:
    def __init__(self, search, max_iters=MAX_ITERS, neighborhood_size=100, delta=0.1, lamb=0.01):
        self.search = search
        self.max_iters = max_iters
        self.neighborhood_size = neighborhood_size
        self.delta = delta
        self.lamb = lamb  # penalty weight

        self.penalties = np.zeros((self.search.dim,))  # one penalty per dimension

    def _generate_neighbors(self, current):
        perturbations = np.random.uniform(-self.delta, self.delta, size=(self.neighborhood_size, self.search.dim))
        neighbors = current + perturbations
        return np.clip(neighbors, self.search.lb, self.search.ub)

    def _penalized_eval(self, x):
        """
        Returns original cost + lambda * penalty term.
        Each dimension contributes penalty if value changes a lot.
        """
        x_arr = np.array(x)
        raw_val = self.search.func.evaluate(x_arr.tolist())
        penalty_term = np.sum(self.penalties * np.abs(x_arr))
        return raw_val + self.lamb * penalty_term

    def _update_penalties(self, x):
        """
        Increases penalty on dimensions with largest utility.
        Utility is abs(value) / (1 + penalty).
        """
        x_arr = np.abs(np.array(x))
        utilities = x_arr / (1 + self.penalties)
        max_util_idx = np.argmax(utilities)
        self.penalties[max_util_idx] += 1

    def run(self):
        current = np.random.uniform(self.search.lb, self.search.ub)
        current_val = self._penalized_eval(current)
        best_true = self.search.func.evaluate(current.tolist())
        best_point = current.copy()

        for _ in range(self.max_iters):
            neighbors = self._generate_neighbors(current)
            evaluated = np.array([self._penalized_eval(n) for n in neighbors])
            best_idx = np.argmin(evaluated)

            if evaluated[best_idx] < current_val:
                current = neighbors[best_idx]
                current_val = evaluated[best_idx]

                raw_val = self.search.func.evaluate(current.tolist())
                if raw_val < best_true:
                    best_true = raw_val
                    best_point = current.copy()
            else:
                self._update_penalties(current)

        return best_point, best_true

class GeneticAlgorithm:
    def __init__(self, search, population_size=50, max_iters=1000, mutation_rate=0.1, tournament_size=3):
        self.search = search
        self.population_size = population_size
        self.max_iters = max_iters
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _initialize_population(self):
        return np.random.uniform(self.search.lb, self.search.ub, size=(self.population_size, self.search.dim))

    def _evaluate_population(self, population):
        return np.array([self.search.func.evaluate(ind.tolist()) for ind in population])

    def _tournament_selection(self, population, fitness):
        indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return population[best_idx]

    def _crossover(self, parent1, parent2):
        mask = np.random.rand(self.search.dim) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual):
        mutation = np.random.uniform(-1, 1, size=self.search.dim)
        mutation_mask = np.random.rand(self.search.dim) < self.mutation_rate
        individual[mutation_mask] += mutation[mutation_mask]
        return np.clip(individual, self.search.lb, self.search.ub)

    def run(self):
        population = self._initialize_population()
        fitness = self._evaluate_population(population)
        best_idx = np.argmin(fitness)
        best_point = population[best_idx].copy()
        best_val = fitness[best_idx]

        for _ in range(self.max_iters):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            population = np.array(new_population)
            fitness = self._evaluate_population(population)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_val:
                best_point = population[current_best_idx].copy()
                best_val = fitness[current_best_idx]

        return best_point, best_val

class WalrusOptimizer:
    def __init__(self, search, population_size=100, max_iters=MAX_ITERS, alpha=1.0, beta=1.5, delta=1, bound_factor=1.05):
        self.search = search
        self.population_size = population_size
        self.max_iters = max_iters
        self.alpha = alpha  # attraction to random other walrus
        self.beta = beta    # attraction to best walrus
        self.delta = delta    # movement influence
        self.bound_factor = bound_factor

    def _initialize_population(self):
        return np.random.uniform(self.search.lb, self.search.ub, size=(self.population_size, self.search.dim))

    def _evaluate(self, population):
        return np.array([self.search.func.evaluate(ind.tolist()) for ind in population])



    def run(self):
        pop = self._initialize_population()
        fitness = self._evaluate(pop)

        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_val = fitness[best_idx]

        bound = 1

        def _update(new_pos):
            nonlocal best_val, best_pos

            new_position = np.clip(new_pos, self.search.lb, self.search.ub)

            new_val = self.search.func.evaluate(new_position.tolist())
            if new_val < fitness[i]:
                pop[i] = new_position
                fitness[i] = new_val

                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_position.copy()

        for _ in range(self.max_iters):
            for i in range(self.population_size):
                # Social-inspired movement:
                rand_vector = np.random.uniform(-1, 1, self.search.dim)
                direction = self.beta * (best_pos - pop[i]) * (self.delta * rand_vector)
                new_position = pop[i] + direction

                _update(new_position)

                # Migration

                rand_vector = np.random.uniform(-1, 1, self.search.dim)
                random_idx = np.random.choice([j for j in range(len(pop)) if j != i])
                direction = self.alpha * (pop[random_idx] - pop[i]) * (self.delta * rand_vector)
                new_position = pop[i] + direction

                _update(new_position)

                # Avoiding predators

                rand_vector = np.random.uniform(-1, 1, self.search.dim)
                direction = (bound * rand_vector)
                bound /= self.bound_factor
                new_position = pop[i] + direction

        return best_pos, best_val

if __name__ == '__main__':
    functions = [
        cec2022.F12022, cec2022.F22022, cec2022.F32022, cec2022.F42022,
        cec2022.F52022, cec2022.F62022, cec2022.F72022, cec2022.F82022,
        cec2022.F92022, cec2022.F102022, cec2022.F112022, cec2022.F122022
    ]

    results = []

    # classes = [BestDescentLocalSearch, GuidedLocalSearch, GeneticAlgorithm, WalrusOptimizer, RandomLocalSearch, GradientDescentLocalSearch, SimulatedAnnealing]
    classes = [WalrusOptimizer]
    for c in classes:
        results = []
        print(f"Running {c.__name__}")

        start_time = time.time()

        for f in functions:
            func = f(ndim=20)
            search = Search(func, func.lb, func.ub)

            currentSearch = c(search, max_iters=10000)
            best_point, best_value = currentSearch.run()
            results.append((f.__name__, best_value, best_point))
            print(f"\t{f.__name__} - best value: {best_value:.5f}")

        print(f"Time consumed: {time.time() - start_time:.2f}")
        print()

        # Save results
        with open(f"{c.__name__}_results.txt", "w") as f:
            for name, val, point in results:
                coords = "\t".join(f"{x:.10f}" for x in point)
                f.write(f"Function: {name}\nMinimum value: {val:.10f}\nCoordinates: {coords}\n\n")
