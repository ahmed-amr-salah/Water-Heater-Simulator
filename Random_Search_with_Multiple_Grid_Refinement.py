import random
import numpy as np

class random_search_with_multiple_grid_refinement:
    
    def __init__(self, initial_grid_size, v40, no_combinations):
        self.ranges = self.get_ranges(v40) 
        self.grid_size = initial_grid_size
        self.no_combinations = no_combinations

        # Generate the initial grid and combinations
        self.initial_grid = self.generate_grid(self.grid_size, self.ranges)
        self.initial_combinations_set = self.generate_combinations(self.initial_grid, self.no_combinations)

    @staticmethod
    def get_ranges(v40):
        """Generate parameter ranges based on the given v40 value."""
        return {
            "first": [1, 5],
            "duration": [2, 6],
            "v40_level_0": [0.1 * v40, 0.2 * v40],
            "v40_level_1": [0.15 * v40, 0.3 * v40],
            "v40_level_2": [0.2 * v40, 0.45 * v40],
            "v40_level_3": [0.4 * v40, 0.6 * v40],
            "v40_inlet_0": [20, 30],
            "v40_inlet_1": [30, 34],
            "v40_inlet_2": [33, 38],
            "v40_inlet_3": [36, 42],
            "v40_outlet_0": [42, 45],
            "v40_outlet_1": [44, 50],
            "v40_outlet_2": [48, 55],
            "v40_outlet_3": [54, 65],
        }
    
    def generate_grid(self, grid_size, ranges):
        grid = []
        for param in ranges:
            step_size = max(1, (ranges[param][1] - ranges[param][0]) // (grid_size - 1))
            grid.append([int(ranges[param][0] + i * step_size) for i in range(grid_size)])
        return grid

    def generate_combinations(self, grid, no_combinations):
        combinations_set = []
        while len(combinations_set) < no_combinations:
            combination = [random.choice(param_values) for param_values in grid]
            if self.is_valid_combination(combination):
                combination.extend([65,70])
                combination[10:10]=[65,70]
                combination[6:6]=[325,450]
                combinations_set.append(combination)
        return combinations_set

    @staticmethod
    def is_valid_combination(combination):
        return (
            combination[2] < combination[3] < combination[4] < combination[5] and
            combination[6] < combination[7] < combination[8] < combination[9] and
            combination[10] < combination[11] < combination[12] < combination[13]
        )

    def grid_refinement_optimization(self, tested_combinations, k, refinement_factor):
        refined_combinations = []
    
        # Sort and select top-k best combinations
        current_best_combinations = tested_combinations.sort_values(
            by="Efficiency", ascending=False).head(k)
    
        refined_grid_size = max(2, self.grid_size // refinement_factor)

        parameters = list(self.ranges.keys())  # Extract parameter names

        for row in current_best_combinations.itertuples(index=False):
            comb = row.Individual  # Access the 'Individual' column
            indices_to_remove = [6, 7, 12, 13, 18, 19]  # Last two indices are 18 & 19 (0-based)
            comb = np.delete(comb, indices_to_remove)
            refined_ranges = {}

            for param_idx, param in enumerate(parameters):
                param_value = comb[param_idx]
                step_size = (self.ranges[param][1] - self.ranges[param][0]) / (refined_grid_size - 1)

                low = max(self.ranges[param][0], param_value - step_size * (refined_grid_size // 2))
                high = min(self.ranges[param][1], param_value + step_size * (refined_grid_size // 2))

                refined_ranges[param] = [low, high]

            refined_grid = self.generate_grid(refined_grid_size, refined_ranges)
            generated_combinations = self.generate_combinations(refined_grid, self.no_combinations // k)
            refined_combinations.extend(generated_combinations)
        self.refined_combinations = refined_combinations
        return self.refined_combinations


