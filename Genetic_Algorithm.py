class GeneticAlgorithm:
    def __init__(self, initial_pop_size=3, mutation_rate=0.9):
        self.initial_pop_size = initial_pop_size
        self.mutation_rate = mutation_rate
        
        # Define fixed ranges for indices 1 and 2
        self.index_0_range = (1, 10)
        self.index_1_range = (5, 15)

        # Define min/max values for segments
        self.index_2_min = 10
        self.index_5_max = 50

        self.index_6_min = 10
        self.index_9_max = 50

        self.index_10_min = 55
        self.index_13_max = 90

    def generate_initial_population(self):
        """Generate initial population arrays"""
        arrays = []
        for _ in range(self.initial_pop_size):
            array = self._generate_single_array()
            arrays.append(array)
        return arrays

    def _generate_single_array(self):
        """Generate a single array with the required constraints"""
        import random
        array = [0] * 14  # Start with 14 zeros
        
        # Assign values for index 1 and 2
        array[0] = random.randint(*self.index_0_range)
        array[1] = random.randint(*self.index_1_range)
        
        # Generate ascending values from index 3 to 8
        array[2] = random.randint(self.index_2_min, self.index_5_max - 30)  # Ensure room for growth
        for i in range(3, 6):
            lower_bound = array[i-1] + 1
            upper_bound = self.index_5_max - (6 - i)
            if lower_bound > upper_bound:  # Prevent empty range error
                lower_bound = upper_bound
            array[i] = random.randint(lower_bound, upper_bound)
            
        array[6] = random.randint(self.index_6_min, self.index_9_max - 30)  # Ensure room for growth
        for i in range(7, 10):
            lower_bound = array[i-1] + 1
            upper_bound = self.index_9_max - (10 - i)
            if lower_bound > upper_bound:  # Prevent empty range error
                lower_bound = upper_bound
            array[i] = random.randint(lower_bound, upper_bound)

        # Generate ascending values from index 9 to 14
        array[10] = random.randint(self.index_10_min, self.index_13_max - (14 - 11))  # Ensure room for growth
        for i in range(11, 14):
            lower_bound = array[i-1] + 1
            upper_bound = self.index_13_max - (14 - i)
            if lower_bound > upper_bound:  # Prevent empty range error
                lower_bound = upper_bound
            array[i] = random.randint(lower_bound, upper_bound)

        # After generating the initial array, perform the insertions and extension
        array.extend([65, 70])  # This adds two elements at the end
        array[10:10] = [65, 70]  # This inserts at index 10
        array[6:6] = [325, 450]  # This inserts at index 6
        return array

    def crossover(self, parent1, parent2):
        """Perform segmented crossover between two parents"""
        arrays = []
        # Define segmentation points
        segment0 = 1  # After the first element
        segment1 = 2  # After the first two elements
        segment2 = 6  # After the sixth element
        segment3 = 10  # After the tenth element  
        
        # Create children by swapping middle segments
        child1 = parent1[:segment0] + parent2[segment0:segment1] + parent1[segment1:segment2] + parent2[segment2:segment3] + parent1[segment3:]
        child2 = parent2[:segment0] + parent1[segment0:segment1] + parent2[segment1:segment2] + parent1[segment2:segment3] + parent2[segment3:]
        
        mutated_child1, mutated_child2 = self.mutation(child1, child2)
        arrays.append(mutated_child1)
        arrays.append(mutated_child2)
        return arrays

    def mutation(self, child1, child2):
        """Perform mutation on two children"""
        import random
        
        def safe_randint(min_val, max_val):
            # Ensure min is less than max
            min_val = int(min_val)
            max_val = int(max_val)
            if min_val >= max_val:
                return min_val  # Return minimum if range is invalid
            return random.randint(min_val, max_val)

        mchild1 = child1.copy()
        mchild2 = child2.copy()

        # Mutation for child1
        for i in range(len(mchild1)):
            if random.random() <= self.mutation_rate:
                if i == 0 or i == 1:  # First two elements
                    mchild1[i] = random.randint(1, 7)
                elif i == 2:  # First element in middle section
                    mchild1[i] = safe_randint(max(1, mchild1[i]-10), mchild1[i+1]-1)
                elif i == 3 or i == 4:  # Middle section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i+1]-1)
                elif i == 5:  # Last index in middle section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i]+10)
                elif i == 6:  # First element in middle section
                    mchild1[i] = safe_randint(max(1, mchild1[i]-10), mchild1[i+1]-1)
                elif i == 7 or i == 8:  # Middle section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i+1]-1)
                elif i == 9:  # Last index in middle section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i]+10)
                elif i == 10:  # First index in last section
                    mchild1[i] = safe_randint(max(1, mchild1[i]-10), mchild1[i+1]-1)
                elif i == 11 or i == 12:  # Last section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i+1]-1)
                elif i == 13:  # Last index in last section
                    mchild1[i] = safe_randint(mchild1[i-1]+1, mchild1[i]+10)

        # Mutation for child2 
        for i in range(len(mchild2)):
            if random.random() <= self.mutation_rate:
                if i == 0 or i == 1:
                    mchild2[i] = random.randint(1, 7)
                elif i == 2:
                    mchild2[i] = safe_randint(max(1, mchild2[i]-10), mchild2[i+1]-1)
                elif i == 3 or i == 4:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i+1]-1)
                elif i == 5:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i]+10)
                elif i == 6:
                    mchild2[i] = safe_randint(max(1, mchild2[i]-10), mchild2[i+1]-1)
                elif i == 7 or i == 8:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i+1]-1)
                elif i == 9:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i]+10)
                elif i == 10:
                    mchild2[i] = safe_randint(max(1, mchild2[i]-10), mchild2[i+1]-1)
                elif i == 11 or i == 12:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i+1]-1)
                elif i == 13:
                    mchild2[i] = safe_randint(mchild2[i-1]+1, mchild2[i]+10)

        return mchild1, mchild2 