import random

# Convert string values to integers
#child1 = [3, 13, 13, 28, 35, 43, 15, 44, 47, 48, 69, 83, 88, 89]  
#child2 = [6, 9, 12, 19, 36, 43, 19, 40, 43, 47, 59, 64, 71, 81]

def mutation(child1, child2):
    mchild1 = child1.copy()
    mchild2 = child2.copy()
    mutation_rate = 0.9

    def safe_randint(min_val, max_val):
        # Ensure min is less than max
        min_val = int(min_val)
        max_val = int(max_val)
        if min_val >= max_val:
            return min_val  # Return minimum if range is invalid
        return random.randint(min_val, max_val)

    # Mutation for child1
    for i in range(len(mchild1)):
        if random.random() <= mutation_rate:
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

    # Mutation for child2 (fixed to use mchild2 instead of mchild1)
    for i in range(len(mchild2)):
        if random.random() <= mutation_rate:
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

# Run mutation function
#mutated_child1, mutated_child2 = mutation(child1, child2)

# Print original and mutated lists
##print('Original 1:', child1)
##print('Mutated 1:', mutated_child1)
##print('Original 2:', child2)
##print('Mutated 2:', mutated_child2)