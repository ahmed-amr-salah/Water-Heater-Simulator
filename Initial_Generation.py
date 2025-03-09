import random

# Define fixed ranges for indices 1 and 2
index_0_range = (1, 10)
index_1_range = (5, 15)

# Define min/max values for segments
index_2_min = 10
index_5_max = 50

index_6_min = 10
index_9_max = 50

index_10_min = 55
index_13_max = 90

def generate_array():
    array = [0] * 14  # Initialize array with 14 zeros
    
    # Assign values for index 1 and 2
    array[0] = random.randint(*index_0_range)
    array[1] = random.randint(*index_1_range)
    
    # Generate ascending values from index 3 to 8
    array[2] = random.randint(index_2_min, index_5_max - 30)  # Ensure room for growth
    for i in range(3, 6):
        lower_bound = array[i-1] + 1
        upper_bound = index_5_max - (6 - i)
        if lower_bound > upper_bound:  # Prevent empty range error
            lower_bound = upper_bound
        array[i] = random.randint(lower_bound, upper_bound)
        
    array[6] = random.randint(index_6_min, index_9_max - 30)  # Ensure room for growth
    for i in range(7, 10):
        lower_bound = array[i-1] + 1
        upper_bound = index_9_max - (10 - i)
        if lower_bound > upper_bound:  # Prevent empty range error
            lower_bound = upper_bound
        array[i] = random.randint(lower_bound, upper_bound)

    # Generate ascending values from index 9 to 14
    array[10] = random.randint(index_10_min, index_13_max - (14 - 11))  # Ensure room for growth
    for i in range(11, 14):
        lower_bound = array[i-1] + 1
        upper_bound = index_13_max - (14 - i)
        if lower_bound > upper_bound:  # Prevent empty range error
            lower_bound = upper_bound
        array[i] = random.randint(lower_bound, upper_bound)

    return array

# Generate 10 arrays
arrays = [generate_array() for _ in range(10)]

# Print the generated arrays
for i, arr in enumerate(arrays, 1):
    print(f"Array {i}: {arr}")

