def segmented_crossover(parent1, parent2):
    # Define segmentation points
    segment0 = 1 #After the first element
    segment1 = 2  # After the first two elements
    segment2 = 6  # After the sixth element
    segment3 = 10 # After the tenth element  
    
    # Create children by swapping middle segments
    child1 = parent1[:segment0] + parent2[segment0:segment1] + parent1[segment1:segment2] + parent2[segment2:segment3] + parent1[segment3:]
    child2 = parent2[:segment0] + parent1[segment0:segment1] + parent2[segment1:segment2] + parent1[segment2:segment3] + parent2[segment3:]
    
    return child1, child2

# Example parents
parent1 = ['3', '13', '14', '28', '35', '43', '15', '44', '47', '48', '69', '83', '88', '89']
parent2 = ['6', '9', '12', '19', '36', '43', '19', '40', '43', '47', '59', '64', '71', '81']

# Perform crossover
child1, child2 = segmented_crossover(parent1, parent2)

# Print results
print(parent1)
print("Child 1:", child1)
print(parent2)
print("Child 2:", child2)


