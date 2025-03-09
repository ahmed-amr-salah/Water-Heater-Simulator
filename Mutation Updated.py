import random

# Convert string values to integers
child1 = [3, 13, 13, 28, 35, 43, 15, 44, 47, 48, 69, 83, 88, 89]  
child2 = [6, 9, 12, 19, 36, 43, 19, 40, 43, 47, 59, 64, 71, 81]

def mutation(child1, child2):
    mchild1 = child1.copy()  # Create a copy to store mutations in another array
    mchild2 = child2.copy()
    mutation_rate = 0.9

    for i in range(len(mchild1)):  # Loop through indices
        if i == 0 or i == 1:  # First two elements
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(1, 7) #these ranges are placeholders
        elif i == 2: #First element in middle section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1) 

        elif i==3 or i==4:  # Middle section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 5: #Last index in middle section 
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)
                
        elif i == 6: #First element in middle section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1) 

        elif i==7 or i==8:  # Middle section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 9: #Last index in middle section 
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)
                
        elif i == 10: # First index in last section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1)
        elif i==11 or i==12:  # Last section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 13: #Last index in last section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)
                
#Nafs el kalam le child2
    for i in range(len(mchild2)):  # Loop through indices
        if i == 0 or i == 1:  # First two elements
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(1, 7) #these ranges are placeholders
        elif i == 2: #First element in middle section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1) 

        elif i==3 or i==4:  # Middle section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 5: #Last index in middle section 
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)
                
        elif i == 6: #First element in middle section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1) 

        elif i==7 or i==8:  # Middle section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 9: #Last index in middle section 
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)
                
        elif i == 10: # First index in last section
            if random.random() <= mutation_rate:
               mchild1[i] = random.randint(mchild1[i]-10, mchild1[i+1]+-1)
        elif i==11 or i==12:  # Last section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i+1]-1)
        elif i == 13: #Last index in last section
            if random.random() <= mutation_rate:
                mchild1[i] = random.randint(mchild1[i-1]+1, mchild1[i]+10)

    return mchild1, mchild2  # Return the mutated list

# Run mutation function
mutated_child1, mutated_child2 = mutation(child1, child2)

# Print original and mutated lists
print('Original 1:', child1)
print('Mutated 1:', mutated_child1)
print('Original 2:', child2)
print('Mutated 2:', mutated_child2)