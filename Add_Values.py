def modify_arrays(list_of_arrays):
    values_end = [100, 101]
    values_after_tenth = [200, 201]
    values_after_sixth = [300, 301]
    modified_arrays = []
    
    for arr in list_of_arrays:
        new_arr = arr.copy()
        
        # Ensure the array has at least 10 elements before inserting after the 10th element
        if len(new_arr) > 10:
            new_arr = new_arr[:10] + values_after_tenth + new_arr[10:]
        
        # Ensure the array has at least 6 elements before inserting after the 6th element
        if len(new_arr) > 6:
            new_arr = new_arr[:6] + values_after_sixth + new_arr[6:]
        
        # Append values at the end
        new_arr += values_end
        
        modified_arrays.append(new_arr)
    
    return modified_arrays

# Example usage
arrays = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
]


modified_arrays = modify_arrays(arrays)
for arr in modified_arrays:
    print(arr)