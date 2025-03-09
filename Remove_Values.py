def remove_inserted_spaces(list_of_arrays):
    cleaned_arrays = []
    
    for arr in list_of_arrays:
        new_arr = arr.copy()
        
        # Remove the last two values
        new_arr = new_arr[:-2]
        
        # Remove two values after the 12th value (index 12)
        if len(new_arr) > 14:
            new_arr = new_arr[:12] + new_arr[14:]
        
        # Remove two values after the 6th value (index 6)
        if len(new_arr) > 8:
            new_arr = new_arr[:6] + new_arr[8:]
        
        cleaned_arrays.append(new_arr)
    
    return cleaned_arrays

# Example usage
arrays = [
    [1, 2, 3, 4, 5, 6, 300, 301, 7, 8, 9, 10, 200, 201, 11, 12, 13, 14, 100, 101],
    [15, 16, 17, 18, 19, 20, 300, 301, 21, 22, 23, 24, 200, 201, 25, 26, 27, 28, 100, 101]
]

cleaned_arrays = remove_inserted_spaces(arrays)
for arr in cleaned_arrays:
    print(arr)