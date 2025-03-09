import numpy as np

class SmartAlgorithm:
    def __init__(self, time_step, heating_profile, round_heater_power, outlet_heater_power, inlet_heater_power, 
    first, duration, v40_levels, inlet_water_temp, outlet_water_temp = [0,0,0,0,0,0], heater_type = 1):
        self.heater_type = heater_type# 1 for round, 0 for flat
        self.first = first
        self.duration = duration
        self.v40_levels = v40_levels
        self.inlet_water_temp = inlet_water_temp
        self.outlet_water_temp = outlet_water_temp
        self.no_time_steps = 30/time_step   #no. time steps in one slot (30 min)
        self.round_heater_power = round_heater_power
        self.inlet_heater_power = inlet_heater_power
        self.outlet_heater_power = outlet_heater_power
        self.heating_profile = heating_profile
        self.time_steps = time_step 
        
        # ----------------------------for testing remove comments ---------#

        #size = len(array_given)
        #self.size = size       
        #self.array_liters = heating_profile [:,0]
        ##------------------------------------------------##
        #-------- add comment to these when testing ------#
        if self.heater_type==1:
            size = len(heating_profile)//int(self.no_time_steps)
            array_temp_off_steps = np.zeros(len(heating_profile))
            array_temp_off = np.zeros((size,2))  # temp off in time steps 
        else:
            size = heating_profile.shape[0]//int(self.no_time_steps)
            array_temp_off = np.zeros((size,2)) #   required temp off
            array_temp_off_steps = np.zeros((len(heating_profile), 2)) # temp off in time steps

        self.size = size    
        self.array_temp_off = array_temp_off
        self.array_temp_off_steps = array_temp_off_steps
        self.array_liters = np.zeros(size) # contains liters of water used based on conversion
        #------------------------------------------------#
        
        self.array_smartLiter = np.zeros(size)
        self.sum_array = np.zeros(size)
        

    def calculate_liters(self, array):
        """Calculate the liters of water heated in a given interval."""
        if self.heater_type ==1 :
            return (self.round_heater_power * np.sum(array)) / (1.163 * (40 - 15) * (60/self.time_steps)) 
        else:
            return (self.outlet_heater_power * np.sum(array[:,0])) / (1.163 * (40 - 15) * (60/self.time_steps)) + (self.inlet_heater_power * np.sum(array[:,1])) / (1.163 * (40 - 15) * (60/self.time_steps)) 

    def calculate_sum(self, start, end, arrayL):
        """ sum for liters"""
        if start <= len(arrayL) and end <= len(arrayL): # case one when endIdx is less than arrayL
            return np.sum(arrayL[start:end])
        elif start <= len(arrayL) and end > len(arrayL): # case 2 when endIdx is greater than. In this case the sum is made of two parts
            remaining = end - len(arrayL)
            part1 = np.sum(arrayL[start:])
            part2 = np.sum(arrayL[:remaining])
            return part1 + part2

    def assign_v40_level(self, sum_lv5_lv4, sum_lv0tolv3=None, preV40Element=None):
        
        """Assigning the correct v40 level"""
        
        if self.v40_levels[4] <= sum_lv5_lv4 < self.v40_levels[5]:
            return [[self.outlet_water_temp[4],self.inlet_water_temp[4]]], self.v40_levels[4]
        elif sum_lv5_lv4 >= self.v40_levels[5]:
            return [[self.outlet_water_temp[5],self.inlet_water_temp[5]]], self.v40_levels[5]
        else:
            if sum_lv0tolv3 is None:
                return None, None
            elif sum_lv0tolv3 <= self.v40_levels[0]:
                return [[self.outlet_water_temp[0],self.inlet_water_temp[0]]], self.v40_levels[0]
            elif sum_lv0tolv3 < self.v40_levels[1]:
                if preV40Element >= self.v40_levels[1]: # special case for vlevel 1
                    return [[self.outlet_water_temp[1],self.inlet_water_temp[1]]], self.v40_levels[1]
                else:
                    return [[self.outlet_water_temp[0],self.inlet_water_temp[0]]], self.v40_levels[0]
            elif self.v40_levels[1] <= sum_lv0tolv3 < self.v40_levels[2]:
                return [[self.outlet_water_temp[1],self.inlet_water_temp[1]]], self.v40_levels[1]
            elif self.v40_levels[2] <= sum_lv0tolv3 < self.v40_levels[3]:
                return [[self.outlet_water_temp[2],self.inlet_water_temp[2]]], self.v40_levels[2]
            elif self.v40_levels[3] <= sum_lv0tolv3 < self.v40_levels[4]:
                return [[self.outlet_water_temp[3],self.inlet_water_temp[3]]], self.v40_levels[3]

    def run(self): # method for running the code 
        #conversion for liters ----- add comment when testing since we are given already liters 
        
        for i in range(self.size):
          start_idx = i * int(self.no_time_steps)
          end_idx = (i + 1) *int( self.no_time_steps)
          if self.heater_type ==1:
              self.array_liters[i] = self.calculate_liters(self.heating_profile[start_idx:end_idx])
          else:
              self.array_liters[i] = self.calculate_liters(self.heating_profile[start_idx:end_idx,:]) 
        
          #------------------------------------------------------------------------------------------- #
        # assigning the v40 level
        
        
        for j in range(self.size):
            start_idx = j + self.first
            levels_4and5_end_idx = j + self.first + 48
            levels_0to3_end_idx = j + self.first + self.duration

            if start_idx <= len(self.array_liters):
                sum_lv4_lv5 = self.calculate_sum(start_idx, levels_4and5_end_idx, self.array_liters)
                temp_off, v40_assigned= self.assign_v40_level(sum_lv4_lv5, None, self.array_smartLiter[j - 1])
                if temp_off is not None:
                    self.array_temp_off[j] = np.array(temp_off[0]).flatten()
                    self.array_smartLiter[j] = v40_assigned
                    self.sum_array[j]= sum_lv4_lv5

                if temp_off is None and v40_assigned is None:
                    sum_lv0Tilllv3 = self.calculate_sum(start_idx, levels_0to3_end_idx, self.array_liters)
                    temp_off, v40_assigned = self.assign_v40_level(sum_lv4_lv5, sum_lv0Tilllv3, self.array_smartLiter[j - 1])
                    self.array_temp_off[j] =  np.array(temp_off[0]).flatten()
                    self.array_smartLiter[j] = v40_assigned
                    self.sum_array[j]= sum_lv0Tilllv3
                    

            elif start_idx > len(self.array_liters):
                diff = start_idx - len(self.array_liters)
                start_new = self.first - diff
                sum_lv4_lv5 = self.calculate_sum(start_new, start_new + 48, self.array_liters)

                temp_off, v40_assigned  = self.assign_v40_level(sum_lv4_lv5, None, self.array_smartLiter[j - 1])
                
                if temp_off is not None:
                    self.array_temp_off[j] = np.array(temp_off[0]).flatten()
                    self.array_smartLiter[j] = v40_assigned
                    self.sum_array[j]= sum_lv4_lv5

                if temp_off is None and v40_assigned is None:
                    sum_lv0Tilllv3 = self.calculate_sum(start_new, start_new + self.duration, self.array_liters)
                    temp_off, v40_assigned = self.assign_v40_level(sum_lv4_lv5, sum_lv0Tilllv3, self.array_smartLiter[j - 1])
                    self.array_temp_off[j] = np.array(temp_off[0]).flatten()
                    self.array_smartLiter[j] = v40_assigned
                    self.sum_array[j]= sum_lv0Tilllv3
                    

        for n in range(self.size):
            start_idx = n * int(self.no_time_steps)
            end_idx = (n + 1) *int( self.no_time_steps)
            if self.heater_type==1:
                self.array_temp_off_steps[start_idx:end_idx] = self.array_temp_off[n,1]   
            else:   
              self.array_temp_off_steps[start_idx:end_idx,:] = self.array_temp_off[n,:]

        return self.array_temp_off_steps
    
    
#testing

#array_given = np.load("flat_profiles.npy")
#array_given = np.load("heating_profiles.npy")

#array_used = array_given
#array_used = array_given[:,2]




