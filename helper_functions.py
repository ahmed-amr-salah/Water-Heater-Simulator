import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from qpr_calc import qpr_calc
from openpyxl.formatting.rule import ColorScaleRule
import openpyxl
import math
import sys
from smart_algo import SmartAlgorithm
#from Random_Search_with_Multiple_Grid_Refinement import random_search_with_multiple_grid_refinement
from Initial_Generation import generate_array
from Crossover_Function import segmented_crossover
from Genetic_Algorithm import GeneticAlgorithm
# Initializing class variables
_C = 4184    # Water specific heat capacity (J/kgK)
_RHO = 1000 # Water density (kg/m^3)
_K = 0.6 # Water thermal conductivity

class WaterHeater:

    def __init__(
        self,
        initial_temperature,
        no_layers,
        QPR,
        capacity,
        diameter,
        power,
        cutoff_temperature,
        hysteresis,
        heating_element_start,
        heating_element_end,
        thermostat_start,
        thermostat_end,
        diffusion_parameter
    ):
        """
        :param float initial_temperature: initial temperature of the WH in degrees Celsius
        :param int no_layers: number of WH layers
        :param string code: model code of the WH
        :param float capacity: volume of the WH tank in Liters
        :param float diameter: diameter of the WH tank in mm
        :param float power: power of the WH heating element in Watts
        :param float cutoff_temperature: cutoff temperature of the thermostat in degrees Celsius
        :param float hysteresis: hysteresis of the thermostat in degrees Celsius
        :param float heating_element_start: start of the heating element in the tank in mm
        :param float heating_element_end: end of the heating element in the tank in mm
        :param float thermostat_start: start of the thermostat in the tank in mm
        :param float thermostat_end: end of the thermostat in the tank in mm
        
        """
        
        self.no_layers = no_layers
        self.heater_on = True
        self.capacity = capacity
        self.power = power
        self.cutoff_temperature = cutoff_temperature
        
        self.__QPR = QPR
        self.__diameter = diameter
        self.__hysteresis = hysteresis
        self.__area = math.pi*self.__diameter**2/(4e6) # Calculating Area in m^2
        self.__layer_capacity = self.capacity/self.no_layers
        self.__layer_height = 4*self.capacity*1e3/(math.pi*self.__diameter**2) / self.no_layers
        self.__heating_element_start = heating_element_start/(self.__layer_height * 1000)
        self.__heating_element_end = heating_element_end/(self.__layer_height * 1000)
        self.__thermostat_start = thermostat_start/(self.__layer_height * 1000)
        self.__thermostat_end = thermostat_end/(self.__layer_height * 1000)
        self.__layer_QPR = QPR * (1000 / 24) / self.no_layers
        self.__layer_power = self.calc_layer_power()
        self.__thermostat_location = self.thermostat_locator()
        self.__diffusion_parameter = diffusion_parameter
    
    
    
    
    def calc_layer_power(self):
        """
        Calculates the power received per layer when the heater is ON

        Returns
        -------
        An array indicating the amount of power received by each layer in Watts
        
        """
        
        layer_power = np.zeros(self.no_layers)
        layer_power[math.floor(self.__heating_element_start)] = math.ceil(self.__heating_element_start) - self.__heating_element_start 
        layer_power[math.floor(self.__heating_element_end)] = self.__heating_element_end - math.floor(self.__heating_element_end)
        layer_power[math.ceil(self.__heating_element_start):math.floor(self.__heating_element_end)] = 1
        layer_power *=  self.power  / sum(layer_power)
        
        return layer_power
        

    def thermostat_locator(self):
        """
        Locates the position of the thermostat relative to the heater layers
        
        Returns
        -------
        An array indicating the contribution % of each layer's temperature to the thermostat temperature

        """
        thermostat_location = np.zeros(self.no_layers)
        thermostat_location[math.floor(self.__thermostat_start)] = math.ceil(self.__thermostat_start) - self.__thermostat_start 
        thermostat_location[math.floor(self.__thermostat_end)] = self.__thermostat_end - math.floor(self.__thermostat_end)
        thermostat_location[math.ceil(self.__thermostat_start):math.floor(self.__thermostat_end)] = 1
        thermostat_location /= sum(thermostat_location)
        
        return thermostat_location
        
    def calc_thermostat_temp(self, temperature):
        """
        Calculates the thermostat reading from the temperature of all the layers

        Parameters
        ----------
        temperature : temperature array of the current timestep

        Returns
        -------
        The thermostat reading at the current time step

        """
        
        return np.dot(temperature, self.__thermostat_location)
    
    
    def thermostat(self, temperature, cutoff):        
        """
        Turns the heater on and off based on thermostat readings

        Parameters
        ----------
        temperature : temperature array of the current timestep

        Returns
        -------
        Operates directly on the heater_on attribute

        """
        thermostat_temp = self.calc_thermostat_temp(temperature)
        if thermostat_temp >= cutoff: #or np.max(self.temperature[timestep, :]) > 100:
            self.heater_on =  False
        elif thermostat_temp <= (cutoff - self.__hysteresis):
            self.heater_on = True
    
        
    
    def power_check(self, time_step):
        power_step = (self.power  * self.heater_on - self.__layer_QPR * self.no_layers) * time_step / 60000
        power_consumed = self.power  * self.heater_on * time_step / 60000
        
        return  power_step, power_consumed
    
    def diffusion(self, temperature, time_step, cutoff = None, first_tank = False):
        if cutoff is None:
            cutoff = self.cutoff_temperature
            
        time_step = time_step * 60
        # Diffusion Parameters
        layer_decrease_ratio = self.__diffusion_parameter[0]
        top_temp_max = self.__diffusion_parameter[1]
        top_temp_min = self.__diffusion_parameter[2]
        no_affected_layers_max = self.__diffusion_parameter[3]
        no_affected_layers_min = self.__diffusion_parameter[4]

        
        def heating(temperature):
            """
            Heats the WH layers given layer power and location of the heating element
            
            Parameters
            ----------
            temperature : temperature array of the current timestep
            time_step : size of the simulator timestep in minutes
        
            Returns
            -------
            An updated temperature array after heating
        
            """
            # Determine whether the heater is on/off based on Thermostat response
            
            # Return the layer temperature based on WH state (on/off),QPR, capacity
            self.thermostat(temperature, cutoff)
            if first_tank:
                self.heater_on = False
            
            return ((self.power  * self.heater_on)) / (self.capacity * _C) * time_step * self.no_layers
        
        def convec_affected_layer(top_temp_layer):
            num_affected_layers = no_affected_layers_min + (top_temp_layer - top_temp_min) * (no_affected_layers_max-no_affected_layers_min)/(top_temp_max-top_temp_min)
            
            return num_affected_layers
        
        def convec_temp_incr_dist(temperature):
            convec_affected_layers = convec_affected_layer(temperature[-1])
            whole_affec_layers = np.floor(convec_affected_layers)
            part_affec_layer = convec_affected_layers - whole_affec_layers
            
            layer_share_power = np.zeros(temperature.shape)
            layer_share_power[int(self.no_layers-whole_affec_layers):] = [layer_decrease_ratio ** (i/self.no_layers) 
                                                                   for i in range(int(self.no_layers-whole_affec_layers), self.no_layers)]
            layer_share_power[int(self.no_layers-whole_affec_layers)-1] = part_affec_layer * layer_decrease_ratio ** ((self.no_layers-whole_affec_layers)-1/ self.no_layers)
            delta_T_increase = heating(temperature)
            temperature += delta_T_increase * layer_share_power/sum(layer_share_power)
            
            return temperature
        
        def conduction(temperature):
            for i in range(self.no_layers):
                if i == 0:  # Bottom-most layer (forward difference)
                    conduction = _K / _C * (temperature[i+1] - temperature[i]) / self.__layer_height**2
                elif i == self.no_layers - 1:  # Top-most layer (backward difference)
                    conduction = _K / _C * (temperature[i-1] - temperature[i])/ self.__layer_height**2
                else:  # Central difference for intermediate layers
                    conduction = _K / _C * (temperature[i+1] - 2*temperature[i] + temperature[i-1])/ self.__layer_height**2
                
                temperature[i] += conduction * time_step  * self.heater_on
                
            return temperature
        
        T = np.copy(temperature)
        T = conduction(T)
        T = convec_temp_incr_dist(T)
        T -= (self.__layer_QPR) / (self.capacity/self.no_layers * _C) * time_step
        T = sorted(T)
        temperature = np.copy(T)
        
        
        return temperature
        
                   
    def tap(self, temperature, inlet_flow_rate, inlet_temperature):
        """
        Simulates the tapping of water from the top of the heater replacing 
        it with cold water at the bottom of the tank.

        Parameters
        ----------
        temperature : temperature array of the current timestep
        inlet_flow_rate : the number of layers tapped at this time step
        inlet_temperature : the temperature of the water replacing the tapped water
        
        Returns
        -------
        An updated temperature array after tapping

        """
        discritized_flow = np.floor(inlet_flow_rate).astype(int)
        fractional_flow = inlet_flow_rate - discritized_flow
        
        # Fully displace the taping layer if it falls within the number of fully displaced layers (discritized_flow)
        temp_before_mixing = np.zeros(self.no_layers)
        output_temperature = np.zeros(self.no_layers)
        for WH_layer in range(self.no_layers):
            if WH_layer < discritized_flow:
                temp_before_mixing[WH_layer] = inlet_temperature
            else:
                temp_before_mixing[WH_layer] = temperature[WH_layer - discritized_flow]  # Shifting layers
                
        # Doing the mixing of the partial layer entering
        for WH_layer in range(self.no_layers):  
            if WH_layer == 0:  # Mixing the first layer with the inlet fractional layer
                output_temperature[WH_layer] = (temp_before_mixing[WH_layer] * (1 - fractional_flow) 
                                                        + inlet_temperature * fractional_flow)
            else:  # Mixing all the other layer with layer below
                output_temperature[WH_layer] = (temp_before_mixing[WH_layer] * (1 - fractional_flow) 
                                                        + temp_before_mixing[WH_layer - 1] * fractional_flow)
        return output_temperature

def plot_results(WaterHeater, time_step, cutoff_temperature, hysteresis, 
                 regulation_timesteps, temperatures = None, mode = 'Top'):
    
    time_array =(np.array(range(np.shape(temperatures)[0])) - regulation_timesteps) * time_step 
    
    plt.figure(dpi=400)
    # Add plot labels
    plt.title("Temperature Profile")
    plt.xlabel("Time (mins)")
    plt.ylabel("Temperature (C)")
    
    # Add horizontal lines indicating cutoff and heating temperatures
    plt.axhline(y=cutoff_temperature, color='r', linestyle='--')
    plt.axhline(y=(cutoff_temperature - hysteresis), color='r', linestyle='--')
       
    match mode:
        case 'Top':
            plt.plot(time_array, temperatures[:, -1])
        
        case 'Thermostat':
            plt.plot(time_array, WaterHeater.calc_thermostat_temp(temperatures))
            
        case 'All':
            for layer in range(np.shape(temperatures)[1]):
                plt.plot(time_array, temperatures[:, layer], label=f"Layer {layer+1}")
            plt.legend()
    

def save_results(filename, sheetname, temperatures, no_layers, time_step,
                 tapped_volume, input_temperature, output_temperature, energy_tapped, 
                 heating_profile, power_error_accumalation):
    
    time_array = np.array(range(len(temperatures))) * time_step
    
    initiation_file_time_temp = pd.DataFrame(temperatures, columns=[f'Layer {i+1}' for i in range(no_layers)])
    initiation_file_time_temp.insert(0, "Time (mins)", time_array)
    initiation_file_time_temp.insert(no_layers+1, "Volume Tapped (L)", tapped_volume)
    initiation_file_time_temp.insert(no_layers+2, "T_in (C)", input_temperature)
    initiation_file_time_temp.insert(no_layers+3, "T_out (C)", output_temperature)
    initiation_file_time_temp.insert(no_layers+4, "Q Tapped (kWh)", energy_tapped)
    initiation_file_time_temp.insert(no_layers+5, "Heating Profile", heating_profile)
    initiation_file_time_temp.insert(no_layers+6, "Accumalated Power Error (kWh)", power_error_accumalation)
    
    with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        initiation_file_time_temp.to_excel(writer, sheet_name = sheetname, index=False)

    # Open the workbook to apply conditional formatting
    wb = openpyxl.load_workbook(filename)
    ws = wb[sheetname]

    # Define color gradient rule for temperature columns
    temp_columns_range = f"B2:{chr(66 + no_layers - 1)}{len(time_array) + 1}"
    color_rule = ColorScaleRule(
        start_type="min", start_color="63BE7B",                   # Blue for minimum
        mid_type="percentile", mid_value=50, mid_color="FFEB84",  # Yellow at median
        end_type="max", end_color="F8696B")                       # Red for maximum
    
    ws.conditional_formatting.add(temp_columns_range, color_rule)

    # Save the file with conditional formatting
    wb.save(filename)
    print(f'Data saved to the sheet "{sheetname}" in {filename}!')

def tapping(WaterHeater, time_step, initial_temperature, tapping_initiation_file, setting = None, cutoff = []):
    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    layer_capacity = (WaterHeater.capacity/WaterHeater.no_layers)
    tapping_start_time = tapping_initiation_file['time (min)']
    inlet_temperature = tapping_initiation_file['Tinlet']
    tapping_power = tapping_initiation_file['Q Tapped (kWh)'] 
    inlet_flow_rate = tapping_initiation_file['flow (L/min)'] * time_step/layer_capacity
    tapping_start_time = np.append(tapping_start_time, np.inf)
    
    if setting is None:
        cutoff = np.full(simulation_timesteps, WaterHeater.cutoff_temperature)
    tap_index = 0  # Index of the current tap (number of taps done)
    current_tap_power = 0
    current_tap_timesteps = 0
    
    discritized_flow = np.floor(inlet_flow_rate).astype(int)  # Number of layers fully displaced
    fractional_flow = inlet_flow_rate - discritized_flow  # Fractional layer displaced
    
    temperature = np.zeros((simulation_timesteps, WaterHeater.no_layers))
    temperature[0] = initial_temperature
    tapped_volume = np.zeros(simulation_timesteps)
    output_temperature = np.zeros(simulation_timesteps)
    inlet_water_temperature = np.zeros(simulation_timesteps)
    power_step = np.zeros(simulation_timesteps)
    accumalated_power = np.zeros(simulation_timesteps)
    energy_tapped = np.zeros(simulation_timesteps)
    internal_energy_change = np.zeros(simulation_timesteps)
    power_error = np.zeros(simulation_timesteps)
    power_error_accumalation = np.zeros(simulation_timesteps)
    heater_profile = np.zeros(simulation_timesteps)
    heater_profile[0] = WaterHeater.heater_on
    
    peak_temperature = []
    minimum_temperature = []
    tap_energy = []
    
    for timestep in range(1, simulation_timesteps):
        temperature[timestep] = WaterHeater.diffusion(temperature[timestep-1], time_step, cutoff[timestep])
        heater_profile[timestep] = WaterHeater.heater_on
        power_step[timestep], power_consumed = WaterHeater.power_check(time_step)
        accumalated_power[timestep] = accumalated_power[timestep-1] + power_consumed 
        
        if tapping_start_time[tap_index] <= timestep * time_step:
            if current_tap_power >= tapping_power[tap_index]:
                peak_temperature.append(float(np.mean(output_temperature[timestep - 1 - current_tap_timesteps: timestep])))
                tap_energy.append(float(np.sum(energy_tapped[timestep - 1 - current_tap_timesteps: timestep])))
            
                tap_index += 1  # Mark the tap as done
                current_tap_power = 0
                current_tap_timesteps = 0
                
            else:
                
                # Store T_m value and incrementing the tap timesteps counter
                if current_tap_timesteps == 0:
                    minimum_temperature.append(float(temperature[timestep, -1]))
                current_tap_timesteps += 1
                    
                output_temperature[timestep] = ((fractional_flow[tap_index] * temperature[timestep, WaterHeater.no_layers - discritized_flow[tap_index]-1]
                                        + sum(temperature[timestep, WaterHeater.no_layers - discritized_flow[tap_index]:WaterHeater.no_layers])) 
                                        / inlet_flow_rate[tap_index])
                
                temperature[timestep] = WaterHeater.tap(temperature[timestep], inlet_flow_rate[tap_index], inlet_temperature[tap_index])
                
                tapped_volume[timestep] = inlet_flow_rate[tap_index] * layer_capacity
                inlet_water_temperature[timestep] = inlet_temperature[tap_index]
                energy_tapped[timestep] = (inlet_flow_rate[tap_index] * layer_capacity * _C * (output_temperature[timestep]
                                                - inlet_water_temperature[timestep])) / 3600000 # Energy tapped in kWh
                current_tap_power += energy_tapped[timestep]
                
        internal_energy_change[timestep] = sum(temperature[timestep] - temperature[timestep-1]) * _C * layer_capacity / 3600000 # Internal Energy change in kWh
        power_error[timestep] = power_step[timestep] - energy_tapped[timestep] -  internal_energy_change[timestep] 
        power_error_accumalation[timestep] = power_error_accumalation[timestep-1] +  power_error[timestep] 
               
    # Total tapped power (as per the standard)
    Qref = sum(tapping_power)
    
    # Total actual tapped power
    Qh2o = sum(energy_tapped)
    
    # Total power provided by the heating element
    Qtestelec = accumalated_power[-1]
    
    # Top of the heater temperature after regulation cycles
    T1 = temperature[0,-1]
    
    # Top of the heater temperature after ERP test
    T2 = temperature[-1,-1]
    
    
    return temperature, Qref, Qh2o, Qtestelec, T1, T2, power_error_accumalation, peak_temperature, minimum_temperature, tap_energy, tapped_volume, output_temperature, energy_tapped, accumalated_power, heater_profile, inlet_water_temperature
                
            
def regulation_cycles(WaterHeater, time_step, initial_temperature, no_cycles):
    """
    Simulates the regulation cycles that are performed at the beginning of tests
    Parameters
    ----------
    WaterHeater : an instance of the WaterHeater class
    time_step : the simulation time step
    initial_temperature : the temperature of the water and the start of the first cycle
    no_cycles : number of regulation cycles to be performed 
    
    Returns
    -------
    The temperature array at every time step throughout the cycles

    """
    time_estimate = math.ceil(4*24*60/time_step)
    temperature = np.zeros((time_estimate, WaterHeater.no_layers))
    temperature[0] = initial_temperature
    no_heater_closing = 0
    regulation_timestep = 1
    while True:
        heater_previous_state = WaterHeater.heater_on
        temperature[regulation_timestep] = WaterHeater.diffusion(temperature[regulation_timestep-1], time_step)
        heater_current_state = WaterHeater.heater_on
        
        regulation_timestep += 1
        if heater_current_state < heater_previous_state:
            no_heater_closing += 1
            if no_heater_closing >= no_cycles and time_step*regulation_timestep > 1440:
                break        
        
    return temperature[:regulation_timestep]

def ERP1(WaterHeater, time_step, tapping_profile, load_profile, initial_temperature, cutoff_temperature, hysteresis, save = False):
    tapping_initiation_file = pd.read_excel(tapping_profile, sheet_name = load_profile)
    regulation_cycles_temperature = regulation_cycles(WaterHeater, time_step, initial_temperature, 2)
   
    ERP_temperature, Qref, Qh2o, Qtestelec, T1, T2, power_error_accumalation, T_p, T_m, Q_tapped, \
        tapped_volume, output_temperature, energy_tapped, accumalated_power, heater_profile, input_temperature = tapping(
            WaterHeater, time_step, regulation_cycles_temperature[-1], tapping_initiation_file)                

    temperature = np.concatenate((regulation_cycles_temperature, ERP_temperature[1:]))
    # time = np.linspace(1440-time_step*(len(temperature)-1), 1440, len(temperature))
    
    Qelec = Qref/Qh2o * (Qtestelec + 1.163/1000*WaterHeater.capacity*(T1-T2))
    Qcor = -0.23*(2.5*(Qelec-Qref))
    efficiency = Qref/(2.5*Qelec+Qcor) * 100
    
    Tp_test = sum(T_p < tapping_initiation_file["Tp"])
    if Tp_test:
        Tp_test = False 
    else:
        Tp_test = True
        
    Tm_test = sum(T_m < tapping_initiation_file["Tm"])
    if Tm_test:
        Tm_test = False 
    else:
        Tm_test = True
        
    print("First Day Basic Results")
    print(f' Qref = {round(Qref, 3)} \n QH2O = {round(Qh2o, 3)} \n QTestElec = {round(Qtestelec, 3)} \n QElec = {round(Qelec, 3)} \n T_1 = {round(T1, 2)} \n T_2 = {round(T2, 2)} \n Qcor = {round(Qcor, 3)}')
    print(f"\n Peak Temperature Test: {Tp_test} \n Minimum Temperature Test: {Tm_test}")
    print(f"Basic Efficiency = {efficiency:.1f}%")
    
    
    plot_results(WaterHeater, time_step, cutoff_temperature, hysteresis, 
                 np.shape(regulation_cycles_temperature)[0], temperature, mode = 'Top')
      

    tapping_results_df = tapping_initiation_file.copy()
    tapping_output = pd.DataFrame({
        'Qtap (kWh)' : Q_tapped,
        'T_m' : T_m,
        'T_p' : T_p})
    
    with pd.ExcelWriter("ERP1 - Tapping Results.xlsx") as writer:
        tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = 'Tapping Results', index=False)
    print("Tapping results saved!")
          
    if save:
        save_results("results.xlsx", "ERP1 Results", ERP_temperature, WaterHeater.no_layers,
                     time_step, tapped_volume, input_temperature, output_temperature, energy_tapped,
                     heater_profile, power_error_accumalation)
        

        
    # plt.plot(time, temperature[:,-1])  
    # # Add plot labels
    # plt.title("Temperature Profile")
    # plt.xlabel("Time (mins)")
    # plt.ylabel("Temperature (C)")
    # plt.show()
    
def ERP2(WaterHeater, time_step, tapping_profile, load_profile1, load_profile2, initial_temperature, cutoff_temperature, hysteresis, smart_parameters):
    
    tapping_initiation_file1 = pd.read_excel(tapping_profile, sheet_name = load_profile1)    
    tapping_initiation_file2 = pd.read_excel(tapping_profile, sheet_name = load_profile2)
    tapping_initiation_offday = pd.read_excel(tapping_profile, sheet_name = 'OFF')
   
    regulation_cycles_temperature = regulation_cycles(WaterHeater, time_step, initial_temperature, 2)
    
    next_day_starting_temp = regulation_cycles_temperature[-1]
    plotting_temp = regulation_cycles_temperature[:,-1]   
    
    temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, plotting_temp = learning_week(WaterHeater, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, next_day_starting_temp, plotting_temp, True)
    
    plotting_cutoff = np.full_like(plotting_temp, cutoff_temperature, dtype=float)
    next_day_starting_temp = temperature[-1,:,6]
    
    plotting_temp, plotting_cutoff, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test = smart_week(WaterHeater, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, next_day_starting_temp, smart_parameters, plotting_temp, plotting_cutoff, True)

    time = np.linspace(20160-time_step*(len(plotting_temp)-1), 20160, len(plotting_temp))
    
    plt.plot(time, plotting_temp, label='Temperature')  # Plot the temperature
    plt.plot(time, plotting_cutoff, 'r--')  # Plot the cutoff as a red dotted line
    plt.plot(time, plotting_cutoff - hysteresis, 'r--')
    
    # Add labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature')
    plt.legend()
    
    Qtestelec_week1 = np.sum(Qtestelec[:, 0])
    Qtestelec_week2 = np.sum(Qtestelec[:, 1])
    SCF = 1 - (Qtestelec_week2/Qtestelec_week1)

    if SCF > 0.007:
        smart = 1 
    else:
        smart = 0        

    Qelec = Qref[0,0]/Qh2o[0,0] * (Qtestelec[0,0] + 1.163/1000* WaterHeater.capacity *(T1[0,0]-T2[0,0]))

    Qcor_basic = -0.23*(2.5*(Qelec-Qref[0,0]))
    Qcor_smart = -0.23*(2.5*(Qelec*(1 - SCF*smart)-Qref[0,0]))

    efficiency_basic = Qref[0,0]/(2.5*Qelec+Qcor_basic) * 100
    efficiency_smart = Qref[0,0]/(2.5*Qelec*(1-SCF*smart) + Qcor_smart) * 100
        
    if sum(Tp_test):
        Tp_test = False 
    else:
        Tp_test = True
        
    if sum(Tm_test):
        Tm_test = False 
    else:
        Tm_test = True

    print("\nFirst Day Basic Results")
    print(f' Qref = {round(Qref[0,0], 3)} \n QH2O = {round(Qh2o[0,0], 3)} \n QTestElec = {round(Qtestelec[0,0], 3)} \n QElec = {round(Qelec, 3)} \n T_1 = {round(T1[0,0], 2)} \n T_2 = {round(T2[0,0], 2)} \n Qcor = {round(Qcor_basic, 3)}')
    print(f"Basic Efficiency = {efficiency_basic:.1f}%")

    print("\nTwo Weeks Smart Results")
    print(f' Qref = {round(Qref[0,0], 3)} \n QElec = {round(Qelec, 3)} \n QTestElec Week 1 = {round(Qtestelec_week1, 3)} \n QTestElec Week 2 = {round(Qtestelec_week2, 3)} \n Smart Correction Factor = {SCF*100:.1f}% \n Smart = {round(smart, 2)} \n Qcor = {round(Qcor_smart, 3)}')
    print(f"\n Peak Temperature Test: {Tp_test} \n Minimum Temperature Test: {Tm_test}")
    print(f"Smart Efficiency = {efficiency_smart:.1f}%")
    

    
    
def learning_week(WaterHeater, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, next_day_starting_temp, plotting_temp = None, save = False):
    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    
    temperature = np.zeros((simulation_timesteps, WaterHeater.no_layers, 14))
    heating_profiles = np.zeros((simulation_timesteps, 7))
    
    Qref = np.zeros((7,2))
    Qh2o = np.zeros((7,2))
    Qtestelec = np.zeros((7,2))
    T1 = np.zeros((7,2))
    T2 = np.zeros((7,2))
    Tp_test = np.zeros((14))
    Tm_test = np.zeros((14))
    
    
    # Learning Week
    for day in range(0,7):
        if day < 5 :
            if day % 2 == 0:
                today_profile = tapping_initiation_file1
            else:
                today_profile = tapping_initiation_file2
        else:
            today_profile = tapping_initiation_offday
            
        temperature[:,:,day], Qref[day,0], Qh2o[day,0], Qtestelec[day,0], T1[day,0], T2[day,0], _, T_p, T_m, Q_tapped, _, _, _, _, heating_profiles[:,day], _ = tapping(WaterHeater, time_step, next_day_starting_temp, today_profile)
        
        Tp_test[day] = sum(T_p < today_profile["Tp"])            
        Tm_test[day] = sum(T_m < today_profile["Tm"])
        
        next_day_starting_temp =  temperature[-1,:,day]
        if plotting_temp is not None:
            plotting_temp = np.hstack((plotting_temp, temperature[:,-1,day]))
        
        
        if save:
            tapping_results_df = today_profile.copy()
            tapping_output = pd.DataFrame({
                'Qtap (kWh)' : Q_tapped,
                'T_m' : T_m,
                'T_p' : T_p})
            
            if day == 0:
                with pd.ExcelWriter("ERP2 - Tapping Results.xlsx") as writer:
                    tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day+1} - Basic', index=False)
                print(f"Day {day+1} tapping results saved!")
            else:   
                with pd.ExcelWriter("ERP2 - Tapping Results.xlsx",mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                    tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day+1} - Basic', index=False)
                print(f"Day {day+1} tapping results saved!")
        
    return temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, plotting_temp


def smart_week(WaterHeater, time_step, 
               tapping_initiation_file1, tapping_initiation_file2, 
               tapping_initiation_offday, temperature, heating_profiles, 
               Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, 
               next_day_starting_temp, smart_parameters,
               plotting_temp = None, plotting_cutoff = None, save = False):
    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    cutoff_temps = np.zeros((simulation_timesteps, 7))
    
    v40_levels = smart_parameters[2:8]
    inlet_tank_cutoffs = smart_parameters[14:20]
    first = smart_parameters[0]
    duration = smart_parameters[1]
    
    
    # Implementation Week
    for day in range(7,14): 
        smart = SmartAlgorithm(time_step, heating_profiles[:,day-7], WaterHeater.power, 0, 0, first, duration, v40_levels, inlet_tank_cutoffs)
        cutoff_temps[:,day-7] = smart.run()
        
        if day < 12 :
            if day % 2 == 1:
                today_profile = tapping_initiation_file1
            else:
                today_profile = tapping_initiation_file2
        else:
            today_profile = tapping_initiation_offday
            
        temperature[:,:,day], Qref[day-7,1], Qh2o[day-7,1], Qtestelec[day-7,1], T1[day-7,1], T2[day-7,1], _, T_p, T_m, Q_tapped, _, _, _, _, _, _ = tapping(WaterHeater, time_step, next_day_starting_temp, today_profile, "Smart", cutoff_temps[:,day-7])
        
        Tp_test[day] = sum(T_p < today_profile["Tp"])            
        Tm_test[day] = sum(T_m < today_profile["Tm"])
        
        next_day_starting_temp =  temperature[-1,:,day]
        
        if plotting_temp is not None:
            plotting_temp = np.hstack((plotting_temp, temperature[:,-1,day]))
            plotting_cutoff = np.hstack((plotting_cutoff, cutoff_temps[:,day-7]))
        
        if save:
            tapping_results_df = today_profile.copy()
            tapping_output = pd.DataFrame({
                'Qtap (kWh)' : Q_tapped,
                'T_m' : T_m,
                'T_p' : T_p})
            
            with pd.ExcelWriter("ERP2 - Tapping Results.xlsx",mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day-6} - Smart', index=False)
            print(f"Day {day+1} tapping results saved!")
            
    return plotting_temp, plotting_cutoff, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test
        
    
def calc_QPR(code, cutoff_temperature, hysteresis, capacity, power  , initial_temperature):
    """
    Parameters
    ----------
    initial_temperature : initial temperature of the WH in degrees Celsius

    Returns
    -------
    QPR per layer in Watts

    """
    # Calculate the QPR and save it in self.QPR in W
    qpr_class = qpr_calc([code, cutoff_temperature, hysteresis, capacity, power  , initial_temperature])
    QPR, _ = qpr_class.predict_qpr()
    
    return QPR 


def flat_regulation_cycles(WaterHeater_out, WaterHeater_in, initial_temperature, time_step, no_cycles):
    time_estimate = math.ceil(4 * 24 * 60 / time_step)
    
    # Combine temperature1 and temperature2 into a single 3D array
    temperatures = np.zeros((2, time_estimate, WaterHeater_out.no_layers))
    
    # Initialize the first timestep for both heaters
    temperatures[0, 0] = initial_temperature[0]
    temperatures[1, 0] = initial_temperature[1]

    no_heater_closing = 0
    regulation_timestep = 1
    while no_heater_closing < no_cycles:
        heater_previous_state = WaterHeater_out.heater_on
        
        # Update the temperature profiles for both heaters
        temperatures[0, regulation_timestep, :WaterHeater_out.no_layers] = WaterHeater_out.diffusion(
            temperatures[0, regulation_timestep - 1, :WaterHeater_out.no_layers], time_step)
        
        temperatures[1, regulation_timestep, :WaterHeater_in.no_layers] = WaterHeater_in.diffusion(
            temperatures[1, regulation_timestep - 1, :WaterHeater_in.no_layers],
            time_step,
            first_tank=WaterHeater_out.heater_on)
        
        heater_current_state = WaterHeater_out.heater_on
        regulation_timestep += 1
        
        if heater_current_state < heater_previous_state:
            no_heater_closing += 1
            
        # while True:
        #     heater_previous_state = WaterHeater_out.heater_on
            
        #     # Update the temperature profiles for both heaters
        #     temperatures[0, regulation_timestep, :WaterHeater_out.no_layers] = WaterHeater_out.diffusion(
        #         temperatures[0, regulation_timestep - 1, :WaterHeater_out.no_layers], time_step)
            
        #     temperatures[1, regulation_timestep, :WaterHeater_in.no_layers] = WaterHeater_in.diffusion(
        #         temperatures[1, regulation_timestep - 1, :WaterHeater_in.no_layers],
        #         time_step,
        #         first_tank=WaterHeater_out.heater_on)
            
        #     heater_current_state = WaterHeater_out.heater_on
        #     regulation_timestep += 1
            
        #     if heater_current_state < heater_previous_state:
        #         no_heater_closing += 1
        #         if no_heater_closing < no_cycles and time_step*regulation_timestep > 1440:
        #             break

        

    # Crop the temperature array to the regulation timestep
    temperatures = temperatures[:, :regulation_timestep, :]
    
    return temperatures


def flat_tapping(WaterHeater_out, WaterHeater_in, time_step, initial_temperature, tapping_initiation_file, smart = None, cutoff = []):

    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    layer_capacity = (WaterHeater_out.capacity/WaterHeater_out.no_layers)
    tapping_start_time = tapping_initiation_file['time (min)']
    inlet_temperature = tapping_initiation_file['Tinlet']
    tapping_power = tapping_initiation_file['Q Tapped (kWh)'] 
    inlet_flow_rate = tapping_initiation_file['flow (L/min)'] * time_step/layer_capacity
    tapping_start_time = np.append(tapping_start_time, np.inf)
    
    if not smart:
        cutoff = np.column_stack((
        np.full(simulation_timesteps, WaterHeater_out.cutoff_temperature),
        np.full(simulation_timesteps, WaterHeater_in.cutoff_temperature)))
            
    tap_index = 0  # Index of the current tap (number of taps done)
    current_tap_power = 0
    current_tap_timesteps = 0
    
    discritized_flow = np.floor(inlet_flow_rate).astype(int)  # Number of layers fully displaced
    fractional_flow = inlet_flow_rate - discritized_flow  # Fractional layer displaced
    
    temperatures = np.zeros((2, simulation_timesteps, WaterHeater_out.no_layers))
    temperatures[0, 0] = initial_temperature[0]
    temperatures[1, 0] = initial_temperature[1]
    tapped_volume = np.zeros((simulation_timesteps, 2))
    output_temperature = np.zeros((simulation_timesteps, 2))
    inlet_water_temperature = np.zeros((simulation_timesteps, 2))
    power_step = np.zeros((simulation_timesteps, 2))
    accumalated_power = np.zeros((simulation_timesteps, 2))
    energy_tapped = np.zeros((simulation_timesteps, 2))
    internal_energy_change = np.zeros((simulation_timesteps, 2))
    power_error = np.zeros((simulation_timesteps, 2))
    power_error_accumalation = np.zeros((simulation_timesteps, 2))
    heater_profile = np.zeros((simulation_timesteps, 2))
    heater_profile[0] = [WaterHeater_out.heater_on, WaterHeater_in.heater_on]
    power_consumed = np.zeros(2)
    
    peak_temperature = []
    minimum_temperature = []
    tap_energy = []
   
    for timestep in range(1, simulation_timesteps):
        temperatures[0, timestep, :WaterHeater_out.no_layers] = WaterHeater_out.diffusion(
            temperatures[0, timestep - 1, :WaterHeater_out.no_layers], time_step, cutoff[timestep, 0])
        
        temperatures[1, timestep, :WaterHeater_in.no_layers] = WaterHeater_in.diffusion(
            temperatures[1, timestep - 1, :WaterHeater_in.no_layers], time_step, cutoff[timestep, 1], WaterHeater_out.heater_on)
        
        heater_profile[timestep] = [WaterHeater_out.heater_on, WaterHeater_in.heater_on]
        power_step[timestep, 0], power_consumed[0] = WaterHeater_out.power_check(time_step)
        power_step[timestep, 1], power_consumed[1] = WaterHeater_in.power_check(time_step)
        accumalated_power[timestep] = accumalated_power[timestep-1] + power_consumed         
        
        
        if tapping_start_time[tap_index] <= timestep * time_step:
            if current_tap_power >= tapping_power[tap_index]:
                peak_temperature.append(float(np.mean(output_temperature[timestep - 1 - current_tap_timesteps: timestep, 1])))
                tap_energy.append(current_tap_power) # float(np.sum(energy_tapped[timestep - 1 - current_tap_timesteps: timestep, 1]))
            
                tap_index += 1  # Mark the tap as done
                current_tap_power = 0
                current_tap_timesteps = 0
                
            else:
                
                # Store T_m value and incrementing the tap timesteps counter
                if current_tap_timesteps == 0:
                    minimum_temperature.append(float(temperatures[1, timestep, -1]))
                current_tap_timesteps += 1
                
                output_temperature[timestep, 0] = ((fractional_flow[tap_index] * temperatures[0, timestep, WaterHeater_out.no_layers - discritized_flow[tap_index]-1]
                                        + sum(temperatures[0, timestep, WaterHeater_out.no_layers - discritized_flow[tap_index]:WaterHeater_out.no_layers])) 
                                        / inlet_flow_rate[tap_index])
                
                output_temperature[timestep, 1] = ((fractional_flow[tap_index] * temperatures[1, timestep, WaterHeater_out.no_layers - discritized_flow[tap_index]-1]
                                        + sum(temperatures[1, timestep, WaterHeater_out.no_layers - discritized_flow[tap_index]:WaterHeater_out.no_layers])) 
                                        / inlet_flow_rate[tap_index])
                
                temperatures[1, timestep, :WaterHeater_in.no_layers] = WaterHeater_in.tap(temperatures[1, timestep, :WaterHeater_in.no_layers], inlet_flow_rate[tap_index], inlet_temperature[tap_index])
                temperatures[0, timestep, :WaterHeater_in.no_layers] = WaterHeater_in.tap(temperatures[0, timestep, :WaterHeater_in.no_layers], inlet_flow_rate[tap_index], output_temperature[timestep, 1])
                
                tapped_volume[timestep, :] = inlet_flow_rate[tap_index] * layer_capacity
                
                inlet_water_temperature[timestep] = [output_temperature[timestep, 1], inlet_temperature[tap_index]]
                energy_tapped[timestep,0] = (inlet_flow_rate[tap_index] * layer_capacity * _C * (output_temperature[timestep,0]
                                                - inlet_water_temperature[timestep,0])) / 3600000 # Energy tapped in kWh
                energy_tapped[timestep,1] = (inlet_flow_rate[tap_index] * layer_capacity * _C * (output_temperature[timestep,1]
                                                - inlet_water_temperature[timestep,1])) / 3600000 # Energy tapped in kWh
                
                current_tap_power += sum(energy_tapped[timestep,:])
                
        internal_energy_change[timestep, 0] = sum(temperatures[0, timestep, :] - temperatures[0, timestep-1, :]) * _C * layer_capacity / 3600000 # Internal Energy change in kWh
        internal_energy_change[timestep, 1] = sum(temperatures[1, timestep, :] - temperatures[1, timestep-1, :]) * _C * layer_capacity / 3600000
        power_error[timestep] = power_step[timestep] - energy_tapped[timestep] -  internal_energy_change[timestep] 
        power_error_accumalation[timestep] = power_error_accumalation[timestep-1] +  power_error[timestep] 
               
    # Total tapped power (as per the standard)
    Qref = sum(tapping_power)
    
    # Total actual tapped power
    Qh2o = sum(tap_energy) # energy_tapped.sum()
    
    # Total power provided by the heating element
    Qtestelec = sum(accumalated_power[-1])
    
    # Top of the heater temperature after regulation cycles
    T1 = temperatures[0, 0,-1]
    
    # Top of the heater temperature after ERP test
    T2 = temperatures[0, -1,-1]
    
    
    
    return temperatures, Qref, Qh2o, Qtestelec, T1, T2, \
        power_error_accumalation, peak_temperature, minimum_temperature,\
            tap_energy, tapped_volume, output_temperature, energy_tapped, \
                accumalated_power, heater_profile, inlet_water_temperature



def flat_ERP1(WaterHeater_out, WaterHeater_in, time_step, tapping_profile, load_profile, initial_temperature, hysteresis, save = False):
    tapping_initiation_file = pd.read_excel(tapping_profile, sheet_name = load_profile)
    regulation_cycles_temperatures = flat_regulation_cycles(WaterHeater_out, WaterHeater_in, initial_temperature, time_step, 3)
   
    ERP_temperatures, Qref, Qh2o, Qtestelec, T1, T2, power_error_accumalation, T_p, T_m, Q_tapped, \
        tapped_volume, output_temperature, energy_tapped, accumalated_power, heater_profile, input_temperature = flat_tapping(WaterHeater_out, WaterHeater_in, time_step, regulation_cycles_temperatures[:,-1,:], tapping_initiation_file)        

    temperatures = np.concatenate((regulation_cycles_temperatures, ERP_temperatures), axis = 1)
    
    Qelec = Qref/Qh2o * (Qtestelec + 1.163/1000*WaterHeater_out.capacity*2*(T1-T2))
    Qcor = -0.23*(2.5*(Qelec-Qref))
    efficiency = Qref/(2.5*Qelec+Qcor) * 100
    
    
    
    Tp_test = sum(T_p < tapping_initiation_file["Tp"])
    if Tp_test:
        Tp_test = False 
    else:
        Tp_test = True
        
    Tm_test = sum(T_m < tapping_initiation_file["Tm"])
    if Tm_test:
        Tm_test = False 
    else:
        Tm_test = True
        
    
    print("First Day Basic Results")
    print(f' Qref = {round(Qref, 3)} \n QH2O = {round(Qh2o, 3)} \n QTestElec = {round(Qtestelec, 3)} \n QElec = {round(Qelec, 3)} \n T_1 = {round(T1, 2)} \n T_2 = {round(T2, 2)} \n Qcor = {round(Qcor, 3)}')
    print(f"\n Peak Temperature Test: {Tp_test} \n Minimum Temperature Test: {Tm_test}")
    print(f" Basic Efficiency = {efficiency:.1f}%")
    
    plt.ioff()

    # Plot results for the first water heater
    plot_results(
        WaterHeater_out,
        time_step,
        WaterHeater_out.cutoff_temperature,
        hysteresis[0],
        np.shape(regulation_cycles_temperatures)[1],
        temperatures[0],
        mode="Top",
    )

    # Get the first figure and axis
    ax1 = plt.gca()

    # Plot results for the second water heater
    plot_results(
        WaterHeater_in,
        time_step,
        WaterHeater_in.cutoff_temperature,
        hysteresis[1],
        np.shape(regulation_cycles_temperatures)[1],
        temperatures[1],
        mode="Top",
    )

    # Get the second figure and axis
    ax2 = plt.gca()

    # Combine plots into a single figure
    fig, ax = plt.subplots()  # Create a new figure
    for line in ax1.lines:  # Copy lines from the first plot
        ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
    for line in ax2.lines:  # Copy lines from the second plot
        ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

    # Customize the combined plot
    ax.set_title("Temperature Profiles")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.axhline(y=WaterHeater_out.cutoff_temperature, color="r", linestyle="--")
    ax.axhline(y=WaterHeater_out.cutoff_temperature - hysteresis[0], color="r", linestyle="--")
    ax.axhline(y=WaterHeater_in.cutoff_temperature, color="b", linestyle="--")
    ax.axhline(y=WaterHeater_in.cutoff_temperature - hysteresis[1], color="b", linestyle="--")

    # Show the combined plot
    plt.show()

    # Re-enable interactive mode
    plt.ion()
       
    tapping_results_df = tapping_initiation_file.copy()
    tapping_output = pd.DataFrame({
        'Qtap (kWh)' : Q_tapped,
        'T_m' : T_m,
        'T_p' : T_p})
    
    with pd.ExcelWriter("ERP1 - Tapping Results.xlsx") as writer:
        tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = 'Tapping Results', index=False)
    print("Tapping results saved!")
    
    if save:
        save_results("results.xlsx", "Outlet Tank", ERP_temperatures[0], WaterHeater_out.no_layers,
                     time_step, tapped_volume[:, 0], input_temperature[:, 0], output_temperature[:, 0],energy_tapped[:, 0],
                     heater_profile[:, 0], power_error_accumalation[:, 0])
        
        save_results("results.xlsx", "Inlet Tank", ERP_temperatures[1], WaterHeater_out.no_layers,
                     time_step, tapped_volume[:, 1], input_temperature[:, 1], output_temperature[:, 1],energy_tapped[:, 1],
                     heater_profile[:, 1], power_error_accumalation[:, 1])
        
    
def flat_ERP2(WaterHeater_out, WaterHeater_in, time_step, tapping_profile, load_profile1, load_profile2, initial_temperature, cutoff_temperature, hysteresis, smart_parameters):
    tapping_initiation_file1 = pd.read_excel(tapping_profile, sheet_name = load_profile1)    
    tapping_initiation_file2 = pd.read_excel(tapping_profile, sheet_name = load_profile2)
    tapping_initiation_offday = pd.read_excel(tapping_profile, sheet_name = 'OFF')
    
    regulation_cycles_temperatures = flat_regulation_cycles(WaterHeater_out, WaterHeater_in, initial_temperature, time_step, 3)
    
    next_day_starting_temp = regulation_cycles_temperatures[:,-1,:]
    plotting_temp = regulation_cycles_temperatures[0, :,-1]
    
    temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, plotting_temp = flat_learning_week(WaterHeater_out, WaterHeater_in, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, next_day_starting_temp, plotting_temp, True)
    
    plotting_cutoff = np.full_like(plotting_temp, cutoff_temperature[0], dtype=float)
    next_day_starting_temp = temperature[:, -1, :, 6]
     
    plotting_temp, plotting_cutoff, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test = flat_smart_week(WaterHeater_out, WaterHeater_in, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, next_day_starting_temp, smart_parameters, plotting_temp, plotting_cutoff, True)
    
    time = np.linspace(20160-time_step*(len(plotting_temp)-1), 20160, len(plotting_temp))
    
    plt.plot(time, plotting_temp, label='Temperature')  # Plot the temperature
    plt.plot(time, plotting_cutoff, 'r--')  # Plot the cutoff as a red dotted line
    plt.plot(time, plotting_cutoff - hysteresis[0], 'r--')
    
    # Add labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature')
    plt.legend()
    
    Qtestelec_week1 = np.sum(Qtestelec[:, 0])
    Qtestelec_week2 = np.sum(Qtestelec[:, 1])
    SCF = 1 - (Qtestelec_week2/Qtestelec_week1)

    if SCF > 0.007:
        smart = 1 
    else:
        smart = 0        

    Qelec = Qref[0,0]/Qh2o[0,0] * (Qtestelec[0,0] + 1.163/1000* WaterHeater_out.capacity*2*(T1[0,0]-T2[0,0]))

    Qcor_basic = -0.23*(2.5*(Qelec-Qref[0,0]))
    Qcor_smart = -0.23*(2.5*(Qelec*(1 - SCF*smart)-Qref[0,0]))

    efficiency_basic = Qref[0,0]/(2.5*Qelec+Qcor_basic) * 100
    efficiency_smart = Qref[0,0]/(2.5*Qelec*(1-SCF*smart) + Qcor_smart) * 100
    
    if sum(Tp_test) > 0:
        Tp_test = False 
    else:
        Tp_test = True
        
    if sum(Tm_test) > 0:
        Tm_test = False 
    else:
        Tm_test = True

    print("\nFirst Day Basic Results")
    print(f' Qref = {round(Qref[0,0], 3)} \n QH2O = {round(Qh2o[0,0], 3)} \n QTestElec = {round(Qtestelec[0,0], 3)} \n QElec = {round(Qelec, 3)} \n T_1 = {round(T1[0,0], 2)} \n T_2 = {round(T2[0,0], 2)} \n Qcor = {round(Qcor_basic, 3)}')
    print(f"Basic Efficiency = {efficiency_basic:.1f}%")

    print("\nTwo Weeks Smart Results")
    print(f' Qref = {round(Qref[0,0], 3)} \n QElec = {round(Qelec, 3)} \n QTestElec Week 1 = {round(Qtestelec_week1, 3)} \n QTestElec Week 2 = {round(Qtestelec_week2, 3)} \n Smart Correction Factor = {SCF*100:.1f}% \n Smart = {round(smart, 2)} \n Qcor = {round(Qcor_smart, 3)}')
    print(f"\n Peak Temperature Test: {Tp_test} \n Minimum Temperature Test: {Tm_test}")
    print(f"Smart Efficiency = {efficiency_smart:.1f}%")
    
    
def flat_learning_week(WaterHeater_out, WaterHeater_in, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, next_day_starting_temp, plotting_temp = None, save = False):
    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    
    temperature = np.zeros((2, simulation_timesteps, WaterHeater_out.no_layers, 14))
    heating_profiles = np.zeros((simulation_timesteps, 2, 7))
    
    Qref = np.zeros((7,2))
    Qh2o = np.zeros((7,2))
    Qtestelec = np.zeros((7,2))
    T1 = np.zeros((7,2))
    T2 = np.zeros((7,2))
    Tp_test = np.zeros(14)
    Tm_test = np.zeros((14))
    
    # Learning Week
    for day in range(0,7):
        if day < 5 :
            if day % 2 == 0:
                today_profile = tapping_initiation_file1
            else:
                today_profile = tapping_initiation_file2
        else:
            today_profile = tapping_initiation_offday
        
        temperature[:, :, :, day], Qref[day,0], Qh2o[day,0], Qtestelec[day,0], T1[day,0], T2[day,0], _, T_p, T_m, Q_tapped, _, _, _, _, heating_profiles[:,:,day], _ = flat_tapping(WaterHeater_out, WaterHeater_in, time_step, next_day_starting_temp, today_profile) 
        
        Tp_test[day] = sum(T_p < today_profile["Tp"])            
        Tm_test[day] = sum(T_m < today_profile["Tm"])
        
        next_day_starting_temp =  temperature[:, -1, :, day]
        
        if plotting_temp is not None:
            plotting_temp = np.hstack((plotting_temp, temperature[0, :, -1, day]))
    
        
        if save:
            tapping_results_df = today_profile.copy()
            tapping_output = pd.DataFrame({
                'Qtap (kWh)' : Q_tapped,
                'T_m' : T_m,
                'T_p' : T_p})
            
            if day == 0:
                with pd.ExcelWriter("ERP2 - Tapping Results.xlsx") as writer:
                    tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day+1} - Basic', index=False)
                print(f"Day {day+1} tapping results saved!")
            else:   
                with pd.ExcelWriter("ERP2 - Tapping Results.xlsx",mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                    tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day+1} - Basic', index=False)
                print(f"Day {day+1} tapping results saved!")
                
    return temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, plotting_temp



def flat_smart_week(WaterHeater_out, WaterHeater_in, time_step, tapping_initiation_file1, tapping_initiation_file2, tapping_initiation_offday, temperature, heating_profiles, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test, next_day_starting_temp, smart_parameters, plotting_temp = None, plotting_cutoff = None, save = False):
    simulation_time = 1440
    simulation_timesteps = math.ceil(simulation_time/time_step)
    
    cutoff_temps = np.zeros((simulation_timesteps, 2, 7))
    
    v40_levels = smart_parameters[0]
    inlet_tank_cutoffs = smart_parameters[1]
    
    first = smart_parameters[3]
    duration = smart_parameters[4]
    
    first = smart_parameters[0]
    duration = smart_parameters[1]
    v40_levels = smart_parameters[2:8]
    inlet_tank_cutoffs = smart_parameters[8:14]
    #outlet_tank_cutoffs = smart_parameters[14:20]
    
    
    # Implementation Week
    for day in range(7,14): 
        smart = SmartAlgorithm(time_step, heating_profiles[:,:,day-7], 0, WaterHeater_out.power, WaterHeater_in.power, first, duration, v40_levels, inlet_tank_cutoffs, 0)
        cutoff_temps[:,:,day-7] = smart.run()
        
        if day < 12 :
            if day % 2 == 1:
                today_profile = tapping_initiation_file1
            else:
                today_profile = tapping_initiation_file2
        else:
            today_profile = tapping_initiation_offday
        
        temperature[:, :, :, day], Qref[day-7,1], Qh2o[day-7,1], Qtestelec[day-7,1], T1[day-7,1], T2[day-7,1], _, T_p, T_m, Q_tapped, _, _, _, _, _, _ = flat_tapping(WaterHeater_out, WaterHeater_in, time_step, next_day_starting_temp, today_profile, "Smart", cutoff_temps[:, :,day-7]) 
        
        Tp_test[day] = sum(T_p < today_profile["Tp"])            
        Tm_test[day] = sum(T_m < today_profile["Tm"])
        
        next_day_starting_temp =  temperature[:, -1, :, day]
        plotting_temp = np.hstack((plotting_temp, temperature[0, :, -1, day]))
        plotting_cutoff = np.hstack((plotting_cutoff, cutoff_temps[:, 0, day-7]))
        
        tapping_results_df = today_profile.copy()
        tapping_output = pd.DataFrame({
            'Qtap (kWh)' : Q_tapped,
            'T_m' : T_m,
            'T_p' : T_p})
        
        with pd.ExcelWriter("ERP2 - Tapping Results.xlsx",mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
            tapping_results_df.join(tapping_output, how = 'right').to_excel(writer, sheet_name = f'Day {day-6} - Smart', index=False)
        print(f"Day {day+1} tapping results saved!")
        
    return plotting_temp, plotting_cutoff, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test


def optimization(WaterHeater, time_step, tapping_profile, load_profile1, load_profile2, initial_temperature, cutoff_temperature, hysteresis, method):
    # ========================== Optimization Parameters ==========================
    #initial_grid_size = 5
    #v40 = 75
    #no_combinations = 4
    no_iterations = 3
    #k = 3
    #refinement_factor = 2
    punishment_const = 7.5
    # =============================================================================

    tapping_initiation_file1 = pd.read_excel(tapping_profile, sheet_name=load_profile1)
    tapping_initiation_file2 = pd.read_excel(tapping_profile, sheet_name=load_profile2)
    tapping_initiation_offday = pd.read_excel(tapping_profile, sheet_name='OFF')

    regulation_cycles_temperature = regulation_cycles(WaterHeater, time_step, initial_temperature, 2)
    next_day_starting_temp = regulation_cycles_temperature[-1]

    temperature_const, heating_profiles_const, Qref_const, Qh2o_const, Qtestelec_const, T1_const, T2_const, Tp_test_const, Tm_test_const, _ = learning_week(
        WaterHeater, time_step, tapping_initiation_file1, tapping_initiation_file2, 
        tapping_initiation_offday, next_day_starting_temp
    )
    
    next_day_starting_temp = temperature_const[-1, :, 6]
    Qelec = Qref_const[0, 0] / Qh2o_const[0, 0] * (Qtestelec_const[0, 0] + 1.163 / 1000 * WaterHeater.capacity * (T1_const[0, 0] - T2_const[0, 0]))
    Qcor_basic = -0.23 * (2.5 * (Qelec - Qref_const[0, 0]))
    efficiency_basic = Qref_const[0, 0] / (2.5 * Qelec + Qcor_basic) * 100

    print(f"\nFirst Day Basic Results\nQref = {round(Qref_const[0,0], 3)}\nQH2O = {round(Qh2o_const[0,0], 3)}\nQTestElec = {round(Qtestelec_const[0,0], 3)}\n"
          f"QElec = {round(Qelec, 3)}\nT_1 = {round(T1_const[0,0], 2)}\nT_2 = {round(T2_const[0,0], 2)}\nQcor = {round(Qcor_basic, 3)}")
    print(f"Basic Efficiency = {efficiency_basic:.1f}%")

    if sum(Tp_test_const) > 0 and sum(Tm_test_const) > 0:
        print("Learning week failed: Both Tp and Tm tests failed.")
        sys.exit()
    elif sum(Tp_test_const) > 0:
        print("Learning week failed: Tp test failed.")
        sys.exit()
    elif sum(Tm_test_const) > 0:
        print("Learning week failed: Tm test failed.")
        sys.exit()
    else:
        print("Learning week passed both tests\n")
        
    for iteration in range(no_iterations):
        if iteration == 0:
            ga = GeneticAlgorithm(initial_pop_size=3)
            current_comb = ga.generate_initial_population()
            print(current_comb)
            current_generational_data = pd.DataFrame(columns=['Individual', 'Efficiency'])
        else:
            current_generational_data.sort_values(by="Efficiency", ascending=False)
            current_comb = ga.crossover(current_generational_data["Individual"].iloc[0], 
                                      current_generational_data["Individual"].iloc[1])
            print("current_comb", current_comb)

        for index, comb in enumerate(current_comb):

            _, _, Qref, Qh2o, Qtestelec, T1, T2, Tp_test, Tm_test = smart_week(
                WaterHeater, time_step, tapping_initiation_file1, tapping_initiation_file2,
                tapping_initiation_offday, temperature_const, heating_profiles_const, Qref_const, Qh2o_const,
                Qtestelec_const, T1_const, T2_const, Tp_test_const, Tm_test_const, next_day_starting_temp, comb
            )
            
        
            Qtestelec_week1 = np.sum(Qtestelec[:, 0])
            Qtestelec_week2 = np.sum(Qtestelec[:, 1])
            SCF = 1 - (Qtestelec_week2 / Qtestelec_week1)
            smart = 1 if SCF > 0.007 else 0

            Qelec = Qref[0, 0] / Qh2o[0, 0] * (Qtestelec[0, 0] + 1.163 / 1000 * WaterHeater.capacity * (T1[0, 0] - T2[0, 0]))
            Qcor_smart = -0.23 * (2.5 * (Qelec * (1 - SCF * smart) - Qref[0, 0]))
            efficiency_smart = Qref[0, 0] / (2.5 * Qelec * (1 - SCF * smart) + Qcor_smart) * 100

            punishment = 2*punishment_const if sum(Tp_test) > 0 and sum(Tm_test) > 0 else punishment_const if sum(Tp_test) > 0 or sum(Tm_test) > 0 else 0
            eta = efficiency_smart - punishment

            current_generational_data.at[index, "Individual"] = comb
            current_generational_data.at[index, "Efficiency"] = eta

        if iteration == 0:
            full_generations_data = current_generational_data.copy()
        else:
            full_generations_data = pd.concat([full_generations_data, current_generational_data], ignore_index=True)
        
        max_efficiency = full_generations_data["Efficiency"].max()
        print(current_generational_data)
        print(f"After iteration {iteration + 1}, maximum efficiency: {max_efficiency:.1f}%")

    full_generations_data = full_generations_data.sort_values(by="Efficiency", ascending=False)
    full_generations_data.to_excel("Optimization Results.xlsx")
    print("Results saved in Optimiation Results.xlsx")