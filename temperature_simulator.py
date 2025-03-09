from helper_functions import WaterHeater, ERP1, ERP2, calc_QPR, flat_ERP1, flat_ERP2, optimization
import pandas as pd
import time

start_time = time.time() 

initiation_file = pd.read_excel("initiation_file.xlsx")

_power = initiation_file['Power (Watts)']
_cutoff_temperature = initiation_file['Cutoff Temp (C)']
_hysteresis = initiation_file['Hysteresis (C)']
_initial_temperature = initiation_file['Initial Temp (C)']
_heating_element_start = initiation_file['Heater Start (mm)']
_heating_element_end = initiation_file['Heater End (mm)']
_thermostat_start = initiation_file['Thermostat Start (mm)']
_thermostat_end = initiation_file['Thermostat End (mm)']

_time_step = initiation_file['Time Step (min)'][0]
_load_profile = initiation_file['Load Profile'][0]
_secondary_LP = initiation_file['Secondary LP'][0]
_capacity = initiation_file['Capacity (L)'][0]
_diameter = initiation_file['Diameter (mm)'][0]
_code = initiation_file['Code'][0]
_test = initiation_file['ERP Level'][0]
_no_layers = int(initiation_file['Number of Layers'][0])
_optimize = 1

# Diffusion Parameters
_layer_decrease_ratio = 1
_top_temp_max = 70
_top_temp_min = 10
_no_affected_layers_max = 6
_no_affected_layers_min = 9

_diffusion_parameter = [_layer_decrease_ratio, _top_temp_max, _top_temp_min, _no_affected_layers_max, _no_affected_layers_min]

# Smart Parameters
_v40_levels = [6, 17, 24, 30, 325, 400]
_inlet_tank_cutoffs = [44, 50, 52, 54, 65, 75]
_outlet_tank_cutoffs = [44, 50, 52, 54, 65, 75]
_first = 4 
_duration = 3

_smart_parameters = [_first, _duration, _v40_levels, _inlet_tank_cutoffs, _outlet_tank_cutoffs]
_smart_parameters =  sum([i if isinstance(i, list) else [i] for i in _smart_parameters], [])


_QPR = 0.97 #calc_QPR(_code, _cutoff_temperature[0], _hysteresis[0], _capacity, _power[0], _initial_temperature[0])

if _code == "Flat":
    _QPR /= 2
    _capacity /= 2
    Heater2 = WaterHeater(_initial_temperature[1], _no_layers, _QPR, _capacity, _diameter,
                          _power[1], _cutoff_temperature[1], _hysteresis[1], _heating_element_start[1],
                          _heating_element_end[1], _thermostat_start[1], _thermostat_end[1], _diffusion_parameter)

Heater1 = WaterHeater(_initial_temperature[0], _no_layers, _QPR, _capacity, _diameter,
                      _power[0], _cutoff_temperature[0], _hysteresis[0], _heating_element_start[0],
                      _heating_element_end[0], _thermostat_start[0], _thermostat_end[0], _diffusion_parameter)

if _optimize:
    if _code == "Flat":
        print("Unavailable")
    else:
        optimization(Heater1, _time_step, "tapping_profile.xlsx", _load_profile, _secondary_LP, _initial_temperature[0], _cutoff_temperature[0], _hysteresis[0], _smart_parameters)
else:
    if _code == "Flat":
        if _test == 1:
            flat_ERP1(Heater1, Heater2, _time_step, "tapping_profile.xlsx", _load_profile, _initial_temperature, _hysteresis, False)
        elif _test == 2:    
            flat_ERP2(Heater1, Heater2, _time_step, "tapping_profile.xlsx", _load_profile, _secondary_LP, _initial_temperature, _cutoff_temperature, _hysteresis, _smart_parameters)
            
    else:
        if _test == 1:
            ERP1(Heater1, _time_step, "tapping_profile.xlsx", _load_profile, _initial_temperature[0], _cutoff_temperature[0], _hysteresis[0], False)
        elif _test == 2:
            ERP2(Heater1, _time_step, "tapping_profile.xlsx", _load_profile, _secondary_LP, _initial_temperature[0], _cutoff_temperature[0], _hysteresis[0], _smart_parameters)

end_time = time.time() 
elapsed_time = (end_time - start_time)/60
print(f"\nExecution Time: {elapsed_time:.1f} minutes")