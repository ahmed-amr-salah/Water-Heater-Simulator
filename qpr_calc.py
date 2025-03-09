# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:14:58 2024

@author: abdal
"""

import joblib
import numpy as np
import pandas as pd

class qpr_calc(object):
    def __init__(self, inputs):
        super(qpr_calc, self).__init__()
        self.inputs = inputs
        self.model_filename = 'gpr_model.pkl'
        self.code_lookup_table = 'code.xlsx'
        self.lab_lookup_table = 'QPR.xlsx'
        
        self.model = self.load_model()
        self.lookup_table = self.load_table()
        
        if isinstance(self.inputs[0], str):
            foam_volume = self.get_foamVolume_from_code()
            self.inputs.pop(0)
            self.inputs = [foam_volume] + self.inputs
            self.model_inputs = np.array(self.inputs).reshape(1,-1)
        else:
            self.model_inputs = np.array(self.inputs).reshape(1,-1)
        
        
    def load_model(self):
        return joblib.load(self.model_filename)
    
    def load_table(self):
        raw_data = pd.read_excel(self.lab_lookup_table)
        raw_data = raw_data.dropna().reset_index(drop=True)
        code = pd.read_excel(self.code_lookup_table)
        
        return pd.merge(raw_data, code[['Foam Volume', 'Code', 'Capacity']], on=['Code','Capacity'], how='left')
    
    def get_foamVolume_from_code(self):
        code_to_match, capacity_to_match, power_to_match = self.inputs[0], self.inputs[3], self.inputs[4]
        matched_data = self.lookup_table[(self.lookup_table['Code']==code_to_match) &
                                         (self.lookup_table['Capacity']==capacity_to_match) &
                                         (self.lookup_table['Nominal power']==power_to_match)]
        if matched_data.empty:
            print('There is no water heater with this specifications')
        else:
            return matched_data['Foam Volume'].to_numpy()[0]
        
    def predict_qpr(self):
        return self.model.predict(self.model_inputs, return_std=True)
    

QPR = qpr_calc(['PC300', 67.6, 7.6, 15, 1600, 20.53]) #the input has to be a list in the following order [Code/Foamvoluem, cutoffTemp, hystersis, capacity, power, ambient_temp(same as initial)]
kk, std = QPR.predict_qpr() #gives a mean and standard deviation which can be used for confidence intervals in the QPR
    
