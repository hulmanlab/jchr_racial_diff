#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:56:35 2023

@author: au605715
"""

import pickle


# Specify the file path
file_path = "/Users/au605715/Documents/GitHub/jchr_racial_diff/Data/processed_data/slet1.pkl"



# Read from file
with open(file_path, 'rb') as file:
    dictionary = pickle.load(file)


PtID = 3
percentage = 100


ptid_training_w = dictionary[PtID, percentage]['y_pred_tlw_w']
ptid_training_b = dictionary[PtID, percentage]['y_pred_tlb_w']
