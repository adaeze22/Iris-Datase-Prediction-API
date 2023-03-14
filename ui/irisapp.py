#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:57:14 2023

@author: ada-eze
"""

import numpy as np 
import pandas as pd
from joblib import load
import streamlit as st 


model = load('../model/iris_model.joblib')


### This is the function/method that handles the prediction
def prediction(sepallength, sepalwidth, petallength, petalwidth):
    prediction = model.predict(np.array([[sepallength, sepalwidth, petallength, petalwidth]]))
    
    return prediction


# function to create the ui
def main():
    st.title("Flower Prediction")
    st.header("Iris Dataset")
    
    sepallength = st.number_input('Enter Sepal Length: ')
    sepalwidth = st.number_input('Enter Sepal Width: ')
    petallength = st.number_input('Enter Petal Length: ')
    petalwidth = st.number_input('Enter Petal Width: ')
    
    button = st.button('Predict')
    
    result = ''
    
    
    if (button):
        result = prediction(sepallength, sepalwidth, petallength, petalwidth)
        st.write('predicting....')
        if result == 0:
            st.success('This is a Setosa')
            
        elif result == 1:
            st.success('This is Versicolor')
        
        elif result ==2:
            st.success('This is Virginica')
            
        else:
            st.success('This flower does not exist')

if __name__ == '__main__':
    main()