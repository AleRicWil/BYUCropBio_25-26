import numpy
import matplotlib.pyplot as plt
import pandas as pd


with open('filtered_flower_data.csv', 'r') as file:
    data = pd.read_csv(file)
    