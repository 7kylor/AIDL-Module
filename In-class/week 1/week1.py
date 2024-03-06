import numpy as np
import pandas as pd

# Numpy
data = np.array([1, 2, 3, 4, 5])

print(data)
print(data[0])
print(data[0:2])

# Pandas

dataTwo = [     
           {"Name": "Alice", "Age": 30, "City": "New York"},    
           {"Name": "Bob", "Age": 25, "City": "Paris"},     
           {"Name": "Charlie", "Age": 35, "City": "London"} ] 


 
dataTwo = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [30, 25, 35],
    "City": ["New York", "Paris", "London"]
})
 

print(dataTwo["Name"])

