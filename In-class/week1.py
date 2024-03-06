import numpy as np
import pandas as pd

# Numpy
data = np.array([1, 2, 3, 4, 5])

print(data)
print(data[0])
print(data[0:2])

# Pandas

dataTwo = [     {"Name": "Alice", "Age": 30, "City": "New York"},     {"Name": "Bob", "Age": 25, "City": "Paris"},     {"Name": "Charlie", "Age": 35, "City": "London"} ] 
# Accessing data requires looping or list comprehensionsprint([person["Name"] for person in data])  # ['Alice', 'Bob', 'Charlie']


 
dataTwo = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [30, 25, 35],
    "City": ["New York", "Paris", "London"]
})
 
# Viewing data is straightforward
print(dataTwo["Name"])
# 0      Alice
# 1        Bob
# 2    Charlie
# Name: Name, dtype: object
