import pandas as pd
import numpy as np  
df = pd.DataFrame(np.array([[1,2,3], [4,5,6]]))
print(df)
print(df.shape)
print("____________________")
print(df.iloc[1,:])