import os
import pandas as pd
master_df = pd.DataFrame()
f1 = pd.read_csv('patient-characteristics-survey-pcs-2013-1.csv')
f2 = pd.read_csv('patient-characteristics-survey-pcs-2017-1.csv')
master_df = master_df.append(f1)
master_df = master_df.append(f2)
master_df.to_csv('Data_Tuto_2013_2017.csv', index=False)

# Path: merge csv files .py
