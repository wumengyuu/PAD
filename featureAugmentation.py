import pandas as pd
import numpy as np
import math

# Load the CSV
df = pd.read_csv('forestfires.csv')


# Function to calculate BUI
def calculate_bui(dmc, dc):
    if dmc == 0 and dc == 0:
        return 0
    if dmc <= 0.4 * dc:
        return (0.8 * dmc * dc) / (dmc + 0.4 * dc + 1e-6)  # avoid div by 0
    else:
        return dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc + 1e-6)) * (0.92 + (0.0114 * dmc) ** 1.7)


# Function to calculate FWI
def calculate_fwi(isi, bui):
    if bui <= 80.0:
        fD = 0.626 * bui ** 0.809 + 2
    else:
        fD = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))

    b = 0.1 * isi * fD
    if b <= 1:
        return b
    else:
        return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647)


# Compute BUI and FWI columns
df['BUI'] = df.apply(lambda row: calculate_bui(row['DMC'], row['DC']), axis=1)
df['FWI'] = df.apply(lambda row: calculate_fwi(row['ISI'], row['BUI']), axis=1)

# Round to 1 decimal place
df['BUI'] = df['BUI'].round(1)
df['FWI'] = df['FWI'].round(1)

# Save updated CSV
df.to_csv('forestfires_with_fwi_bui.csv', index=False)

print("FWI and BUI columns added and saved to 'forestfires_with_fwi_bui.csv'.")
