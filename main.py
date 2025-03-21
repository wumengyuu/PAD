import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px

# Load the data
df = pd.read_csv('forestfires.csv')

df.info()
df.describe()
