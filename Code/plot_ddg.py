import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ddg_data = pd.read_csv('calculated_ddg.csv')

plt.scatter(ddg_data['experimental_ddg'], ddg_data['calculated_ddg'])
# Add a line to show the perfect correlation
min_x = min(ddg_data['experimental_ddg'].tolist()+ddg_data['calculated_ddg'].tolist())
max_x = max(ddg_data['experimental_ddg'].tolist()+ddg_data['calculated_ddg'].tolist())
plt.plot([min_x, max_x], [min_x, max_x],"--", color='black')
# linear regression
m, b = np.polyfit(ddg_data['experimental_ddg'], ddg_data['calculated_ddg'], 1)
plt.plot([min_x, max_x], [m*min_x+b, m*max_x+b], color='red')
# calculate the correlation coefficient
correlation = ddg_data['experimental_ddg'].corr(ddg_data['calculated_ddg'])
plt.text(min_x, max_x, f'Correlation: {correlation:.2f}', color='black')
# Add labels
plt.xlabel('Experimental ddG')
plt.ylabel('Calculated ddG')
plt.title('Calculated vs Experimental ddG')
plt.legend(['Samples','Perfect correlation', 'Linear regression'])
plt.savefig('../ddg_plot.png')


