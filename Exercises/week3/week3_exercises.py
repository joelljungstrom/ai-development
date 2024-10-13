import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# 1
country_dict = {
    'country_name': [], 
    'population': [], 
    'area': [],
    'continent': []
    }

country_df = pd.DataFrame(country_dict)
#print(country_dict)
#print(country_df)

# 2
sample_data = pd.read_csv('ai-development/Exercises/week3/sample_data0.csv')
#print(sample_data)

# 3 
department_median_salary = sample_data.groupby('Department')['Salary'].median().reset_index()
#print(department_median_salary)

# 4
best_performance_city = sample_data.groupby('City')['Performance_Score'].idxmax()
best_performing_people_city = sample_data.loc[best_performance_city]
#print(best_performing_people_city)

# 5
sample_data['Years_Experience'] = sample_data['Years_Experience'].replace(0, np.nan)
sample_data['Salary_Per_Experience_Year'] = sample_data['Salary'] / sample_data['Years_Experience']
#print(sample_data)

# 6 
reformat_sample_data = pd.melt(
    sample_data, 
    id_vars=['Name', 'Age', 'City', 'Department', 'Performance_Score', 'Training_Hours', 'Satisfaction_Level', 'Promotion_Eligible'], 
    value_vars=['Salary', 'Performance_Score'], 
    var_name='Metric', 
    value_name='Value'
    )

#print(reformat_sample_data)

# 7
pivot_sample_data = pd.melt(
    sample_data, 
    id_vars=['City', 'Department'], 
    value_vars=['Salary'], 
    var_name='Metric', 
    value_name='Value',
    ignore_index=True
    )

mean_salaries = pivot_sample_data.groupby(['City', 'Department'])['Value'].mean().reset_index()
mean_salaries.columns = ['City', 'Department', 'Mean_Salary']

#print(mean_salaries)

# 8
matrix = np.random.randint(10,size=(3,3))
# print(matrix)

# 9
matrix_1 = np.random.randint(10,size=(4,4))
matrix_2 = np.random.randint(10,size=(4,4))
matrix_3 = matrix_1 * matrix_2

# print(matrix_3)

# 10 
matrix_a = np.random.randint(10,size=(2,2))
matrix_b = np.random.randint(10,size=(1,2)).reshape(2,)

lin_eq = np.linalg.solve(matrix_a, matrix_b)

#print(lin_eq)

# 11
binomial_array = np.random.binomial(n=10, p=0.5, size=1000)

#print(binomial_array)

# 12
identity_array = np.identity(5, dtype=float)
#print(identity_array)
new_diagonal = np.random.randint(5, size=5)
#print(new_diagonal)
np.fill_diagonal(identity_array, new_diagonal)
#print(identity_array)

# 13
sample_data['Experience_Category'] = pd.cut(sample_data['Years_Experience'], bins=[0, 5, 10, 20, 30], labels=['Entry', 'Mid', 'Senior', 'Expert'])
experience_per_city = sample_data.groupby(['Experience_Category', 'City'])['Name'].count().reset_index()
#print(experience_per_city)
'''
sns.displot(
    sample_data, 
    x='City', 
    hue='Experience_Category', 
    multiple='stack')
plt.show()
'''
'''
plt.figure(figsize=(12,6))
sns.barplot(
    data=experience_per_city.stack(), 
    x='City', 
    y='Name', 
    hue='Experience_Category'
)
plt.show()
'''

# 14
'''
sns.pairplot(
    sample_data
)
plt.show()
'''

# 15
'''
sns.violinplot(
    data=sample_data,
    x='Department',
    y='Performance_Score'
)
plt.show()
'''
# 16
employees_per_department = sample_data.groupby('Department')['Name'].count().reset_index()
employees_per_department = employees_per_department.rename(columns={'Name': 'Employees'})
#print(employees_per_department)
'''
plt.pie(
    x=employees_per_department['Employees'],
    labels=employees_per_department['Department']
)
plt.show()
'''

# 17 
'''
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0, 0].plot(sample_data['Performance_Score'], sample_data['Training_Hours'])
axs[0, 0].set_title('Performance by Training h')
axs[0, 0].set_ylabel('Hours')

axs[0, 1].scatter(sample_data['Years_Experience'], sample_data['Salary'])
axs[0, 1].set_title('Salary by Experience')
axs[0, 1].set_ylabel('$ Annually')

axs[1, 0].bar(sample_data['Age'], sample_data['Training_Hours'])
axs[1, 0].set_title('Age by Training h')
axs[1, 0].set_ylabel('Hours')

axs[1, 1].hist(sample_data['Salary'], bins=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
axs[1, 1].set_title('Salary Histogram')
axs[1, 1].set_ylabel('$ Annually')
'''

# 18
numeric_variables = sample_data[['Age', 'Performance_Score', 'Salary', 'Salary_Per_Experience_Year', 'Training_Hours', 'Years_Experience']]
sample_data_correlation = numeric_variables.corr()
'''
sns.heatmap(sample_data_correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
'''

# 19 
random_integers = np.random.randint(100, size=1000)
random_integers = pd.DataFrame(random_integers)
'''
random_integers.hist(bins=20)
plt.title('Histogram of random values')
plt.show()
'''

# 20
salary_per_city_department = sample_data.groupby(['City', 'Department'])['Salary'].mean().reset_index()
pivot_salary_data = salary_per_city_department.pivot_table(index='City', columns='Department', values='Salary')
pivot_salary_data.reset_index(inplace=True)
#print(salary_per_city_department)
#print(pivot_salary_data)
'''
pivot_salary_data.plot(
    x='City',
    kind='bar',
    stacked=False
    )
plt.show()
'''

# 21
sample_data = sample_data.dropna()
X = sample_data[['Age', 'Years_Experience', 'Performance_Score']]
y = sample_data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 22
model = LinearRegression()
model.fit(X_train[['Years_Experience']], y_train)

predictions = model.predict(X_train[['Years_Experience']])

plt.figure(figsize=(12, 6))
plt.scatter(X_train['Years_Experience'], y_train, color='blue', label='Data Points')

X_range = np.linspace(X_train['Years_Experience'].min(), X_train['Years_Experience'].max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)

plt.plot(X_range, y_range, color='red', label='Regression Line', linewidth=2)

plt.title('Salary vs Years of Experience with Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
