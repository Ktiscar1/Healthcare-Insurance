# Healthcare-Insurance

This project aims to develop a robust machine learning pipeline to analyze demographic and health indicators to estimate what variables affect the increase cost of insurance. The primary focus is to move beyond descriptive statistics and build an automated system capable of quantifying financial risk for individual for increased health insurance.

# Approach

1. Data Cleaning
2. Perform EDA
3. Machine Learning Modeling
4. Final Interpretation

# Data Cleaning

An statistical summary is performed to find the mean, median, and standard deviation

<img width="615" height="423" alt="image" src="https://github.com/user-attachments/assets/f035c42e-2daa-43c9-ac97-436b75ae7504" />

Before performing an EDA is important to clean the data in order to ensure accuracy and reliability. The data was cleaned performing a check for null values and duplicates and the unique value for each column is checked.

```python
dataset.isnull().sum()

dataset.nunique()
```

<img width="172" height="358" alt="image" src="https://github.com/user-attachments/assets/02997741-ce86-480a-b1df-e62cab151f8e" />
<img width="181" height="365" alt="image" src="https://github.com/user-attachments/assets/b71d84b1-a008-4f32-b859-cebf23e8d2d8" />

# Exploratory Data Analysis

Plots are created to better comprehend and describe the data, which enables to create a robust and more comprehensive model. Creating a pairplot to pairwise relationships between variables in the dataset.

[!Pairplot Smoker](https://github.com/Ktiscar1/Healthcare-Insurance/blob/3acb4738e6ed431a4daf5361478ae77e4df9f24a/pairplothealth.png)

Insights:

Age vs Charges:
  * Low-cost line that increases slowly with age. This represents healthy, non-smoking individuals.
  * A mix of smokers and non-smokers with higher costs, likely due to pre-existing conditions or minor health complications.
  * Even 20-year-old smokers can face higher insurance charges than a 60-year-old non-smoker

BMI vs Charges:
  * Non-smokers, charges remain relatively flat as the BMI increases.
  * For smokers, there is a massive breakout at BMI 30. This suggests that the combinationof smoking and obesity act as a "multiplier," sending insurance costs into the $35,000 - $60,000 range.

Age vs BMI:
  * There is no concrete observable data.
  * Variables are independent.

Data Distribution:
  * High concentration of younger people (around age 20).
  * "Charges" shows a massive positive skew.
  * Most people have low costs, while a small amount of people (mostly smokers) account for high-cost.

For the next step is necessary to "simplify" the columns for "Sex." This can be done in multiple ways but the easier way is take the dataset from "Sex" and convert the categorical factors into dummies. The result of this will be the creation of a table with the columns "female" and "male."

```python
pd.get_dummies(dataset['sex'])
```
<img width="243" height="566" alt="image" src="https://github.com/user-attachments/assets/ab7ce34a-41e3-4fe2-bb62-8d58278b9ec4" />

The next step is to drop the new first column, only leaving the "male" column for analysis.

```python
male = pd.get_dummies(dataset['sex'], drop_first=True)
male
```
<img width="144" height="537" alt="image" src="https://github.com/user-attachments/assets/978c98fb-8167-499b-b72b-097850606db8" />

This process is repeated in order to isolate the data. Dummies for "region" and "smoker" are generated. 

```python
#Region
dataset['region'].unique()
pd.get_dummies(dataset['region'])
region = pd.get_dummies(dataset['region'], drop_first=True)

#Smoke4
pd.get_dummies(dataset['smoker'])
smoke = pd.get_dummies(dataset['smoker'], drop_first=True)
```

With the creation of these new tables (region, male, smoke) a new dataset is created with only the relevant data for the analysis.

```python
dataset1 = pd.concat([dataset, region, male, smoke], axis=1)
dataset1.head()
```

<img width="1249" height="303" alt="image" src="https://github.com/user-attachments/assets/46a08717-3fbe-414c-95a9-9c0278c1838d" />

# Machine Learning Modeling

Linear regresion used in otder to create "best fit line" that predicts the charges.

In this section of the analysis the data set is inyected into the model. "X" and "Y" are assigned to the dataset and "charges" column respectively.

```python
x = dataset1.drop(['charges'], axis=1)
y = dataset1['charges']
```

The data is splited train and test for the prediction. For this dataset the test size is assigned to 0.2 and a random state is assigned.
The data is fitted to the model.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=101)
lr = LinearRegression()
lr.fit(x_train, y_train)
```

The prediciton is assign:

```python
prediction = lr.predict(x_test)
```

4. Final Interpretation

After the prediction is generated using linear regresion is possible to generate plots in order to understand the data.

![Scatter Plot](https://github.com/Ktiscar1/Healthcare-Insurance/blob/d38b7a5de31ae0291bac83f39a68980ed7faa398/scatter%20plot%20health.png)

The scatter plot is generated using x=y_test and y=prediciton. The objective of this graph is to observe the model performance by comparing actual values from the test set against values of the prediction (model).
  * The scatter plot shows that there is a strong possitive connection between the predictions and the actual values.
  * The model has successfully learned the primary driver of insurance costs.
  * The lower are lane likely shows non-smokers.
  * The upper lane are likely the smokers.
  * Outliner where the cost was over $60,000 but the model predicts closer to $40,000.

The correlation heatmap provides a mathematical score for how much each feature influences the insurance charges.

![Heatmap](https://github.com/Ktiscar1/Healthcare-Insurance/blob/9442e06e60874404f135e2fa98d148907a79559a/heatmap%20health.png)

  * The model place a massive weight on the smoker variable (0.79).
  * Being male or female has no impact into the cost of insurance.
  * The region has near zero impact in the charges.
  * BMI has a lower correlation with charges than age. BMI only becomes a massive cost driver when the person is also a smoker.
  * As age increases, charges tend to increase.
