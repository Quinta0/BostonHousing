
# Boston Housing Price Prediction

## Project Overview

The Boston Housing Price Prediction project involves building a machine learning model to predict housing prices in the Boston area based on various features such as crime rate, average number of rooms per dwelling, and accessibility to radial highways.

### Scope of the Project

The main objectives of this project are:
- To preprocess and explore the Boston housing dataset.
- To build and evaluate multiple regression models.
- To select the best model based on performance metrics.
- To make predictions on new data.

## Project Structure

The project is organized as follows:

1. **Data Loading and Exploration**: Load the dataset and perform initial data exploration to understand the structure and key statistics.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
3. **Model Building and Evaluation**: Build multiple regression models, evaluate their performance, and select the best model.
4. **Model Prediction**: Use the selected model to make predictions on new data.

## How to Run the Project

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Necessary libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/boston_housing_prediction.git
   cd boston_housing_prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook BostonHousing.ipynb
   ```

4. **Execute the Notebook Cells**
   Open the `BostonHousing.ipynb` notebook and run each cell sequentially to reproduce the results.

## Data Description

The Boston housing dataset contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. It has the following features:
- `CRIM`: per capita crime rate by town.
- `ZN`: proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: proportion of non-retail business acres per town.
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- `NOX`: nitrogen oxides concentration (parts per 10 million).
- `RM`: average number of rooms per dwelling.
- `AGE`: proportion of owner-occupied units built prior to 1940.
- `DIS`: weighted distances to five Boston employment centres.
- `RAD`: index of accessibility to radial highways.
- `TAX`: full-value property tax rate per $10,000.
- `PTRATIO`: pupil-teacher ratio by town.
- `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
- `LSTAT`: percentage of lower status of the population.
- `MEDV`: median value of owner-occupied homes in $1000s.

## Conclusions

The project demonstrates the process of building a machine learning model to predict housing prices. It involves data preprocessing, model selection, evaluation, and making predictions. The key takeaways include:
- The importance of data preprocessing for model performance.
- The evaluation of multiple models to select the best performing one.
- The ability to make predictions on new data using the selected model.


# The question
How can the analysis of various socio-economic and environmental factors, like crime rate, land zoning, business proportion, pollution levels, housing features, and educational resources, accurately predict the market values of houses in Boston?

This question seeks to understand the impact of a wide range of factors on house values, aiming to develop a comprehensive model that accounts for the multifaceted nature of real estate valuation.

# Perform Exploratory Data Analysis (EDA)

Perform an exploratory analysis on a broader set of variables relevant to the research question.
# Visualizing the data
## Some comments and findings
The whole dataset is composed by 506 entries spread alongside 12 columns, in this dataset there are no missing values. 

### The Statistics 
1) MEDV (Median Value of Houses):
- Range: $5,000 - $50,000
- Mean: $22,533
- Standard Deviation: $9,197
2) RM (Average Number of Rooms per Dwelling):
- Range: 3.56 - 8.78 rooms
- Mean: 6.28 rooms
- Standard Deviation: 0.70 rooms
3) RAD (Index of Accessibility to Radial Highways):
- Range: 1 - 24
- Mean: 9.55
- Standard Deviation: 8.71

### Visualizations
1. Histograms:
- MEDV: Appears to have a semi-normal distribution with a peak at around the values $20,000 - $25,000.
- RM: Shows a normal distribution centered around 6 rooms.
- RAD: Exhibits a bimodal distribution, indicating two groups of areas based on highway accessibility.

2. Scatter Plots:
MEDV vs. RM: 
- Shows a positive correlation, suggesting that houses with more rooms tend to have higher median values.
MEDV vs. RAD: 
- The relationship is not as clear, but there seems to be a trend
- where higher accessibility to highways correlates to lower house values, possibly due to noise or other factors.

3. Correlation Matrix:
- MEDV and RM: Strong positive correlation (0.70).
- MEDV and RAD: Negative correlation (-0.38).

## Initial Findings
• The number of rooms (RM) has a significant positive impact on the median value of houses.

• Accessibility to highways (RAD) appears to negatively impact house values, though this relationship is less clear and might be influenced by other factors like noise or pollution.
# Formal Analysis 
The proces is the folowwing:

1. *Data Splitting*:
- Create a function to split the dataset into a 90% training set and a 10% test set.
2. *Model Fitting*:
- Develop a function to fit a linear regression model using the training set.
3. *Coefficient Interpretation*:
- Interpret the coefficients for rad and crim.
4. *Bootstrap Analysis*:
- Implement a bootstrap method to test the significance of coefficients.
5. *Hypothesis Testing and Model Refinement*:
- Use hypothesis testing to identify significant variables.
- Refine the model based on significant variables.
- Evaluate the refined model on the test set using Mean Squared Error (MSE)

In our confindence list of the intervals the data that is considered statistically significant are: CRIM, ZN, NOX, RM, DIS, RAD, TAX, PTRATIO and LSTAT
# Model Coefficients 
- crim       -0.131907
- zn          0.048926
- indus       0.024283
- nox       -18.026122
- rm          3.520948
- age         0.003638
- dis        -1.535510
- rad         0.334010
- tax        -0.013442
- ptratio    -1.004608
- lstat      -0.583388

## Interpreting the coefficients 
The coefficients of the linear regression model, which predicts the median value of houses (medv) using various features, are as follows:
- CRIM: -0.132 (per capita crime rate by town)
- ZN: 0.049 (proportion of residential land zoned for lots over 25,000 sq.ft.)
- INDUS: 0.024 (proportion of non-retail business acres per town)
- NOX: -18.03 (nitric oxides concentration)
- RM: 3.52 (average number of rooms per dwelling)
- AGE: 0.004 (proportion of owner-occupied units built prior to 1940)
- DIS: -1.54 (weighted distances to five Boston employment centers)
- RAD: 0.334 (index of accessibility to radial highways)
- TAX: -0.013 (full-value property-tax rate per $10,000)
- PTRATIO: -1.005 (pupil-teacher ratio by town)
- LSTAT: -0.583 (% lower status of the population)

_Values can vary because of the manual splits function_ 

*Interpretation of Key Coefficients*:
- RM (Rooms): A coefficient of 3.52 suggests a strong positive relationship between the
number of rooms and house value. Each additional room is associated with an increase in
the median value by approximately $3521.
- RAD (Accessibility to Highways): The coefficient of 0.334 indicates a positive
relationship, meaning that increased accessibility to highways is associated with a higher
median house value. However, the impact is relatively smaller compared to other
variables like rm.
*Mean Squared Error (MSE) on Test Set*:
- The MSE of the model on the test set is 21.58. This value represents the average
squared difference between the actual and predicted house values, providing a measure
of the model's accuracy.
## Bootstrap Analysis:
- The bootstrap analysis performed on the linear regression model provides insightful statistics about the stability and significance of each coefficient. By calculating the standard deviation of coefficients across multiple bootstrap samples, we gain an understanding of their variability. This variability measurement is crucial for assessing the reliability of each feature's impact on the model. 

- Additionally, the 95% confidence intervals for each coefficient are calculated. These intervals give a range in which the true value of the coefficient is likely to fall, offering a degree of certainty about the estimations. This approach is particularly useful in determining the robustness of each feature's influence on the target variable.

- Lastly, significance testing is conducted to determine whether the coefficients are significantly different from zero. A coefficient significantly different from zero suggests that the corresponding feature plays a meaningful role in predicting the target variable. This step is vital for feature selection and model refinement, as it helps in identifying the most impactful predictors among the available features.
-Overall, the bootstrap analysis deepens our understanding of the model's behavior by providing a comprehensive view of the importance and reliability of each feature in the dataset
# Refined model
Bar charts are used to compare the MSE of the original and refined models.
Violin plots are drawn to display the distribution of predictions from both models against the actual values, offering a comprehensive view of the model's performance.
## Results of the refined model

*Mean Squared Error (MSE) Comparison*:
- The MSE of the original model was 21.576.
- After refining the model to focus on the variables RM (number of rooms) and RAD
(accessibility to highways), the MSE increased to 21.69. This increase suggests that while these two variables are important, other variables might be needed to be included to have a better model and predictions

*Coefficients of the Refined Model*:
- RM (Rooms): Coefficient of 8.341. This larger coefficient in the refined model
underscores the strong positive impact of the number of rooms on house value.
- RAD (Accessibility to Highways): Coefficient of -0.266. In the refined model, this variable shows a negative impact, contrary to the original model. This suggests that when considered alone with RM, highway accessibility might be inversely related to house values.

*Visualizations*:
1) MSE Comparison: The bar chart compares the MSEs of the original and refined models, illustrating the change in prediction accuracy.
2) Coefficients of the Refined Model: The bar chart displays the coefficients of RM and RAD in the refined model, highlighting their relative impacts on house value.
## Some Considerations 
Since the model was so little and the split function made by myself was not randomized enough it means that the MSE and the impact of the coefficients are not as accurate as the pre defined one like from the 'sklearn' library or the 'xgboost' ones below we can see how close or far we got from these
As we can see if we had used the already existing split functions the model would have been much better 
# The Conclusion

In conclusion, this analysis employs a data-driven approach to unravel the intricate factors influencing house values in Boston. By considering a broad spectrum of socio-economic and environmental variables, the study provides valuable insights into the housing market, highlighting the significance of certain features while also emphasizing the complexity and multifaceted nature of real estate valuation.

