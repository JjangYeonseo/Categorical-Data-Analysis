#1
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Contingency table
data = np.array([[2, 4, 13, 3],
                 [2, 6, 22, 4],
                 [0, 1, 15, 8],
                 [0, 3, 13, 8]])

# Perform Chi-Squared test
chi2, p, dof, expected = chi2_contingency(data)

# Calculate standardized residuals
standardized_residuals = (data - expected) / np.sqrt(expected)

# Output results
print(f"Chi-Squared Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)
print("Standardized Residuals:")
print(standardized_residuals)

#2
(a)
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# Data for the contingency table
data = {
    "Death Penalty": [19, 0, 11, 6],
    "No Death Penalty": [132, 9, 52, 97]
}

# Create a DataFrame
df = pd.DataFrame(data, index=["White-White", "White-Black", "Black-White", "Black-Black"])

# Calculate odds ratios for white victims
odds_ratio_white = (df.loc["White-White"]["Death Penalty"] / df.loc["White-White"]["No Death Penalty"]) / \
                   (df.loc["Black-White"]["Death Penalty"] / df.loc["Black-White"]["No Death Penalty"])

# Calculate odds ratios for black victims
odds_ratio_black = (df.loc["White-Black"]["Death Penalty"] / df.loc["White-Black"]["No Death Penalty"]) / \
                   (df.loc["Black-Black"]["Death Penalty"] / df.loc["Black-Black"]["No Death Penalty"])

# Calculate the marginal odds ratio
marginal_odds_ratio = (df["Death Penalty"].sum() / df["No Death Penalty"].sum())

# Output results
print(f"Odds Ratio for White Victims: {odds_ratio_white:.4f}")
print(f"Odds Ratio for Black Victims: {odds_ratio_black:.4f}")
print(f"Marginal Odds Ratio: {marginal_odds_ratio:.4f}")

(b)
# Define the scores
income_scores = np.array([3, 10, 20, 35])
job_satisfaction_scores = np.array([1, 3, 4, 5])

# Flatten the counts for regression analysis
y = np.repeat(job_satisfaction_scores, data.sum(axis=1))
x = np.tile(income_scores, data.sum(axis=1))

# Perform linear regression
slope, intercept = np.polyfit(x, y, 1)

# Output results
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

#4
# Given values
x = 12  # Number of decades since 1904

# Linear probability model prediction
linear_intercept = 0.6930
linear_slope = -0.0662
linear_prediction = linear_intercept + linear_slope * x

# Logistic regression prediction
logistic_prediction = 0.034

# Output results
print(f"Predicted proportion of complete games for 2024 using linear model: {linear_prediction:.4f}")
print(f"Predicted proportion of complete games for 2024 using logistic model: {logistic_prediction:.4f}")

