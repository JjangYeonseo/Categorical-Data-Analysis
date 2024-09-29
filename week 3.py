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

#6-7
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# 데이터 설정
treatment_A = [8, 7, 6, 6, 3, 4, 7, 2, 3, 4]
treatment_B = [9, 9, 8, 14, 8, 13, 11, 5, 7, 6]

data = pd.DataFrame({
    'imperfections': treatment_A + treatment_B,
    'treatments': ['A']*10 + ['B']*10
})

# 처리 B를 1로, 처리 A를 0으로 인코딩
data['treatment_B'] = (data['treatments'] == 'B').astype(int)

# 모델 fitting
model = smf.glm(formula="imperfections ~ treatment_B", data=data, family=sm.families.Poisson())
results = model.fit()

# 결과 출력
print(results.summary())

beta = results.params['treatment_B']
print(f"β̂ = {beta:.4f}")
print(f"exp(β̂) = {np.exp(beta):.4f}")

# Wald 검정
print(results.summary().tables[1])

# 우도비 검정
null_model = smf.glm(formula="imperfections ~ 1", data=data, family=sm.families.Poisson()).fit()
lr_statistic = -2 * (null_model.llf - results.llf)
lr_pvalue = stats.chi2.sf(lr_statistic, 1)
print(f"Likelihood Ratio Test p-value: {lr_pvalue:.4f}")

# β에 대한 신뢰구간 계산
ci_beta = results.conf_int().loc['treatment_B']

# μB/μA에 대한 신뢰구간 (지수화)
ci_ratio = np.exp(ci_beta)

print("95% CI for μB/μA:")
print(f"({ci_ratio[0]:.4f}, {ci_ratio[1]:.4f})")

