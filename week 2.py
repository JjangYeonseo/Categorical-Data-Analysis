#1
# Given probabilities
P_C = 0.04  # Probability of having prostate cancer
P_not_C = 1 - P_C  # Probability of not having prostate cancer

P_pos_given_C = 0.75  # Sensitivity: Probability of a positive test given cancer
P_neg_given_C = 0.25  # Probability of a negative test given cancer

P_pos_given_not_C = 0.1  # False positive rate: Probability of a positive test given no cancer
P_neg_given_not_C = 0.9  # Probability of a negative test given no cancer

# Step 1: Calculate joint probabilities for the 2x2 table
P_C_and_pos = P_C * P_pos_given_C
P_C_and_neg = P_C * P_neg_given_C
P_not_C_and_pos = P_not_C * P_pos_given_not_C
P_not_C_and_neg = P_not_C * P_neg_given_not_C

# Step 2: Create the 2x2 joint probability table
joint_prob_table = {
    "C_and_pos": P_C_and_pos,
    "C_and_neg": P_C_and_neg,
    "not_C_and_pos": P_not_C_and_pos,
    "not_C_and_neg": P_not_C_and_neg
}

# Print the joint probability table
print("Joint Probability Table (P(C and +), P(C and -), P(~C and +), P(~C and -)):")
for key, value in joint_prob_table.items():
    print(f"{key}: {value:.4f}")

# Step 3: Calculate the marginal probabilities for diagnosis
P_pos = P_C_and_pos + P_not_C_and_pos  # P(+) = P(C and +) + P(~C and +)
P_neg = P_C_and_neg + P_not_C_and_neg  # P(-) = P(C and -) + P(~C and -)

# Print marginal probabilities
print(f"\nMarginal Probability of Positive Test (P(+)): {P_pos:.4f}")
print(f"Marginal Probability of Negative Test (P(-)): {P_neg:.4f}")

# Step 4: Apply Bayes' Theorem to find P(C | +)
P_C_given_pos = P_C_and_pos / P_pos  # P(C | +) = P(C and +) / P(+)

# Print the result of Bayes' Theorem
print(f"\nProbability of having cancer given a positive test (P(C | +)): {P_C_given_pos:.4f}")


#2
# Given values
gamma = 0.01        # Prevalence of the disease
pi1 = 0.86          # Sensitivity (P(Y = 1 | X = 1))
pi2 = 0.88          # Specificity (P(Y = 2 | X = 2))

# (a) Calculate the positive predictive value (PPV)
numerator = pi1 * gamma
denominator = (pi1 * gamma) + ((1 - pi2) * (1 - gamma))
PPV = numerator / denominator

print(f"Positive Predictive Value (PPV): {PPV:.4f}")

# (b) Calculate joint probabilities for the 2x2 table

# P(X = 1, Y = 1): True Positive
P_X1_Y1 = pi1 * gamma

# P(X = 1, Y = 2): False Negative
P_X1_Y2 = (1 - pi1) * gamma

# P(X = 2, Y = 1): False Positive
P_X2_Y1 = (1 - pi2) * (1 - gamma)

# P(X = 2, Y = 2): True Negative
P_X2_Y2 = pi2 * (1 - gamma)

# Display joint probabilities
print(f"Joint Probability P(X=1, Y=1) (True Positive): {P_X1_Y1:.4f}")
print(f"Joint Probability P(X=1, Y=2) (False Negative): {P_X1_Y2:.4f}")
print(f"Joint Probability P(X=2, Y=1) (False Positive): {P_X2_Y1:.4f}")
print(f"Joint Probability P(X=2, Y=2) (True Negative): {P_X2_Y2:.4f}")


#3
# Given probabilities for referral
P_whites = 0.906  # Probability of referral for whites
P_blacks = 0.847  # Probability of referral for blacks

# Step 1: Calculate the odds for whites and blacks
def calculate_odds(probability):
    return probability / (1 - probability)

odds_whites = calculate_odds(P_whites)
odds_blacks = calculate_odds(P_blacks)

# Step 2: Calculate the odds ratio (OR)
odds_ratio = odds_blacks / odds_whites

# Step 3: Calculate the relative risk (RR)
relative_risk = P_blacks / P_whites

# Step 4: Output the results
print(f"Odds for Whites: {odds_whites:.4f}")
print(f"Odds for Blacks: {odds_blacks:.4f}")
print(f"Odds Ratio (OR): {odds_ratio:.4f}")
print(f"Relative Risk (RR): {relative_risk:.4f}")


#4
import numpy as np
import math

# Given data from Table 2.11
O_OO = 802  # Obama in 2008 and Obama in 2012
O_OR = 53   # Obama in 2008 and Romney in 2012
O_MO = 34   # McCain in 2008 and Obama in 2012
O_MR = 494  # McCain in 2008 and Romney in 2012

# Step 1: Calculate the Odds Ratio (OR)
odds_ratio = (O_OO * O_MR) / (O_OR * O_MO)

# Step 2: Calculate the log(OR)
log_odds_ratio = math.log(odds_ratio)

# Step 3: Calculate the Standard Error (SE) of log(OR)
SE_log_OR = math.sqrt((1 / O_OO) + (1 / O_OR) + (1 / O_MO) + (1 / O_MR))

# Step 4: Calculate the 95% Confidence Interval for the log(OR)
z_value = 1.96  # For a 95% confidence interval
lower_log_CI = log_odds_ratio - z_value * SE_log_OR
upper_log_CI = log_odds_ratio + z_value * SE_log_OR

# Step 5: Exponentiate to get the confidence interval for the Odds Ratio
lower_CI = math.exp(lower_log_CI)
upper_CI = math.exp(upper_log_CI)

# Step 6: Output the results
print(f"Odds Ratio (OR): {odds_ratio:.2f}")
print(f"95% Confidence Interval for OR: ({lower_CI:.2f}, {upper_CI:.2f})")


#5
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Define the observed data (observed frequencies)
data = np.array([[21, 159, 110],   # Above average income
                 [53, 372, 221],   # Average income
                 [94, 249, 83]])   # Below average income

# Create a DataFrame for better visualization
df = pd.DataFrame(data, columns=['Not Too Happy', 'Pretty Happy', 'Very Happy'], 
                  index=['Above average', 'Average', 'Below average'])

# Perform chi-square test for independence
chi2_stat, p_value, dof, expected = chi2_contingency(data)

# Calculate standardized residuals
standardized_residuals = (data - expected) / np.sqrt(expected)

# Display the results
print("Observed Data:")
print(df, "\n")

print("Expected Data (under null hypothesis of independence):")
print(pd.DataFrame(np.round(expected, 1), columns=df.columns, index=df.index), "\n")

print(f"Chi-square statistic: {chi2_stat:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.5f}\n")

print("Standardized Residuals:")
print(pd.DataFrame(np.round(standardized_residuals, 3), columns=df.columns, index=df.index))

# Interpretation for P-value and standardized residuals
if p_value < 0.05:
    print("\nThe P-value is less than 0.05, so we reject the null hypothesis. Income and happiness are not independent.")
else:
    print("\nThe P-value is greater than 0.05, so we fail to reject the null hypothesis. Income and happiness may be independent.")
