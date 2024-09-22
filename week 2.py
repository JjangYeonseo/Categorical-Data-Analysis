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
# Given data
gamma = 0.01  # Prevalence of the disease (1%)
pi_1 = 0.86   # Sensitivity (P(Y = 1 | X = 1))
specificity = 0.88  # Specificity
pi_2 = 1 - specificity  # False positive rate (P(Y = 1 | X = 2))

# Part (a) - Using Bayes' Theorem to find P(X = 1 | Y = 1)
def bayes_theorem(pi_1, pi_2, gamma):
    # Bayes' Theorem to calculate P(X = 1 | Y = 1)
    numerator = pi_1 * gamma
    denominator = (pi_1 * gamma) + (pi_2 * (1 - gamma))
    return numerator / denominator

# Part (b) - Find the positive predictive value (PPV)
ppv = bayes_theorem(pi_1, pi_2, gamma)
print(f"Positive Predictive Value (PPV): {ppv:.4f}")

# Part (c) - Finding the joint probabilities for the 2x2 table

# P(X = 1 and Y = 1) -> True positives
true_positives = gamma * pi_1

# P(X = 1 and Y = 2) -> False negatives
false_negatives = gamma * (1 - pi_1)

# P(X = 2 and Y = 1) -> False positives
false_positives = (1 - gamma) * pi_2

# P(X = 2 and Y = 2) -> True negatives
true_negatives = (1 - gamma) * (1 - pi_2)

# Print the 2x2 joint distribution table
print("\nJoint Distribution Table (2x2):")
print(f"{'':<20}{'Positive Test (Y=1)':<25}{'Negative Test (Y=2)'}")
print(f"{'Has Disease (X=1)':<20}{true_positives:<25.4f}{false_negatives:.4f}")
print(f"{'No Disease (X=2)':<20}{false_positives:<25.4f}{true_negatives:.4f}")

# Discussion of relative sizes
print("\nDiscussion:")
print(f"True Positives (X=1 and Y=1): {true_positives:.4f}")
print(f"False Negatives (X=1 and Y=2): {false_negatives:.4f}")
print(f"False Positives (X=2 and Y=1): {false_positives:.4f}")
print(f"True Negatives (X=2 and Y=2): {true_negatives:.4f}")

# Observations on the sizes
if false_positives > true_positives:
    print("\nObservation: There are more false positives than true positives.")
if true_negatives > false_negatives:
    print("Observation: There are far more true negatives than false negatives.")
if false_negatives < true_positives:
    print("Observation: The number of false negatives is smaller than true positives.")


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
