#week 1

1. In a specific city, let the population proportion, 𝜋, represent those who agree with the minimum wage increase. <br>
Consider two random samples, and let 𝑌 be the number of people supporting the wage increase. <br><br>

a. When 𝜋=0.50, determine the possible values that Y can take and calculate the corresponding probabilities. <br>
Also, find the mean and variance of the probability distribution of 𝑌. <br><br>

b. Assuming π is unknown and you observe y=1, calculate the likelihood function and represent it graphically. <br>
Based on the plot of the likelihood function, explain why the maximum likelihood estimator pi-hat=0.50. <br><br>

2. In the 2010 General Social Survey, 486 out of 1374 respondents answered "yes" to the question of whether they would be willing to lower their standard of living to protect the environment. <br><br>

a. Estimate the population proportion of those who would answer "yes." <br>
Calculate the 99% confidence interval for this population proportion and interpret its meaning. <br><br>

b. Conduct a significance test to determine whether the proportion of people answering "yes" is a majority or minority. <br>
Provide the p-value and its interpretation. <br><br>

*The Wald test, the score test, and the likelihood ratio test may yield the same conclusion regarding rejection, but the test statistics may differ. <br> 
When choosing one of the three methods, the analyst should select the most appropriate one. <br>
The likelihood ratio test is characterized by using the maximum likelihood estimate (MLE) directly, while the score test always uses the maximized value of 𝜋(𝜋0), <br>
resulting in a standard error (SE) that is always larger than that used in the Wald test, which employs pi-hat. <br>
Therefore, the z-value, which uses SE as the denominator, will be smaller, leading to a more conservative approach to rejection (less likely to reject the null hypothesis). <br>
SAS shows all three cases, but R applies different testing methods depending on the form used.
