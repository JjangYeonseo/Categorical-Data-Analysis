#1
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit

# 데이터 입력
data = {
    'Gender': ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male'],
    'Location': ['Rural', 'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban'],
    'SeatBelt': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Injured': [973, 757, 996, 759, 1084, 513, 812, 380],
    'NotInjured': [3246, 6134, 7287, 11587, 6123, 6693, 10381, 10969]
}

df = pd.DataFrame(data)

# 1단계: S (SeatBelt) ~ G (Gender) + L (Location)
df['Total'] = df['Injured'] + df['NotInjured']
df['SeatBeltBinary'] = (df['SeatBelt'] == 'Yes').astype(int)

logit_model_1 = logit("SeatBeltBinary ~ Gender + Location", data=df)
result_1 = logit_model_1.fit()
print(result_1.summary())

# 2단계: I (Injured) ~ G (Gender) + L (Location) + S (SeatBelt)
df['InjuryRate'] = df['Injured'] / df['Total']
logit_model_2 = logit("InjuryRate ~ Gender + Location + SeatBeltBinary", data=df)
result_2 = logit_model_2.fit()
print(result_2.summary())

#2
