#1
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 제공된 데이터를 바탕으로 데이터셋 구성
data = {
    'SnoringLevel': [0, 2, 4, 5],     # 코골이 정도 (0, 2, 4, 5)
    'HeartDisease_Yes': [34, 35, 21, 30],  # 심장병 있는 환자 수
    'HeartDisease_No': [1355, 603, 192, 224] # 심장병 없는 환자 수
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 심장병 발생 확률 계산
df['Total'] = df['HeartDisease_Yes'] + df['HeartDisease_No']
df['HeartDisease_Prob'] = df['HeartDisease_Yes'] / df['Total']

# 데이터 확장 (로지스틱 회귀에 맞게 Yes/No 데이터 생성)
expanded_data = pd.DataFrame({
    'SnoringLevel': np.repeat(df['SnoringLevel'].values, df['Total'].values),
    'HeartDisease': np.repeat([1, 1, 1, 1], df['HeartDisease_Yes'].values).tolist() +
                    np.repeat([0, 0, 0, 0], df['HeartDisease_No'].values).tolist()
})

# 절편 추가
expanded_data['intercept'] = 1

# 로지스틱 회귀 모델 적합
logit_model = sm.Logit(expanded_data['HeartDisease'], expanded_data[['intercept', 'SnoringLevel']])
result = logit_model.fit()

# 모델 요약 출력
print(result.summary())

# SnoringLevel의 계수 추출 및 오즈비 계산
beta_snoring = result.params['SnoringLevel']
odds_ratio = np.exp(beta_snoring)
print(f"코골이 정도 1단위 증가에 따른 오즈비: {odds_ratio:.4f}")

# 예측을 위한 SnoringLevel 범위 생성
x_range = np.linspace(0, 5, 100)

# 로지스틱 함수 정의
def logistic(x, beta0, beta1):
    return 1 / (1 + np.exp(-(beta0 + beta1 * x)))

# 모델 파라미터 추출
beta0 = result.params['intercept']
beta1 = result.params['SnoringLevel']

# 예측 확률 계산
y_pred = logistic(x_range, beta0, beta1)

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_pred, 'r-', label='Logistic Regression Model')
plt.scatter(df['SnoringLevel'], df['HeartDisease_Prob'], color='blue', label='Observed Data')

plt.xlabel('Snoring Level')
plt.ylabel('HeartDisease_Prob')
plt.title('probability of heart disease according to snoring degree')
plt.legend()
plt.grid(True)

# y축 범위를 0에서 1로 설정
plt.ylim(0, 1)

plt.show()

# 추가적인 해석
print(f"모델의 절편 (beta0): {beta0:.4f}")
print(f"코골이 정도의 계수 (beta1): {beta1:.4f}")
print(f"코골이 정도가 1단위 증가할 때마다 심장병 발생 오즈가 {odds_ratio:.4f}배 증가합니다.")

#2
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# Data: Temperatures and whether thermal distress occurred (TD)
data = {'Temp': [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58],
        'TD': [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]}

df = pd.DataFrame(data)

# Add constant for the intercept
df['intercept'] = 1

# Fit the logistic regression model
logit_model = sm.Logit(df['TD'], df[['intercept', 'Temp']])
result = logit_model.fit()

# a. Print summary of the model
print("a. 로지스틱 회귀모형 적합 결과:")
print(result.summary())
print("\n해석: 기온이 증가할수록 열 문제 발생 확률이 감소하는 것으로 보입니다.")

# Coefficients from the fitted model
beta_0 = result.params['intercept']
beta_1 = result.params['Temp']

# b. Calculate the probability of thermal distress at 31°F
temp_challenger = 31
prob_challenger = 1 / (1 + np.exp(-(beta_0 + beta_1 * temp_challenger)))
print(f"\nb. 챌린저호 비행 시 기온(31°F)에서의 열 문제 발생 확률: {prob_challenger:.4f}")

# c. Find the temperature where the probability is 0.50
temp_50_prob = -beta_0 / beta_1
print(f"\nc. 예측 확률이 0.50이 되는 기온: {temp_50_prob:.2f}°F")

# Linear approximation
delta = 0.01
prob_plus_1 = 1 / (1 + np.exp(-(beta_0 + beta_1 * (temp_50_prob + 1))))
slope = (prob_plus_1 - 0.5) / 1

print(f"   선형근사식: P(열 문제) ≈ 0.5 + {slope:.4f} * (온도 - {temp_50_prob:.2f})")

# d. Interpret the effect of temperature on the odds of thermal distress
odds_ratio = np.exp(beta_1)
print(f"\nd. 오즈비: {odds_ratio:.4f}")
print(f"   해석: 기온이 1°F 증가할 때마다 열 문제 발생 오즈가 {odds_ratio:.4f}배가 됩니다.")

# e. Hypothesis test for the effect of temperature
p_value = result.pvalues['Temp']
print(f"\ne. 가설검정 결과:")
print(f"   p-value: {p_value:.4f}")
print("   결론: " + ("기온의 효과가 없다는 가설을 기각합니다." if p_value < 0.05 else "기온의 효과가 없다는 가설을 기각할 수 없습니다."))

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['Temp'], df['TD'], color='blue', label='Observed Data')

# Generate points for the logistic curve
temp_range = np.linspace(df['Temp'].min(), df['Temp'].max(), 100)
prob_range = 1 / (1 + np.exp(-(beta_0 + beta_1 * temp_range)))

plt.plot(temp_range, prob_range, color='red', label='Logistic Regression Model')
plt.plot([temp_challenger, temp_challenger], [0, prob_challenger], 'g--', label='Challenger Temperature')
plt.plot([temp_50_prob, temp_50_prob], [0, 0.5], 'k--', label='P=0.50 Temperature')

plt.xlabel('Temperature (°F)')
plt.ylabel('Probability of Thermal Distress')
plt.title('Logistic Regression: Temperature vs Probability of Thermal Distress')
plt.legend()
plt.grid(True)

plt.show()

#3
from scipy.stats import chi2

# Deviance 값 설정
null_deviance = 8.3499
residual_deviance = 1.3835

# Deviance 차이 (카이제곱 통계량)
chi_squared_stat = null_deviance - residual_deviance

# 자유도 설정
df = 2

# P-값 계산
p_value = chi2.sf(chi_squared_stat, df)

print(f"카이제곱 통계량: {chi_squared_stat:.4f}")
print(f"P-값: {p_value:.4f}")

import pandas as pd

# 데이터셋 설정
data = {
    'yes': [14, 32, 8, 12],   # AIDS 증상이 나타난 환자 수
    'no': [93, 81, 52, 43],   # 증상이 나타나지 않은 환자 수
    'azt': [1, 0, 1, 0],      # AZT 치료 여부 (1 = 즉시 치료, 0 = 지연 치료)
    'racewhite': [1, 1, 0, 0] # 인종 (1 = 백인, 0 = 흑인)
}

df = pd.DataFrame(data)

# 총 환자 수
df['total'] = df['yes'] + df['no']
df['response'] = df['yes'] / df['total']

# 로지스틱 회귀 모델 설정
import statsmodels.api as sm

# 절편 추가
df['intercept'] = 1

# 로지스틱 회귀 모델 적합
logit_model = sm.Logit(df['yes'] / df['total'], df[['intercept', 'azt', 'racewhite']], weights=df['total'])
result = logit_model.fit()

# 요약 결과 출력
print(result.summary())
