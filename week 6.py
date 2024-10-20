#1
import pandas as pd
import statsmodels.api as sm

# 데이터셋 불러오기
url = "http://users.stat.ufl.edu/~aa/intro-cda/data/MBTI.dat"
df = pd.read_table(url)

# 'no'와 'yes' 응답 열 추가
df['no'] = df['n'] - df['drink']
df['yes'] = df['drink']

# 이진 변수(EI, SN, TF, JP) 생성
df['EI'] = df['EI'].map({'e': 0, 'i': 1})  # 외향성 = 0, 내향성 = 1
df['SN'] = df['SN'].map({'s': 0, 'n': 1})  # 감각 = 0, 직관 = 1
df['TF'] = df['TF'].map({'t': 0, 'f': 1})  # 사고 = 0, 감정 = 1
df['JP'] = df['JP'].map({'j': 0, 'p': 1})  # 판단 = 0, 인식 = 1

# 독립 변수와 종속 변수 정의
X = df[['EI', 'SN', 'TF', 'JP']]  # 독립 변수
X = sm.add_constant(X)  # 상수항 추가
y = df['yes'] / (df['yes'] + df['no'])  # 종속 변수(음주 확률)

# 로지스틱 회귀 모델 적합
model = sm.GLM(y, X, family=sm.families.Binomial(), weights=df['yes'] + df['no'])
result = model.fit()

# 모델 요약 출력
print(result.summary())

import numpy as np

# 각 MBTI 유형에 대한 예측 확률 계산
df['predicted_prob'] = result.predict(X)

# 예측 확률 출력
print(df[['EI', 'SN', 'TF', 'JP', 'predicted_prob']])

# ENTP 유형에 대한 예측 확률 계산 (EI=0, SN=1, TF=0, JP=1)
ENTP = np.array([1, 0, 1, 0, 1])  # 상수항 포함
ENTP_prob = result.predict(ENTP)

print(f"ENTP 유형의 예측 확률은 {ENTP_prob[0]}입니다.")

# 모든 MBTI 유형에 대한 예측 확률 컬럼 추가
df['predicted_prob'] = result.predict(X)

# 예측 확률을 포함한 전체 테이블 출력
print(df[['EI', 'SN', 'TF', 'JP', 'predicted_prob']])

#2
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

# 데이터 불러오기
url = "http://users.stat.ufl.edu/~aa/intro-cda/data/Crabs.dat"
crabs_data = pd.read_table(url)

# 데이터 구조 확인
print(crabs_data.head())

# 종속 변수(y = satellite 여부), 독립 변수(x = weight)
y = crabs_data['satell']
X = crabs_data[['weight']]
X = sm.add_constant(X)  # 상수항 추가

# 로지스틱 회귀 모델 적합
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# 결과 요약 출력
print(result.summary())

# 회귀 계수 추출
params = result.params
beta_0 = params[0]  # 상수항 (절편)
beta_1 = params[1]  # 무게에 대한 회귀 계수

print(f"예측 방정식: log(π(x) / (1 - π(x))) = {beta_0} + {beta_1} * weight")

mean_weight = 2.437
log_odds = beta_0 + beta_1 * mean_weight
prob = 1 / (1 + np.exp(-log_odds))

print(f"평균 무게 {mean_weight}kg에서의 위성 존재 확률: {prob:.4f}")

weight_increase_1kg = beta_1 * 1
prob_increase_1kg = 1 / (1 + np.exp(-(log_odds + weight_increase_1kg)))

print(f"1kg 증가 시 확률: {prob_increase_1kg:.4f}")

weight_increase_0_1kg = beta_1 * 0.1
prob_increase_0_1kg = 1 / (1 + np.exp(-(log_odds + weight_increase_0_1kg)))

print(f"0.10kg 증가 시 확률: {prob_increase_0_1kg:.4f}")

weight_increase_sd = beta_1 * 0.58
prob_increase_sd = 1 / (1 + np.exp(-(log_odds + weight_increase_sd)))

print(f"표준 편차 0.58kg 증가 시 확률: {prob_increase_sd:.4f}")

# 평균 한계 효과 계산
marginal_effect_0_1kg = beta_1 * 0.1
print(f"0.10kg 증가에 따른 평균 한계 효과: {marginal_effect_0_1kg}")

# 예측 확률 계산
y_pred_prob = result.predict(X)
y_pred = (y_pred_prob > y.mean()).astype(int)

# 분류 테이블
classification_table = pd.crosstab(y, y_pred, rownames=['실제값'], colnames=['예측값'])
print(classification_table)

# 성능 지표 계산
sensitivity = classification_table[1][1] / (classification_table[1][1] + classification_table[0][1])
specificity = classification_table[0][0] / (classification_table[0][0] + classification_table[1][0])

print(f"감도(Sensitivity): {sensitivity:.4f}")
print(f"특이도(Specificity): {specificity:.4f}")

# ROC 곡선 그리기
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f"ROC AUC: {roc_auc:.4f}")

