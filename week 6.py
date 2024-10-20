#1
import pandas as pd
import statsmodels.api as sm
import numpy as np

# 데이터프레임 생성
data = {
    'E': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'S': [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    'T': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    'J': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'Yes': [10, 8, 5, 7, 3, 2, 4, 15, 17, 3, 6, 4, 1, 5, 1, 6],
    'No': [67, 3, 101, 72, 20, 16, 27, 65, 123, 49, 132, 102, 12, 30, 30, 73]
}

df = pd.DataFrame(data)

# 독립 변수와 종속 변수 정의
X = df[['E', 'S', 'T', 'J']]  # 독립 변수
y = df['Yes'] / (df['Yes'] + df['No'])  # 종속 변수(음주 확률)

# 상수항 추가
X = sm.add_constant(X)

# 로지스틱 회귀 모델 적합
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# 모델 요약 출력
print(result.summary())

# 예측 확률 계산
df['predicted_prob'] = result.predict(X)

# 각 MBTI 유형에 대한 예측 확률 출력
print(df[['E', 'S', 'T', 'J', 'predicted_prob']])

# MBTI 유형에 대한 예측 확률 계산
mbti_types = {
    'ESTJ': [1, 1, 0, 0],
    'ESTP': [1, 1, 0, 1],
    'ESFJ': [1, 0, 0, 0],
    'ESFP': [1, 0, 0, 1],
    'ISTJ': [0, 1, 0, 0],
    'ISTP': [0, 1, 0, 1],
    'ISFJ': [0, 0, 0, 0],
    'ISFP': [0, 0, 0, 1],
    'ENTJ': [1, 1, 1, 0],
    'ENTP': [1, 1, 1, 1],
    'ENFJ': [1, 0, 1, 0],
    'ENFP': [1, 0, 1, 1],
    'INTJ': [0, 1, 1, 0],
    'INTP': [0, 1, 1, 1],
    'INFP': [0, 0, 1, 1],
    'INFJ': [0, 0, 1, 0]
}

# 각 MBTI 유형의 예측 확률 계산 및 출력
print("\n각 MBTI 유형의 음주 확률 예측:")
for mbti, values in mbti_types.items():
    values_with_const = [1] + values  # 상수항 추가
    prob = result.predict(values_with_const)
    print(f"{mbti} 유형의 음주 확률은 {prob[0]}입니다.")


#2
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 데이터 불러오기
url = "http://users.stat.ufl.edu/~aa/intro-cda/data/Crabs.dat(Table%203.2)"
df = pd.read_table(url, delim_whitespace=True)
df.head()

# 설명 변수와 종속 변수 설정
X = df['weight']
y = df['y']

# 상수항 추가 (intercept를 포함하기 위해)
X = sm.add_constant(X)

# 로지스틱 회귀 모델 피팅
model = sm.Logit(y, X)
result = model.fit()

# 결과 출력
print(result.summary())

# 1kg 증가에 대한 효과
effect_1kg = result.params['weight']
print(f"1kg 증가의 효과: {effect_1kg}")

# 0.1kg 증가에 대한 효과
effect_0_1kg = effect_1kg * 0.1
print(f"0.1kg 증가의 효과: {effect_0_1kg}")

# 0.58kg 증가에 대한 효과
effect_0_58kg = effect_1kg * 0.58
print(f"0.58kg 증가의 효과: {effect_0_58kg}")

# 예측 확률 계산
df['pred_prob'] = result.predict(X)

# 평균 한계 효과 계산 (1kg 증가당)
marginal_effect = np.mean(df['pred_prob'] * (1 - df['pred_prob']) * result.params['weight'])
print(f"평균 한계 효과 (1kg당): {marginal_effect}")

# 0.1kg 증가에 대한 평균 한계 효과
marginal_effect_0_1kg = marginal_effect * 0.1
print(f"평균 한계 효과 (0.1kg당): {marginal_effect_0_1kg}")

# 임계값 설정
threshold = df['y'].mean()

# 예측값과 실제값을 기반으로 분류 테이블 생성
df['pred_class'] = (df['pred_prob'] > threshold).astype(int)
conf_matrix = confusion_matrix(df['y'], df['pred_class'])

print(f"분류 테이블:\n{conf_matrix}")

# 민감도(Sensitivity) 계산
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
print(f"민감도: {sensitivity}")

# 특이도(Specificity) 계산
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print(f"특이도: {specificity}")

# ROC 곡선 및 AUC 계산
fpr, tpr, thresholds = roc_curve(df['y'], df['pred_prob'])
roc_auc = roc_auc_score(df['y'], df['pred_prob'])

# ROC 곡선 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# AUC 출력
print(f"AUC: {roc_auc}")
