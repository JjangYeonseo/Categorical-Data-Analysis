#1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from scipy import stats

# 데이터 불러오기
df = pd.read_csv("http://www.stat.ufl.edu/~aa/cat/data/Students.dat", sep="\s+")

# 컬럼 재배열
abor_col = df.pop('abor')
df['abor'] = abor_col

# 설명 변수 선택
x_pot = df[['abor', 'ideol', 'relig', 'news', 'hsgpa', 'gender']]
print(x_pot.head())

# 상관관계 분석
plt.figure(figsize=(10, 8))
sns.heatmap(x_pot.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 단변량 분석
def fit_logistic(X, y):
    model = sm.Logit(y, add_constant(X))
    results = model.fit(disp=0)
    return results

# 개별 변수에 대한 로지스틱 회귀
variables = ['ideol', 'relig', 'news', 'hsgpa', 'gender']
univariate_results = []

for var in variables:
    X = x_pot[var]
    y = x_pot['abor']
    results = fit_logistic(X, y)
    
    summary_df = pd.DataFrame({
        'Variable': [var],
        'Coefficient': [results.params[1]],
        'Std Error': [results.bse[1]],
        'z-value': [results.tvalues[1]],
        'p-value': [results.pvalues[1]]
    })
    univariate_results.append(summary_df)

univariate_results_df = pd.concat(univariate_results)
print("\nUnivariate Analysis Results:")
print(univariate_results_df)

# 다변량 분석
# Model 6: ideol + relig + news
X = x_pot[['ideol', 'relig', 'news']]
y = x_pot['abor']
model6 = fit_logistic(X, y)

# Model 7: ideol + relig
X = x_pot[['ideol', 'relig']]
y = x_pot['abor']
model7 = fit_logistic(X, y)

# Model 8: ideol + news
X = x_pot[['ideol', 'news']]
y = x_pot['abor']
model8 = fit_logistic(X, y)

print("\nMultivariate Analysis Results:")
print("\nModel 6 Summary:")
print(model6.summary().tables[1])
print("\nModel 7 Summary:")
print(model7.summary().tables[1])
print("\nModel 8 Summary:")
print(model8.summary().tables[1])

# 최종 모델 (Model A)
X = x_pot[['ideol', 'news', 'hsgpa']]
y = x_pot['abor']
final_model_a = fit_logistic(X, y)

# 간단한 모델 (Model B)
X = x_pot[['ideol', 'news']]
y = x_pot['abor']
final_model_b = fit_logistic(X, y)

# 모델 진단
def calculate_influence(model, X):
    influence = stats.outliers_influence.OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    return cooks_d

# Cook's Distance 계산
X_a = add_constant(x_pot[['ideol', 'news', 'hsgpa']])
cooks_d_a = calculate_influence(final_model_a, X_a)

X_b = add_constant(x_pot[['ideol', 'news']])
cooks_d_b = calculate_influence(final_model_b, X_b)

# Cook's Distance 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(cooks_d_a)), cooks_d_a)
plt.title("Cook's Distance - Model A")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")

plt.subplot(1, 2, 2)
plt.scatter(range(len(cooks_d_b)), cooks_d_b)
plt.title("Cook's Distance - Model B")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.tight_layout()
plt.show()

# 다중공선성 검정
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

print("\nVIF for Model A:")
print(calculate_vif(x_pot[['ideol', 'news', 'hsgpa']]))
print("\nVIF for Model B:")
print(calculate_vif(x_pot[['ideol', 'news']]))

# 변수 선택 (bestglm 대체)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer, log_loss

# 전진 선택법 구현
def forward_selection(X, y, max_features=None):
    if max_features is None:
        max_features = X.shape[1]
        
    sfs = SequentialFeatureSelector(
        LogisticRegression(random_state=42),
        n_features_to_select=max_features,
        direction='forward',
        scoring=make_scorer(log_loss, needs_proba=True)
    )
    
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()].tolist()
    return selected_features

# 변수 선택 실행
X = df.drop(['abor', 'subject', 'affil', 'life'], axis=1)
y = df['abor']

print("\nBest features (all):")
print(forward_selection(X, y))

print("\nBest features (max 3):")
print(forward_selection(X, y, max_features=3))

#2
import numpy as np

# 주어진 계수
coef_D_I = 3.3  # 민주당 대 무소속 (절편)
slope_D_I = -0.2  # 민주당 대 무소속 (소득 기울기)

coef_R_I = 1.0  # 공화당 대 무소속 (절편)
slope_R_I = 0.3  # 공화당 대 무소속 (소득 기울기)

# (a) log(πR/πD)를 계산하는 함수 정의
def log_piR_piD(x):
    return -2.3 + 0.5 * x

# (b) πR > πD가 성립하는 x의 범위 계산
x_threshold = 2.3 / 0.5
print("πR > πD가 성립하는 소득 범위: x >", x_threshold, "(즉, 소득 >", x_threshold * 10000, ")")

# (c) πI, πD, πR에 대한 예측 방정식 정의
def pi_I(x):
    denominator = 1 + np.exp(coef_D_I + slope_D_I * x) + np.exp(coef_R_I + slope_R_I * x)
    return 1 / denominator

def pi_D(x):
    denominator = 1 + np.exp(coef_D_I + slope_D_I * x) + np.exp(coef_R_I + slope_R_I * x)
    return np.exp(coef_D_I + slope_D_I * x) / denominator

def pi_R(x):
    denominator = 1 + np.exp(coef_D_I + slope_D_I * x) + np.exp(coef_R_I + slope_R_I * x)
    return np.exp(coef_R_I + slope_R_I * x) / denominator

#3
import numpy as np

# Given logistic regression coefficient for "Invertebrate" vs. "Other" with respect to length
coef_invertebrate_other = -2.4654

odds_ratio_invertebrate_other = np.exp(coef_invertebrate_other)
inverse_odds_ratio_invertebrate_other = 1 / odds_ratio_invertebrate_other

print("Odds Ratio (Invertebrate vs. Other):", odds_ratio_invertebrate_other)
print("Inverse Odds Ratio (for interpretation):", inverse_odds_ratio_invertebrate_other)
