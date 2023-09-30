# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 보간법과 보외법
# ### 보간법(interpolation)
#     - 범위 안의 값을 예측
#     - 선형, 스플라인 등 존재
# ### 보외법(extrapolation)
#     - 범위 바깥의 있는 값을 예측  
#     
# ![image-2.png](attachment:image-2.png)
#
# ## mpg 데이터를 통한 결측치 대체
# ### 변수 설명
# - mpg: 연비 => 높을수록 효율 좋음.
# - cylinders: 실린더 => 많을 수록 마력이 크게 나옴.
# - displacement: 배기량 => 클수록 마력은 크지만 연비가 낮아짐.
# - horsepower: 마력(엔진 출력 단위) => 클수록 자동차 성능이 좋음.
# - weight: 무게
# - acceleration:제로백 => 정지->일정 속도까지 가속하는데 걸리는 시간

# +
# 라이브러리 부착 및 데이터 불러오기

# 누락값을 가진 feature를 제외한 나머지 feature를 사용하여 해당 feature 값을 예측하는 회귀 모델 훈련 
from sklearn.experimental import enable_iterative_imputer
# enable_iterative_imputer를 먼저 import 해야함.
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
mpg = sns.load_dataset('mpg')
mpg.info() # horsepower 결측치 존재
# -

# ## 지난 주 떠올리기
# 결측치를  
# - 기술통계로의 대체(평균, 중앙값 등)
# - 수학적 함수에 의한 보간법
#     - linear, spline 등등
#     
# ## 오늘 목표
# - scikit-learn imputer를 통해 실제 결측값 대체
# - 해당 결측값 대체한 것을 회귀분석을 통해 MSE차이가 어떤지 비교
#     - MSE에 대한 개념과 예측값/실제값의 차이
# - interpolation이 과연 옳은지 판단

# 결측치 확실히 확인
mpg.isnull().sum()

# horsepower 이외로도 몇 개 feature에 강제로 NA값 주입
mpg.dropna(inplace = True)
mpgWithNa = mpg[['displacement', 'horsepower', 'weight', 'acceleration']]
mpgWithNa.info()

# 임의의 인덱스에 NA값 생성 과정(인덱스 중복 허용 X)
NA_index = np.random.choice(np.arange(0,393), size = 50, replace = False)
mpgWithNa.iloc[NA_index,:] = np.nan
mpgWithNa.isnull().sum()

# mpg feature 추가
# 하나씩 보기
mpgConcat = pd.concat([mpg['mpg'], mpgWithNa], axis = 1)
mpgConcat.horsepower.plot()
print(display(mpgConcat))
mpgConcat_2 =  pd.concat([mpg['mpg'], mpgWithNa], axis = 1)
mpgConcat_3 =  pd.concat([mpg['mpg'], mpgWithNa], axis = 1)
mpgConcat_4 = pd.concat([mpg['mpg'], mpgWithNa], axis = 1)

# ## 평균으로 대체

mpgWithNa.describe().loc['mean',:]

# 잘 대체됨.
mpgConcat['displacement'] = mpgConcat['displacement'].fillna(mpgWithNa.displacement.mean())
mpgConcat['horsepower'] = mpgConcat['horsepower'].fillna(mpgWithNa.horsepower.mean())
mpgConcat['weight'] = mpgConcat['weight'].fillna(mpgWithNa.weight.mean())
mpgConcat['acceleration'] = mpgConcat['acceleration'].fillna(mpgWithNa.acceleration.mean())
mpgConcat.isnull().sum()

# apply로 과정 줄이기
cols = ['displacement', 'horsepower' , 'weight', 'acceleration']
mpgConcat[cols] = mpgConcat[cols].apply(lambda x: x.fillna(mpgWithNa[x.name].mean()))
print(mpgConcat.isnull().sum())
display(mpgConcat)

# +
# 데이터셋 분리
from sklearn.model_selection import train_test_split

# 회귀분석에 필요한 모듈
import statsmodels.api as sm

# mpg가 종속변수, 나머지가 독립변수임.
x_train, x_test, y_train, y_test = train_test_split(mpgConcat.drop(columns = 'mpg',axis = 1), mpgConcat['mpg'], test_size = 0.3, random_state = 1)
# -

x_train.shape,y_train.shape,x_test.shape, y_test.shape

reg_fit.predict(x_test).shape

# 다중회귀분석 진행하기
reg_fit = sm.OLS(y_train, x_train)
reg_fit

# 적합 회귀 모형 
# 추정
# 학습
reg_fit = reg_fit.fit()
reg_fit

reg_fit.predict(x_test)

# 예측 vs 실제
plt.plot(np.array(reg_fit.predict(x_test)), label = 'Prediction')
plt.plot(np.array(y_test), label = 'Real Value')
plt.legend()
plt.show()

# np.array()가 없으면 인덱스와 값이 살아있어서 그래프가 이상하게 찍힘.
reg_fit.predict(x_test), np.array(reg_fit.predict(x_test))

# 평균 대체를 통해 도출된 회귀식의 평가 지표
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true = y_test, y_pred = np.array(reg_fit.predict(x_test)))
MSE

# ## 보간법 대체

# 잘 대체됨.
mpgConcat_2['displacement'] = mpgConcat_2['displacement'].interpolate(method = 'linear')
mpgConcat_2['horsepower'] = mpgConcat_2['horsepower'].interpolate(method = 'linear')
mpgConcat_2['weight'] = mpgConcat_2['weight'].interpolate(method = 'linear')
mpgConcat_2['acceleration'] = mpgConcat_2['acceleration'].interpolate(method = 'linear')
mpgConcat_2.isnull().sum()

# apply를 통해 보간
cols = mpgWithNa.columns.tolist()
mpgConcat_2[cols] = mpgConcat_2[cols].apply(lambda x:x.interpolate(method = 'linear'))
mpgConcat_2.isnull().sum()

train_test_split(mpgConcat_2.drop(columns = 'mpg'), mpgConcat_2['mpg'], test_size = 0.3, random_state = 11)

# 훈련 데이터셋에 최소 제곱법 적용할 것임을 알림.
# sm.OLS(): y= b0+b1x1+b2x2+... (b는 베타) 라는 회귀방정식을 사용할 것임을 설정.
reg_fit_2 = sm.OLS(y_train, x_train)
reg_fit_2

# 모델 학습
# 모수 추정
reg_fit_2 = reg_fit_2.fit()
reg_fit_2

plt.plot(np.array(reg_fit_2.predict(x_test)), label = 'Prediction')
plt.plot(np.array(y_test), label = 'Real Value')
plt.legend()
plt.show()

MSE_2 = mean_squared_error(y_true = y_test, y_pred = reg_fit_2.predict(x_test))
MSE_2

# ## iterativeimputer로 대체

# imputation_order: feature에서 NA값이 많은/적은 순서대로 대체 시작. descending이므로 NA값이 가장 많은 feature부터 대체 시작.
# max_iter: NA 대체 작업을 지정한만큼 반복. 반복의 수는 정답이 없고 최적의 값은 스스로 찾아야 함.
# random_state: 시드 설정(공부 중이므로 같은 값만을 보기 위해)
# n_nearest_features: NA값 대체 시 참조할 feature 개수. None: 모든 feature 참조.
imputer = IterativeImputer(imputation_order = "descending",
                          max_iter = 10, random_state = 3,
                          n_nearest_features = 4)
#imputer를 통해 대체 작업.
mpgConcat_3 = imputer.fit_transform(mpgConcat_3)
mpgConcat_3 # 리턴은 np.array로

# df 변환 후 예측 진행
mpgConcat_3 = pd.DataFrame(mpgConcat_3, columns=mpgConcat_2.columns)
mpgConcat_3.isnull().sum()
mpgConcat_3.isnull()

x_train, x_test, y_train, y_test = train_test_split(mpgConcat_3.drop(columns = 'mpg', axis = 1), mpgConcat_3['mpg'],test_size=0.3, random_state=111)

reg_fit_3 = sm.OLS(y_train, x_train)
reg_fit_3 = reg_fit_3.fit()

plt.plot(np.array(reg_fit_3.predict(x_test)),label='Prediction')
plt.plot(np.array(y_test), label='Real Value')
plt.legend()
plt.show()

MSE_3 = mean_squared_error(y_true = y_test, y_pred = reg_fit_3.predict(x_test))
MSE_3

# MSE 비교
print(f'{MSE}: "단순평균대치"')
print(f'{MSE_2}: "선형보간"')
print(f'{MSE_3}: "iterative imputer"')

# # 지표 해석 방법
# 절대적으로 가장 좋은 대체 방법은 존재하지 않음. => 데이터에 대한 도메인 지식을 토대로 결정해야함.

# 데이터 분포 확인 => 매우 중요
sns.distplot(mpg['displacement'])

# 봉우리가 2개 존재
sns.distplot(mpg['horsepower'])

sns.distplot(mpg['weight'])

# 거의 종모양의 정규 분포
sns.distplot(mpg['acceleration'])






