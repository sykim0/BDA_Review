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

# ## 필수과제 1
# 1. simpleimputer (간단한 개념이라 리서치하시고 실제 코드 작성하시고 값에 대해서 결과 정리까지 부탁드립니다. )
# - 다양하게 통계치를 지정할 수 있다. ( 평균, 최빈값이 등등 ) 결과가 어떤식으로 바뀌는지 mpg 데이터를 가지고 확인해 주세요!
# - 최종 결과물은  PDF파일로 정리하여 5장 내로 결과만 정리해서 공유해 주세요!

# # SimpleImputer
# - scikit-learn에서 제공.
# - 각 컬럼의 기술통계(평균, 중앙값, 최빈값)등을 사용하여 NA 대체
# - 단변량 대체기. (IterativeImputer: 다변량 대체기)
#
# ## 파라미터
# - strategy: 평균, 중앙값, 최빈값, 상수 중 어떤 기법으로?
# - fill_value: 'constant(상수)' strategy 사용 시 대체할 특정 값
# - copy:데이터 복사본을 만들지 boolean type
# - add_indicator: NA값의 위치를 나타내는 이진 지표를 출력에 추가할지
# - keep_empty_features: 모든 값이 NA인 feature를 결과에 포함할지

# ## 1. 필요한 라이브러리 부착 및 데이터 불러오기

from IPython.display import display
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
mpg = sns.load_dataset('mpg')
mpg.head()

# 단변량에 대해서만 대체 가능하므로 horsepower에 대해서 NA 대체 진행.
mpg.isnull().sum()

# ## 2. 평균/중앙값/최빈값/상수 대체
# 모든 순서는  
# 1. 사용할 모형 설정
# 2. 데이터로 학습 => 반드시 numpy array 타입으로 학습
# 3. 모델 성능 평가

horsepower = mpg.loc[:,'horsepower']
horsepower.shape

# array shpae 변경
np_horsepower = np.array(horsepower).reshape(-1,1)
np_horsepower.shape

# +
# 각 strategy에 따른 모델 학습

# 평균
horsepower_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
horsepower_mean.fit(np_horsepower)

# 중앙값
horsepower_median = SimpleImputer(missing_values = np.nan, strategy = 'median')
horsepower_median.fit(np_horsepower)

# 최빈값
horsepower_mf = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
horsepower_mf.fit(np_horsepower)

# 상수 (임의로 100 사용)
horsepower_const = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 100)
horsepower_const.fit(np_horsepower)
# -

# 대체
mean_trans = horsepower_mean.transform(np_horsepower)
median_trans = horsepower_median.transform(np_horsepower)
mf_trans = horsepower_mf.transform(np_horsepower)
const_trans = horsepower_const.transform(np_horsepower)

# +
# 위의 학습-대체 과정을 fit_transform을 이용하여 한 번에 처리 가능
# horsepower_mean.fit_transform(np_horsepower)
# horsepower_median.fit_transform(np_horsepower)
# horsepower_mf.fit_transform(np_horsepower)
# horsepower_const.fit_transform(np_horsepower)
# -

# ## 3. 대체된 값 확인

# 원래 데이터셋에서 nan값의 행 인덱스 찾기
# 어차피 horsepower만 결측치이므로 df에서의 방식 사용해도 되고
mpg[mpg.isnull().any(axis = 1)].index.tolist()

# series에서는 이렇게 찾으면 됨.
a= mpg.loc[:,'horsepower']
NA_index = a[a.isnull()].index.tolist()
NA_index

mean_trans = mean_trans.reshape(-1)
median_trans = median_trans.reshape(-1)
mf_trans = mf_trans.reshape(-1)
const_trans = const_trans.reshape(-1)

# +
# 결측치 올바르게 대체된 것 확인됨.
mean_trans = pd.Series(mean_trans)
median_trans = pd.Series(median_trans)
mf_trans = pd.Series(mf_trans)
const_trans = pd.Series(const_trans)
mean_trans

data = {
    'original': [np.nan]*6,
    'mean': mean_trans[NA_index].tolist(),
    'median': median_trans[NA_index].tolist(),
    'most_freq':mf_trans[NA_index].tolist(),
    'constant': const_trans[NA_index].tolist()
}

df = pd.DataFrame(data)
df.index = NA_index
df

# +
# 예측값과 실제값 비교를 위해 df로 만들 것.
mpg['horsepower'] = mean_trans
df_means = mpg.copy()

mpg['horsepower'] = median_trans
df_median = mpg.copy()

mpg['horsepower'] = mf_trans
df_mf = mpg.copy()

mpg['horsepower'] = const_trans
df_const = mpg.copy()

mpg.info()
# -

# object형 컬럼은 제외할 것.
# 결측치 없는 것도 확인완료.
dfs = [df_means, df_median, df_mf, df_const]
[i.drop(columns = ['origin', 'name'], inplace = True) for i in dfs]
[i.isnull().sum() for i in dfs]

# ## 4. 플롯을 통해 예측값과 실제값 비교
# 위의 데이터셋을 통해 train/test로 분리 후 모델을 학습 시켜서 확인해보면 된다.

# ### 평균 대체 데이터셋

# +
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

x_train, x_test, y_train, y_test = train_test_split(df_means.drop(columns='horsepower'),df_means['horsepower'],
                                                   test_size = 0.3, random_state = 1)
# -

# 모델 학습.
reg_fit = sm.OLS(y_train, x_train)
reg_fit = reg_fit.fit()

# +
# 실제값 vs 예측값

import matplotlib.pyplot as plt
plt.plot(np.array(reg_fit.predict(x_test)), label = "Pred")
plt.plot(np.array(y_test), label = "Real Value")
plt.legend()
plt.show()
# -

# 무지 큼.
from sklearn.metrics import mean_squared_error
MSE_mean = mean_squared_error(y_true = y_test, y_pred = np.array(reg_fit.predict(x_test)))
MSE_mean

# ### 중앙값 대체 데이터 셋

x_train, x_test,y_train, y_test = train_test_split(df_median.drop(columns = 'horsepower'),
                                                  df_median['horsepower'])

reg_fit = sm.OLS(y_train, x_train).fit()

plt.plot(np.array(y_test), label = "Real Value")
plt.plot(np.array(reg_fit.predict(x_test)), label = "Pred")
plt.legend()
plt.show()
MSE_median = mean_squared_error(y_true = y_test, y_pred = reg_fit.predict(x_test))
MSE_median

# ### 최빈값 대체

x_train, x_test,y_train, y_test = train_test_split(df_mf.drop(columns = 'horsepower'),
                                                  df_mf['horsepower'])
reg_fit = sm.OLS(y_train, x_train).fit()
plt.plot(np.array(y_test), label = "Real Value")
plt.plot(np.array(reg_fit.predict(x_test)), label = "Pred")
plt.legend()
plt.show()
MSE_mf = mean_squared_error(y_true = y_test, y_pred = reg_fit.predict(x_test))
MSE_mf

# ### 상수값(100)으로 대체

x_train, x_test,y_train, y_test = train_test_split(df_const.drop(columns = 'horsepower'),
                                                  df_const['horsepower'])
reg_fit = sm.OLS(y_train, x_train).fit()
plt.plot(np.array(y_test), label = "Real Value")
plt.plot(np.array(reg_fit.predict(x_test)), label = "Pred")
plt.legend()
plt.show()
MSE_const = mean_squared_error(y_true = y_test, y_pred = reg_fit.predict(x_test))
MSE_const

print(f''' MSE:

    평균 대체:{MSE_mean},
    중앙값 대체:{MSE_median},
    최빈값 대체:{MSE_mf},
    상수 대체:{MSE_const}
''')
