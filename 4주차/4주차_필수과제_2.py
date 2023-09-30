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

# ## 필수과제 2
# - 시계열데이터는 제가 구글드라이브에 업로드할 예정입니다.
# 2. 만약 시계열 데이터라면? 추세가 반영될 수 있기 때문에 선형보간이 더 좋은 결과가 나올 수 있다.
# - 시계열데이터를 공유할 예정 -> 해당 데이터를 가지고 결측값을 만들고 오늘 배웠던
# - 평균 또는 최빈값등 기초통계량
# - 선형보간 등 interpolation
# - iterative imputer
#
# - 시계열데이터는 어떤 식으로 결측값을 대체 했을 때 더 좋은 결과가 나오지는 확인하기!
# - 다만 시계열데이터를 결측값을 만들기 전에  기존 데이터의 분포와 관계, 컬럼들에 대한 관계들을 확인해 보시고 ( 간단한 시각화로 )
# - 결측치를 만들어서 결과를 비교해 주세요.
# - 최종 결과물은  PDF파일로 정리하여 5장 내로 결과만 정리해서 공유해 주세요!
#
# [필수과제 시계열 데이터 설명]
# - seattle-weather.csv ( 구글 업로드 완료 )
# - kaggle 출처
# - 시애틀 데이터 컬럼은 너무 직관적이라 설명 생략 
# - y값 precipitation 
# - 사용 컬럼 temp_max, temp_min, wind 
# - wather 컬럼은 인코딩이 필요해서 아시는 분은 사용하시고 사용 안 하셔도 됩니다.
#
# ## 꼭 참고할 게 시계열 데이터이므로 train_test_split 사용하면 안 됩니다! 
# 따라서 인덱스 기준으로 데이터 잘라서 진행해 주시길 바랍니다. 
# 전체 데이터 중 7:3 비중으로 자르기 
#
# 그리고 결측값은 모든 컬럼에 최소 50개 이상은 만들어 주세요!
# ( 만약 여유로우신 분은 컬럼 별로 다르게 결측치를 만들어서 100, 500, 1000개 로 만들어 보시고 진행하셔도 좋습니다 ! )
#
# 과제 잘 부탁드립니다!!

# <br><br><br><br>

# # 1. 라이브러리 부착/데이터 불러오기/결측치 포함한 데이터셋 생성

# +
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# date가 object로, weather는 category로
data = pd.read_csv('../data/seattle-weather.csv')
data.info()
display(data.head())
print(data.weather.unique())
# -

data.date = pd.to_datetime(data.date)
data.weather = data.weather.astype('category')
data.info()

# 결측치 없어서 만들어야 함. => 50개
data.isnull().sum()
# 과제에서 주어진 컬럼으로만 df 구성하기(수치형 변수)
df = data.iloc[:,1:5]
NA_index = np.random.choice(np.arange(1,1462), size = 100, replace = False)
df.iloc[NA_index,:] = np.nan
df.isnull().sum()

# # 2. 각 방법론에 따른 결측치 대체

# ## 1. SimpleImputer

# +
from sklearn.impute import SimpleImputer
def NanTransform(method):
    imp = SimpleImputer(missing_values = np.nan, strategy = method)
    trans = imp.fit_transform(df.copy())
    df_method = pd.DataFrame(trans, columns = df.columns.tolist())
    return df_method

# 1. 평균 대체
df_mean = NanTransform('mean')
print(df_mean.isnull().sum())

## 2. 중앙값 대체
    
df_median = NanTransform('median')
print(df_median.isnull().sum())

## 3. 최빈값 대체

df_mf = NanTransform('most_frequent')
print(df_mf.isnull().sum())


# -

# ## 2. interpolate

# +
# method = 'linear'
def NanCount(dataframe):
    print(dataframe.isnull().sum())
df_linear = df.copy()
df_linear.interpolate(method = 'linear', inplace = True)
NanCount(df_linear)

# method = 'spline'
df_spline_1 = df.copy().interpolate(method = 'slinear')
NanCount(df_spline_1)

df_spline_2 = df.copy().interpolate(method = 'spline', order = 2)
NanCount(df_spline_2)
# -

# ## 3. IterativeImputer 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# 50개로 동일 개수의 결측치를 보유하고 있으므로 order는 정하지 않음.
imputer = IterativeImputer(max_iter = 100,
                          random_state = 0,
                          n_nearest_features = None) # 모든 feature 참조
df_imp = imputer.fit_transform(df)
df_iterative = pd.DataFrame(df_imp, columns = df.columns.tolist())
NanCount(df_iterative)

# # 3. 모델 학습 
# - 시계열 데이터이므로 train_test_split을 통해 데이터셋을 분리하면 안 됨  
#     => 인덱스 슬라이싱으로 (7:3으로 주어짐.) 

# +
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

train_count = math.ceil(len(df)*0.7)

def train_test_split(dataframe):
    x_train = dataframe.iloc[0:train_count+1,1:]
    x_test = dataframe.iloc[train_count+1:,1:]
    y_train = dataframe.iloc[0:train_count+1,0]
    y_test = dataframe.iloc[train_count+1:,0]
    return x_train, x_test, y_train, y_test

def model_fit(trainX,trainY):
    train_const = sm.add_constant(trainX) # 상수항을 추가해야 모델이 정확해짐.
    reg_fit = sm.OLS(trainY, train_const).fit()
    return reg_fit

def draw_plot(testX, testY,title):
    plt.figure()
    test_const = sm.add_constant(testX) # 동일하게 추가.
    plt.plot(np.array(reg_fit.predict(test_const)), label = "Pred")
    plt.plot(np.array(testY), label = "Real Value")
    plt.title(title)
    plt.legend()
    
def MSE(x,y):
    test_const = sm.add_constant(x)
    mse = mean_squared_error(y_true = y, y_pred = reg_fit.predict(test_const))
    return mse
    


# +
# 평균
x_train, x_test, y_train, y_test = train_test_split(df_mean)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Mean_Imp')
mean_mse = MSE(x_test,y_test)

# 중앙값
x_train, x_test, y_train, y_test = train_test_split(df_median)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Median_Imp')
median_mse = MSE(x_test,y_test)

# 최빈값
x_train, x_test, y_train, y_test = train_test_split(df_mf)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Most_freq_Imp')
mf_mse = MSE(x_test,y_test)


# 선형보간
x_train, x_test, y_train, y_test = train_test_split(df_linear)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Linear_Imp')
linear_mse = MSE(x_test,y_test)


# 스플라인 보간
x_train, x_test, y_train, y_test = train_test_split(df_spline_1)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Slinear_Imp')
spline_1_mse = MSE(x_test,y_test)

x_train, x_test, y_train, y_test = train_test_split(df_spline_2)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test,'Quadratic_Imp')
spline_2_mse = MSE(x_test,y_test)


# IterativeImputer
x_train, x_test, y_train, y_test = train_test_split(df_iterative)
reg_fit = model_fit(x_train, y_train)
draw_plot(x_test, y_test, 'Iterative_Imp')
iterative_mse = MSE(x_test,y_test)

# -

# 강수량이 음수가 나올 수 없는데 quadratic 보간에서 음수값이 나와서 mse가 다른 기법보다 2배 가까이 큼.
# 날씨라는 범주형 변수를 원 핫 인코딩을 통해 사용하였다면 mse가 더 줄었을 것 같다.
pd.DataFrame(index = ['평균 대체', '중앙값 대체', '최빈값 대체', '선형 보간', '스플라인 보간(1차)', '스플라인 보간(2차)', 'IterativeImputer'],
             data = [mean_mse, median_mse, mf_mse, linear_mse, spline_1_mse, spline_2_mse, iterative_mse], columns = ['기법'])


