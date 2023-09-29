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

import pandas as pd
import numpy as np
import missingno as msno
import warnings as wn
from IPython.display import display as dp
import matplotlib.pyplot as plt
wn.filterwarnings('ignore')
bike = pd.read_csv('../data/bike_sharing_daily.csv')
dp(bike.head())
dp(bike.tail())

# dteday는 datetime이 아님.
# yr=0이면 2011년, yr=1이면 2012년인 듯
# 우선 non-null count가 731이 안되는 컬럼을 확인함.
bike.info()

# 결측치 수치 확인 및 시각화
print(bike.isnull().sum())
msno.matrix(bike)

# 우선 데이터 타입 변경 dteday를 인덱스로
bike['dteday'] = pd.to_datetime(bike['dteday'])

# method = spline
bike['hum'].interpolate(method='spline', order=1).plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# method = linear
bike['hum'].interpolate(method='linear').plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# NA값을 그냥 중앙값으로 대체
bike['hum'].fillna(bike['hum'].describe().median()).plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

bike.set_index('dteday', inplace = True)
# 결측치가 존재하는 채로 습도 플롯 찍어보기
bike['hum'].plot()
plt.show() 

# method='time'을 이용한 시계열 데이터 보간법 => 시간 간격이 일정하지 않을 경우 더 유용.
# 빨간 부분이 보간된 부분 => 연속적인 그래프를 확인 가능.
#
bike['hum'].interpolate(method='time').plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# ## method = 'polynomial':다항식 보간법
# - 주어진 데이터 포인트 모두를 통과하는 하나의 다항식 찾음.
# - 데이터 포인트가 많아질수록 차수가 높아져 overfitting 될 가능성이 높음.

# polynomial, order = 2  == quadratic
bike['hum'].interpolate(method='polynomial',order=2).plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# ## method = 'nearest' : 최근접 보간법
# - 결측치 보간 시 가장 가까운 인접 데이터 포인트 사용

dp(bike[bike['hum'].isnull() == True][['hum']].head(10))
nearest=bike[['hum']].interpolate(method='nearest')
nearest.iloc[9:15,]

bike['hum'].interpolate(method='nearest').plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# ## method = 'values'
# - y축의 실제 값에 대해 선형 보간
# - 비정규 간격의 데이터에 유용

dp(bike[bike['hum'].isnull() == True][['hum']].head(10))
values=bike[['hum']].interpolate(method='values')
values.iloc[9:15,]

bike['hum'].interpolate(method='values').plot(color='red')
bike['hum'].plot(color='blue')
plt.show()

# ## method = 'krogh' 
# - 각 데이터 포인트에서 값과 기울기 고려

dp(bike[bike['hum'].isnull() == True][['hum']].head(10))
krogh=bike[['hum']].interpolate(method='krogh')
krogh.iloc[9:15,]

bike['hum'].interpolate(method='krogh').plot(color='red')
bike['hum'].plot(color='blue')
plt.show()
