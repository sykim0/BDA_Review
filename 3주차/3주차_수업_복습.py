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

# # 결측치와 누락값, 이상치
# 결측치
# - NA값 그 자체
# - 사람의 실수로 인해 누락된 값(누락치)
#
# => 결측치는 데이터 분석/머신러닝에 방해가 됨.
# => 처리할 필요가 분명히 있음(여러 가지 방법론에 따라 처리)

# ## 결측치 처리 방법

# ### 0. 확인

# 데이터프레임 여러 개를 프레임 형태로 출력하기 위해
from IPython.display import display
# missingno: 결측치 데이터 시각화 패키지
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# +
bike = pd.read_csv('../data/bike_sharing_daily.csv')
country = pd.read_csv('../data/country_timeseries.csv')

# info()를 통해서도 바로 결측치 존재유무는 확인 가능.
bike.info()
display(bike.head())
country.info()
display(country.head())
# -

# 직접적으로 결측치 수 확인해보기
# 컬럼별로 결측치 수 확인
print(bike.isnull().sum(),end="\n\n")
print(country.isnull().sum())

# 결측치 시각화
# 결측치 패턴 확인 가능
msno.matrix(bike)
msno.matrix(country)
plt.show()

# 막대 그래프로 확인
msno.bar(bike)
plt.show()

msno.bar(country)
plt.show()

# ### 1. 제거

# dropna: NA값이 있는 row는 삭제
display(bike)
display(bike.dropna()) # NA가 1개라도 존재하는 row 약 100개 정도 삭제

# ### 2. 통계값 대체

# fillna() 원하는 값으로 NA값을 대체
display(country)
country.fillna(country.mean(numeric_only = True)) # 대체되었으므로 NA를 포함하는 df와 shape 동일

# ffill(front fill): 누락값이 나타나기 직전의 값으로 누락값 변경
# bfill(back fill): 누락값이 나타난 직후의 값으로 누락값 변경
display(country[['Cases_Guinea']])
display(country[['Cases_Guinea']].fillna(method='ffill'))
display(country[['Cases_Guinea']].fillna(method='bfill'))

# ### 3. 보간법
# - 선형 보간
# - 스플라인 보간
# - 시계열 데이터 보간

# interpolate(method=""): 누락값을 평균으로 대체
# 3번 row가 NA였으므로 2번과 4번 row의 평균값으로 대체됨.
display(country[['Cases_Guinea']].interpolate())

# 결측값을 대체하지 않으면 그래프가 연속적이지 않게 됨.
country['Cases_Guinea'].plot()

# 따라서 결측치를 대체해야함.
#
country['Cases_Guinea'].interpolate().plot()
country['Cases_Guinea'].fillna(method = 'ffill').plot()
country['Cases_Guinea'].fillna(method = 'bfill').plot()

# - 선형 및 스플라인 보간법

# interpolate()의 method를 지정하지 않으면 기본적으로 linear
# 두 점 사이의 거리로 대체(=결측값 직전/후 데이터의 평균값)
# 일치함을 확인 가능.
country['Cases_Guinea'].interpolate(method = 'linear').plot()
country['Cases_Guinea'].interpolate().plot()

# +
country['Cases_Guinea'].interpolate().plot()

# 항을 높이면서 보간
# 선형 스플라인 => 각 두 점 사이를 직선 세그먼트로 연결
country['Cases_Guinea'].interpolate('slinear').plot()

# 3차 스플라인 보간법: 더 매끄러운 곡선 가능.
# method='spline', order=3 이랑 동일
country['Cases_Guinea'].interpolate('cubic').plot()
# -

# order: 차수
country['Cases_Guinea'].interpolate(method='spline',order=2).plot()
plt.show()

# 시계열 데이터 보간법
country.info() # Date가 object형
country['Date'] = pd.to_datetime(country['Date'])
country.set_index('Date', inplace = True)
country.info()

# 시계열로 대체할 때는 양쪽의 값을 가지고 NA을 대체
# 반드시 index가 시계열 데이터여야 함.
country['Cases_Guinea'].interpolate(method='time')


