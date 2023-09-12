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

# # 2주차
# ## 데이터프레임 기초문법(Pandas)
# pandas
# - 2주차에 다룬 것
#     - query()
#     - [[]]
#     - sort_values()
#     - groupby()
#     - assign(), lambda
#     - agg
# - 다루지 않은 것
#     - merge
#     - concat()
#     - loc, iloc
#     - str

# 데이터 불러오기
import pandas as pd
exam = pd.read_csv('../data/exam.csv')
exam.head(5)

# ### query()
# - 행 데이터 추출 시에 사용

exam.nclass.dtypes

# id, nclass는 범주형이므로 category로 변경해줌.
exam.id = exam.id.astype('category')
exam.nclass = exam.nclass.astype('category')
exam.info()

# 기술통계 확인하기
exam.describe()

# 기술통계를 확인해서 영어 점수가 평균 이상인 학생들을 확인해보자.
eng_mean = exam.describe()['english'].loc['mean']
exam.query(f'english >= {eng_mean}')

# 각 과목에서 평균 이상의 점수를 받은 학생들을 조회해보자
math_mean = exam.describe()['math'].loc['mean']
sci_mean = exam.describe()['science'].loc['mean']
print(f'수학 평균: {math_mean}, \n과학 평균: {sci_mean}, \n영어 평균: {eng_mean}')
exam.query(f'math >= {math_mean} and science >= {sci_mean} and english >= {eng_mean}')

# ### [], [[]]
# - 열 추출 시에 사용

# 학급과 수학 점수만 추출
nclass_math = exam[['nclass', 'math']]
nclass_math

# 학급은 필요없어서 삭제해보자
math = nclass_math.drop(columns ='nclass')
math

# +
# 만약 nclass_math 자체에서 학급 열을 삭제하고 싶다면
# inplace를 True를 기억하자.
# nclass_math.drop(columns=['nclass'], inplace = True)
# -

# ## 정렬

# 수학을 기준으로 오름차순 정렬
exam.sort_values('math')

# 내림차순
exam.sort_values('math', ascending = False)

# 쿼리 메서드와 함께 이용
exam.sort_values('math', ascending=False)[['math','id']].query('math>30')

# 2개 이상 조건을 가지고 정렬
# 먼저 수학을 기준으로 오름차순 정렬 후 점수가 같다면 id로 정렬
exam.sort_values(['math', 'id'])

# ascending을 통해 따로따로 정렬 기준을 부여할 수도 있음.
exam.sort_values(['math', 'id'], ascending= [True, False])

# ### 파생변수 만들기

# 1. 컬럼을 가지고 직접 만들기

import numpy as np
exam['best_1'] = exam[['math', 'science', 'english']].max(axis = 1)
exam.head()

# 2. lambda 이용, assign())

# assign은 새로운 컬럼을 만들 때 사용됨. 
exam = exam.assign(best_2 = exam[['math', 'science', 'english']].max(axis = 1))
exam

# lambda와 assign
# df = df.assign(사용할 변수명 = lambda x: (...))
exam = exam.assign(best_3 = lambda x: x[['math', 'science', 'english']].max(axis = 1))
exam

# ### groupby

# 학급별로 묶어서 데이터를 조회해보자
# 그룹화 후에는 기술통계값만 조회 가능
exam.groupby('nclass').agg(math_mean = ('math', 'mean'))

# 단일 컬럼에 대해서는  이렇게도 조회 가능
exam.groupby('nclass')['math'].mean()

# 반별로 여러 가지 기술통계량도 사용가능
exam.groupby('nclass').agg(math_mean = ('math', 'mean'),
                          eng_sum = ('english', 'sum'),
                          sci_var = ('science', 'var'))

# 하지만 기술통계 조회 그냥 하는 것이 더 나을듯
exam[exam['nclass']==1].describe()










