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

# # 문법 복습

# ## 2차원 리스트

# 데이터 프레임은 2차원 리스트와 유사하다고 생각하자.
a = [
    [1,2],
    [3,4]
]
a[0],a[1],a[0][1]

# ## 데이터프레임

# +
# 데이터프레임 간단하게 만들어보기
import pandas as pd
data = {
    'IT융합대학': ['컴퓨터공학과', 'SW학과', '전자공학과'],
    '사회과학대학': ['응용통계학과', '유아교육학과', '미디어커뮤니케이션학과'],
    '경영대학' : ['경영학과', '금융수학과', '경제학과']
}

df = pd.DataFrame(data)
df

# +
# 각 컬럼의 길이가 다르면 안 만들어짐.
# 데이터프레임 간단하게 만들어보기
data = {
    'IT융합대학': ['컴퓨터공학과', 'SW학과', '전자공학과',''],
    '사회과학대학': ['응용통계학과', '유아교육학과', '미디어커뮤니케이션학과'],
    '경영대학' : ['경영학과', '금융수학과', '경제학과']
}

df = pd.DataFrame(data)
df
# -

# ## 데이터 접근

#mpg 데이터 불러오기
mpg = pd.read_csv('../data/mpg.csv')
mpg

# 원하는 컬럼 가져오기
# mpg['manufacturer'] # 시리즈 형태로 가져옴
mpg[['manufacturer']]

# 인덱스 슬라이싱
mpg[2:100]
# mpg[2:101:2]

# loc는 인덱스명 기준
mpg.loc[2:101:2]

# iloc는 데이터프레임에서의 실제 순서
mpg.iloc[2:101:2]

# ## 파생변수 만들어보기

# 데이터 먼저 확인
mpg.info()

# 의미는 두지 말고 아무런 변수나 만들어보자!
mpg['any'] = mpg['displ'] * mpg['cyl']
mpg[['any']]

# 기술 통계 확인하기
mpg.describe()

mpg.sum(numeric_only=True)










