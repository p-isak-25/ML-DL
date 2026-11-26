from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

practice_dir = Path(__file__).resolve().parent


#1차원 및 2차원 벡터의 행렬
one_dim_vec = np.array([1,2,3,4,5])
print(one_dim_vec)
print("-------------------")
two_dim_vec = np.array([[1,2,3], [4,5,6]])
print(two_dim_vec)

#1차원 및 2차원 벡터의 크기
print(one_dim_vec.shape)
print(two_dim_vec.shape)

#3차원 벡터
three_dim_vec = np.array([[[1,2,3], [4,5,6], [7,8,9]],
                          [[2,2,2], [3,3,3], [4,4,4]]])
print(three_dim_vec)
print(three_dim_vec.shape)

#한글경로깨짐짐
import cv2
image_path = practice_dir / "dog.jpg"
if not image_path.is_file():
    image_path = next((practice_dir.parents[1] / "classification_project").glob("project_data_image_*.png"))
    print(f"이미지 실습에 프로젝트의 예제 그림을 사용합니다: {image_path.name}")
image = cv2.imread(str(image_path))

#한 픽셀의값 194, 232, 250
print(image)
print(image.shape)


#배열생성: 0~9 수 
arr1 = np.arange(10)
print(arr1)

#1~까9지 2칸씩 띄어서 배열 생성
arr2 = np.arange(1, 10, 3)
print(arr2)

#0~11 숫자로 배열생성 1행 12열
arr = np.arange(12)
print(arr)
print("----------------------")
reshaped_arr = arr.reshape(3, 4)
print(reshaped_arr)

# np.random() 함수 
# 이외에도 np.random.rand, np.random.randn 함수가 있음
# 0~1 사이의 요소를 가진 3x3배열 생성성
rand_arr = np.random.random((3, 3))
print(rand_arr)

#슬라이싱 "[:]" 
#[a, :] a행 전부
#[:, b] b열 전부부
arr5 = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

print(arr5[0, :])  
print("--------------------")
print(arr5[:, 1])  
print("--------------------")
print(arr5[:2, :2])  #상단 2x2부분


#차원을 다루는 numpy와 달리 data를 다루는 pandas
sr = pd.Series([17000, 18000, 1000, 5000],
               index = ["피자", "치킨", "콜라", "맥주"])
print(sr)

values = [[1,2,3],[4,5,6],[7,8,9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index = index, columns = columns)
print(df)

data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
 ]
df1 = pd.DataFrame(data)
print(df1)
print("----------------------------")

data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 
'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
}
df2 = pd.DataFrame(data)
print(df2)
print("--------------------")

#판다스는 문제지에대한 정답을 알려주는 역할을 한다.
def read_practice_excel(name, sample, *, has_header=True):
    path = practice_dir / name
    if path.is_file():
        return pd.read_excel(path, header=0 if has_header else None)

    print(f"{name} 실습에 예제 데이터를 사용합니다.")
    workbook = BytesIO()
    sample.to_excel(workbook, index=False, header=has_header, engine="openpyxl")
    workbook.seek(0)
    return pd.read_excel(workbook, header=0 if has_header else None)


df = read_practice_excel("Example.xlsx", df2)
print(df)

df = read_practice_excel(
    "spam.xlsx",
    pd.DataFrame([["일반 메시지", 0], ["광고 메시지", 1]]),
    has_header=False,
)
print(df)
print("------------------------------")

x = df.iloc[:, 0]
y = df.iloc[:, 1]

print(x)
print("------------------------------")
print(y)


plt.title("test")
plt.plot([1,2,3,4], [2,4,6,8])
plt.xlabel("hours")
plt.ylabel("score")
plt.show()

plt.title("Students")
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1.5,2.5,3.5,4.5], [3,5,8,10])
plt.xlabel("hours")
plt.ylabel("score")
#두그래프를 동시에 보며 비교가능
plt.legend(['A student', 'B student'])
plt.show()
