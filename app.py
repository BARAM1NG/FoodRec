import streamlit as st
import folium
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tkinter.tix import COLUMN
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #torch.nn.functional.logsigmoid 함수 사용
import math

# 외부 API
from Latitude_longitude_API import get_restaurant_address, geocoding
from model import MLP, CustomDataset, Mapping, apply_log_to_columns

# warnings ignore
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Exception ignored in")

# MLP 로드
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MLP(8, device = DEVICE)
PATH = 'Toy_project_weight.pth'
model.load_state_dict(torch.load(PATH))


# DATA 로드
# 0: 가게 이름, 1: 주소, 2: 음식 종류, 3: 가격, 4: 네이버 지도 별점, 5: 방문자 리뷰?, 6: 주관적 별점?
# 한글 폰트 설정 
plt.rcParams['font.family'] = 'AppleGothic'

# 꽉찬 화면 구성
st.set_page_config(layout="wide")

# 페이지 구상
st.title('	:school: Kyung Hee University FoodRec Service')
st.write('---')

# 가고 싶은 식당 검색 input
title = st.text_input(
  label = '가고 싶은 식당이 있나요?',
  placeholder = '식당 이름을 입력해주세요'
)

# 데이터 로드
# data_csv = 'data.csv'
# sample_df = pd.read_csv(data_csv, header=0, encoding='utf-8')

# 가게이름, 음식 종류, 주소 데이터
# df = sample_df.iloc[:, 0:3]




# 화면 2분할
col2, col3 = st.columns(2)

data = pd.read_csv('data2.csv', header=0, encoding='utf-8')

# 식당 신뢰도 출력
with col2:
  
  # MLP 점수 입력
  if title == '':
      score = 0
  else:
      x = data[data.iloc[:, 0] == title].iloc[:, [2, 3, 4, 5, 6]]
      x = apply_log_to_columns(x)
      kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
      x[0] = kind[x[0]]
      tensor = torch.tensor(x).unsqueeze(0)
      tensor = model(tensor.to(DEVICE))
      score = Mapping(tensor).item() * 100
  
  st.header(f' 식당 점수: {score}점')
  if score >= 90:
    st.subheader('반드시 찾아가야 하는 식당 :+1:')
  elif score >= 80:
    st.subheader('한번쯤은 찾아가볼만한 식당 :ok_hand:')
  elif score >= 70:
    st.subheader('실패하지는 않는 식당 :pig_nose:')
  elif score >= 60:
    st.subheader('굳이 가지 않는 식당 :imp:')
  else:
    st.subheader('경희대 학생이라면 가지 않는 식당.. :-1:')

  option = st.selectbox(
    '추천 식당 위치 선택',
    ('정건', '중상')
  )
  
  food_kind = st.selectbox(
    '음식 종류',
    (['한식', '양식', '아시아음식', '일식', '중식', '분식', '카페', '뷔페', '기타'])
  )

#  new_df = df[df.iloc[:, 1] == food_kind]
#  restaurant_address = get_restaurant_address(title)
#  restaurant_location = geocoding(restaurant_address)


# 해당 식당 지도 화면 불러오기
with col3:

  def main():

    try:
        if title:
            restaurant_address = get_restaurant_address(title)
            if restaurant_address:
                restaurant_location = geocoding(restaurant_address)
                map = folium.Map(location=restaurant_location, zoom_start=40, control_scale=True)

                # 지도 마커 표시
                marker = folium.Marker(restaurant_location, popup=title, icon=folium.Icon(color='blue'))
                marker.add_to(map)
                
                map
            else:
                st.error("해당 식당의 주소를 찾을 수 없습니다.")
                
    except Exception as e:
        st.error(f"에러 발생: {e}")

  if __name__ == "__main__":
    main()


  
# 추천 식당 tab
tab1, tab2 = st.tabs(['추천 맛집 1', '추천 맛집 2'])


# 추천 맛집 
with tab1:
  st.header('추천 맛집 1')
  col4, col5 = st.columns(2)
  
  with col4:
    if option == '정건':
      if food_kind == '한식':
        # 여기에 해당 종류 위도 경도 데이터 로드
        location = geocoding(get_restaurant_address('방울엄마국밥'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '양식':
        location = geocoding(get_restaurant_address('소울피'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '아시아음식':
        location = geocoding(get_restaurant_address('메콩타이 영통점'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '일식':
        location = geocoding(get_restaurant_address('부타센세'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '중식':
        location = geocoding(get_restaurant_address('얜시부'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '분식':
        location = geocoding(get_restaurant_address('보용만두'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '카페':
        location = geocoding(get_restaurant_address('라이킷'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '뷔페':
        pass
      elif food_kind == '기타':
        location = geocoding(get_restaurant_address('도스마스'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map

    elif option == '중상':
      if food_kind == '한식':
        # 여기에 해당 종류 위도 경도 데이터 로드
        location = geocoding(get_restaurant_address('장터밥상'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '양식':
        location = geocoding(get_restaurant_address('차츰'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
        # folium.Marker([위도, 경도], popup='마커 이름').add_to(map)
      elif food_kind == '아시아음식':
        location = geocoding(get_restaurant_address('베트남쌀국수포포'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '일식':
        location = geocoding(get_restaurant_address('삿뽀로 수원점'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '중식':
        location = geocoding(get_restaurant_address('짬뽕타임'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '분식':
        location = geocoding(get_restaurant_address('보영만두'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '카페':
        location = geocoding(get_restaurant_address('콜링우드'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '뷔페':
        location = geocoding(get_restaurant_address('고메스퀘어'))
        map = folium.Map(location=location, zoom_start=16, control_scale=True)
        marker = folium.Marker(location, popup=title, icon=folium.Icon(color='blue'))
        marker.add_to(map)
        map
      elif food_kind == '기타':
        pass
        
  with col5:
    if option == '정건':
      if food_kind == '한식':
        x = data[data.iloc[:, 0] == '방울엄마국밥'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 방울엄마국밥')
        st.subheader('음식 종류: 한식')
        st.subheader('가격: 10,000원')
        st.subheader('구글 별점: 4.27')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '양식':
        x = data[data.iloc[:, 0] == '쏘울피'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 소울피')
        st.subheader('음식 종류: 양식')
        st.subheader('가격: 11,000원')
        st.subheader('구글 별점: 4.41')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '아시아음식':
        x = data[data.iloc[:, 0] == '메콩타이'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 메콩타이')
        st.subheader('음식 종류: 아시아음식')
        st.subheader('가격: 11,500원')
        st.subheader('구글 별점: 4.48')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '일식':
        x = data[data.iloc[:, 0] == '부타센세'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 부타센세')
        st.subheader('음식 종류: 일식')
        st.subheader('가격: 9000원')
        st.subheader('구글 별점: 4.54')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '중식':
        x = data[data.iloc[:, 0] == '얜시부'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 얜시부')
        st.subheader('음식 종류: 중식')
        st.subheader('가격: 8400원')
        st.subheader('구글 별점: 4.48')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '분식':
        x = data[data.iloc[:, 0] == '보영만두'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 보영만두')
        st.subheader('음식 종류: 분식')
        st.subheader('가격: 8000원')
        st.subheader('구글 별점: 4.28')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '카페':
        x = data[data.iloc[:, 0] == '라이킷'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 라이킷')
        st.subheader('음식 종류: 카페')
        st.subheader('가격: 4000원')
        st.subheader('구글 별점: 4.65')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '뷔페':
        pass
      elif food_kind == '기타':
        x = data[data.iloc[:, 0] == '도스마스'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 도스마스')
        st.subheader('음식 종류: 기타')
        st.subheader('가격: 5,000원')
        st.subheader('구글 별점: 4.28')
        st.subheader(f'식당 점수: {score}점')
        
    elif option == '중상':
      if food_kind == '한식':
        x = data[data.iloc[:, 0] == '장터밥상'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 장터밥상')
        st.subheader('음식 종류: 한식')
        st.subheader('가격: 7,000원')
        st.subheader('구글 별점: 4.38')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '양식':
        x = data[data.iloc[:, 0] == '차츰'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 차츰')
        st.subheader('음식 종류: 양식')
        st.subheader('가격: 19,000원')
        st.subheader('구글 별점: 4.75')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '아시아음식':
        x = data[data.iloc[:, 0] == '베트남쌀국수 포포 & 월남쌈'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 베트남쌀국수 포포 & 월남쌈')
        st.subheader('음식 종류: 아시아음식')
        st.subheader('가격: 11,000원')
        st.subheader('구글 별점: 4.46')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '일식':
        x = data[data.iloc[:, 0] == '삿뽀로'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 삿뽀로')
        st.subheader('음식 종류: 일식')
        st.subheader('가격: 47,000원')
        st.subheader('구글 별점: 4.32')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '중식':
        x = data[data.iloc[:, 0] == '짬뽕타임'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 짬뽕타임')
        st.subheader('음식 종류: 중식')
        st.subheader('가격: 9,000원')
        st.subheader('구글 별점: 4.29')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '분식':
        x = data[data.iloc[:, 0] == '보영만두'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 보영만두')
        st.subheader('음식 종류: 분식')
        st.subheader('가격: 8,000원')
        st.subheader('구글 별점: 4.28')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '카페':
        x = data[data.iloc[:, 0] == '콜링우드'].iloc[:, [2, 3, 4, 5, 6]]
        x = apply_log_to_columns(x)
        kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
        x[0] = kind[x[0]]
        tensor = torch.tensor(x).unsqueeze(0)
        tensor = model(tensor.to(DEVICE))
        score = Mapping(tensor).item() * 100
        st.subheader('식당 이름: 콜링우드')
        st.subheader('음식 종류: 카페')
        st.subheader('가격: 4,500원')
        st.subheader('구글 별점: 4.42')
        st.subheader(f'식당 점수: {score}점')
      elif food_kind == '뷔페':
        st.write('식당명')
        st.write('음식 종류')
        st.write('가격')
        st.write('네이버 별점')
        st.write('신뢰도')
      elif food_kind == '기타':
        pass