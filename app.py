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

from Latitude_longitude_API import get_restaurant_address, geocoding



# warnings ignore
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Exception ignored in")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'

# 꽉찬 화면 구성
st.set_page_config(layout="wide")

# 페이지 구상
st.title('	:school: Kyung Hee University FoodRec Service')
st.write('---')

# 가고 싶은 식당 입력
title = st.text_input(
  label = '가고 싶은 식당이 있나요?',
  placeholder = '식당 이름을 입력해주세요'
)

# 검색 input
title


# 화면 2분할
col2, col3 = st.columns(2)

# 식당 신뢰도 출력
with col2:
  st.write('식당 신뢰도 출력 칸')

  # 해당 식당 지도 화면 불러오기
with col3:

  # 지도 API 불러오기
  restaurant_address = get_restaurant_address(title)
  restaurant_location = geocoding(restaurant_address)
  map = folium.Map(location=restaurant_location, zoom_start=13, control_scale=True)

  # 지도 마커 표시
  marker = folium.Marker(restaurant_location, popup = title, icon = folium.Icon(color = 'blue'))
  marker.add_to(map)
  map
  
# 추천 식당 tab
tab1, tab2, tab3 = st.tabs(['추천 맛집 1', '추천 맛집 2', '추천 맛집 3'])


# 추천 맛집 
with tab1:
  st.header('추천 식당 1')
  col4, col5 = st.columns(2)
  
  with col4:
    map = folium.Map(location=[37.53897093698831, 127.05461953077439], zoom_start=13, control_scale=True)
    
    map
  
  with col5:
    st.write('식당명')
    st.write('음식 종류')
    st.write('가격')
    st.write('네이버 별점')
    
with tab2:
  st.header('추천 식당 2')

  col6, col7 = st.columns(2)
  
  with col6:
    map = folium.Map(location=[37.53897093698831, 127.05461953077439], zoom_start=13, control_scale=True)
    
    map
  
  with col7:
    st.write('식당명')
    st.write('음식 종류')
    st.write('가격')
    st.write('네이버 별점')

with tab3:
  st.header('추천 식당 3')
  col8, col9 = st.columns(2)
  
  with col8:
    map = folium.Map(location=[37.53897093698831, 127.05461953077439], zoom_start=13, control_scale=True)
    
    map
  
  with col9:
    st.write('식당명')
    st.write('음식 종류')
    st.write('가격')
    st.write('네이버 별점')