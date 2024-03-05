from re import A
import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

data_csv = 'data.csv'
data_use = pd.read_csv(data_csv, encoding='CP949')
data_array = np.array(data_use)
data_address = data_array[:,2]

def geocoding(data_address):
  geo_local = Nominatim(user_agent='South Korea', timeout=30)

  ## (위도, 경도)로 바꿔서 추가하기
  coordinate = []
  for address in data_address:
    try:
        location = geo_local.geocode(address)
        geo = [location.latitude, location.longitude]
        if location:
          coordinate.append(geo)
        else:
          coordinate.append([0,0])
    except:
      coordinate.append([0,0])

  ## (이름, 위도, 경도)로 출력하기
  info=[]
  for k in range(len(coordinate)):
     m = data_array[k,0]
     n = data_array[k,1]
     q= []
     q.append(m)
     q.append(coordinate[k][0])
     q.append(coordinate[k][1])
     q.append(n)
     info.insert(k, q)
  return info

# latitude/longitude = 현재 사용자의 위도/경도
# limit =  m(미터) 기준, 사용자 지정 거리 범위
# info=geocoding(data_address) ([식당명, 위도, 경도] 리스트)
# menu= 음식점분류 (ex. 한식)

def possiblelist(latitude, longitude, limit, info, menu):

  import math
  R = 6378.135     # km기준, 지구의 반지름

  # 위도 1도 = 지구 반지름 * 1도 * 1rad
  latitude_1 = R*1*math.radians(1)
  # 1m당 위도 이동값
  move_latitude = (1/latitude_1)/1000

  # 경도 1도 = 지구 반지름 * 1도 * cos(위도) * 1rad
  longitude_1 = R*1*math.cos(latitude)*math.radians(1)
  # 1m당 경도 이동값
  move_longitude = (1/longitude_1)/1000


  # 먼저 원하는 limit을 위도/경도로 바꿔주고, 해당 구역 내에 있는 식당 리스트를 뽑는다.
  # ex. 500m의 위도/경도 이동값이 0.1/0.2이고 현재 위치가 (3,5)라고 하면
  # x좌표가 2.9-3.1 사이에 있고 y좌표가 4.8-5.2 사이에 있는 식당의 리스트를 만든다.
  # 거리 계산해야하는데 모든 값에 대해 계산하면 오래걸리니까 일단 이렇게 거르기.
  expected_list = []
  for v in range(len(info)):
     if (latitude-limit*move_latitude)<= info[v][1] <= (latitude+limit*move_latitude) and (longitude-limit*move_longitude)<= info[v][2] <= (longitude+limit*longitude):
       expected_list.append(info[v])
     else:
      pass

  if not expected_list:
    return '조건에 맞는 식당이 없습니다. 거리 제한을 늘려주세요.'
  else:
    pass


  ## 이제 이 리스트를 바탕으로 정확한 거리를 계산해서 기준에 포함되는 좌표만 모은다.
  ## 아까껀 정사각형 범위로 한거고, 이제 여기서 하나씩 계산해서 원래 하려고 했던, 진짜 조건 만족하는 원형 범위 내의 식당 고를거임
  real_expected_list = []
  for w in range(len(expected_list)):
    d_latitude = math.radians(abs(latitude-expected_list[w][1]))
    d_longitude = math.radians(abs(longitude-expected_list[w][2]))
    a = math.sin(d_latitude / 2) * math.sin(d_latitude / 2) + math.cos(math.radians(latitude)) * math.cos(math.radians(expected_list[w][1])) * math.sin(d_longitude / 2) * math.sin(d_longitude / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000
    if distance <= limit:
      real_expected_list.append(expected_list[w])
    else:
      pass
  if not real_expected_list:
    return '조건에 맞는 식당이 없습니다. 거리 제한을 늘려주세요.'
  else:
    pass


  ## 이제 줄세우기해서 정하자. 만약 real_expected_list가 [이름, 위도, 경도, 분류, 평점이라면]
  suggest_list = sorted(real_expected_list, key=lambda x: x[4], reverse=True)     ## 별점이 높은 순서대로 나열
  real_suggest_list = []
  for r in range(len(suggest_list)):
    if suggest_list[r][3]==menu:                              ## ex. menu=한식
      real_suggest_list(suggest_list[r][0])                   ## 한식 음식점의 이름만 별점 순서대로 저장됨
    else:
      pass

  if not real_suggest_list:
    return ('주어진 조건에서 menu 식당이 없습니다. 메뉴 종류와 상관없이 추천 식당을 알려드립니다.', suggest_list[:3])
  else:
    return real_suggest_list[:3]                     ## 높은 순서로 3개 출력




