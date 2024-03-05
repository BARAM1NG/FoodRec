import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import haversine
import random

# 위도 , 경도의 형태로 리스트를 받아옵니다
# 입력을 받으면 특정 거리(1km) 내에서 가장 점수가 높은 3개



# 두점사이의 거리 https://kayuse88.github.io/haversine/ haversine 이 경도, 위도값을 받아서 거리를 계산해주는 모듈입니다

# distance는 m단위 입니다



# 주소 통해 위도 경도 받아오기
import googlemaps
from datetime import datetime
my_key = "AIzaSyD-3sRu9IQcs67ci_9cCrXoSa5a1URELuE" #구글맵 API 키값
maps = googlemaps.Client(key=my_key)  # 구글맵 api 가져오기


def geocoding(address):
    try:
        geo_location = maps.geocode(address)[0].get('geometry')
        geo = [geo_location['location']['lat'], geo_location['location']['lng']]
        return geo
        
    # 좌표를 가져오지 못한 경우 에러 출력
    except:
        print('[0, 0]')



def yeongtong_data_making(csv_file):

    # csv_file을 넣었을때 [0: 음식점 이름 , 1: 위도 , 2: 경도] 로 numpy 배열을 반환하는 함수 /영통 음식점 리스트만 반환합니다

    yeongtong_name_data = np.array([])
    
    # 주소 데이터 str 변경
    for n in range(len(csv_file[:,2])):
        csv_file[n,2] = str(csv_file[n,2])

    for m in range(len(csv_file[:,2])):

        if  '영통' in csv_file[m,2]:
            
            # 음식점 이름 데이터
            yeongtong_name_data = np.append(yeongtong_name_data,csv_file[m,0])

    # numpy 변환
    data_array = np.array(csv_file)
    data_address = data_array[:, 2]

    a = []

    # data에 '영통' 단어가 포함되어 있는 경우, a에 영통 식당 주소 추가
    for address in data_address:

        if '영통' in address:
            b = geocoding(str(address))
            a.append(b)

    a = np.array(a)

    # ??
    yeongtong_coordinates = a.reshape(-1,2)
    yeongtong_name_data = yeongtong_name_data.reshape(-1,1)

    yeongtong_data = np.concatenate((yeongtong_name_data,yeongtong_coordinates),axis =1)

    return yeongtong_data


# yeongtong_data_making 함수를 쓸때 각 원소가 string으로만 반환되서 따로 float형으로 바꾸는 작업을 해야합니다 concatenate에서 발생하는거 같긴한데 잘모르겠음...


def distance_func(now_local_info,file_data):
    # now_local_info -> 현재 [경도,위도]를 받아옵니다 1x2 리스트 형태
    # all_local_info -> 영통 내 모든 [경도,위도] 데이터를 받아옵니다 nx2 리스트 형태 (영통내 데이터)
    #
    latitude= np.array([])
    longitude= np.array([])
    now_local_info = np.array(now_local_info)

    for i in range(len(file_data[:,0])):
        latitude = np.append(latitude,file_data[i,1])
        longitude = np.append(longitude,file_data[i,2])


    latitude = latitude.reshape(-1,1)
    longitude = longitude.reshape(-1, 1)


    latitude = latitude.astype(float)
    longitude = longitude.astype(float)
    now_local_info = now_local_info.astype(float)
    coordinates = np.concatenate((latitude, longitude), axis=1)

    distance = np.array([])
    # file_data는 0열에 이름,1열에 위도, 2열에 경도

    for m in range(len(latitude)):


        distance = np.append(distance,haversine.haversine(coordinates[m],now_local_info,unit='m'))
    #distance의 단위는 m입니다.
    distance = distance.reshape(-1,1)

    yeongtong_name = file_data[:,0].reshape(-1,1)
    local_info_dist = np.concatenate((yeongtong_name,distance),axis=1)
    # local_info_dist = np.concatenate(local_info_dist,target_score,axis=1)

    # score함수가 완성되면 위에 주석을 풀고 함수 인자로 score를 하나 더 받아와주세요


    return local_info_dist

# local_info_dist -> [0: 이름 1: 거리 ]
# input_data에는 이름, 위도, 경도 형태로 넘파이 배열이 들어간다 3x1

def search_func(input_data,file_data,target_score):

    #input_data = 현재 위치/고정된 위치의 [이름, 위도, 경도] 형태로 넘파이 배열이 들어갑니다 1x3
    #file_data = [이름 , 위도, 경도] 형태의 nx3 넘파이 배열이 들어갑니다 / 거리 구하는거는 search_func내부에서 돌리니까 데이터 그대로 넣어주시면 됩니다


    local_info_dist = distance_func(input_data[1:],file_data)
    coordinate = local_info_dist[:,1].astype(float)
    coordinate = coordinate.reshape(-1,1)
    inner_restaurant = np.array([])
    inner_restaurant_value = np.array([])
    inner_target_score = np.array([])
    ts = np.array([])

    for m in range(len(coordinate)):
        if coordinate[m] <= 1000:
            inner_restaurant = np.append(inner_restaurant,local_info_dist[m,0])
            inner_restaurant_value = np.append(inner_restaurant_value,local_info_dist[m,1])
            inner_target_score =np.append(inner_target_score,target_score[m])
    inner_restaurant = inner_restaurant.reshape(-1,1)
    inner_restaurant_value = inner_restaurant_value.reshape(-1,1)


    sorted_indices = np.argsort(inner_target_score)[::-1]
    sorted_inner_restaurant = inner_restaurant[sorted_indices]

    return sorted_inner_restaurant[:3]# 1km(1000m)이내의 음식점 중 가장 높은 스코어의 음식점을 3개 리스트를 가져옵니다.


# test code 입니다. encoding은 cp949로 안하면 한글 깨지는거 같습니다.

# ##test code1 : distance_func 체크 (거리 구하는 함수) / a는 yeongtong_data_making함수를 써서 뽑아낸 리스트중 일부입니다.
#
# a= [['초밥쟁이' ,'37.25243925' ,'127.07791934107411'],
#  ['만고쿠 영통점', '37.2531844', '127.07487949315782'],
#  ['꽃찬찜닭 영통점' ,'37.24849535', '127.07677232644994'],
#  ['DU COUPLE', '37.2735152', '127.0587676'],
#  ['동떡이아주대점' ,'37.2752617' ,'127.0446671'],
#  ['보약족발' ,'37.2525241','127.07780066486929'],
#  ['두가지떡볶이' ,'37.24886585' ,'127.07633665'],
#  ['전광수 커피하우스', '37.2495858' ,'127.07790039574452'],
#  ['쏘울피', '37.2479394', '127.07735835'],
#  ['키와마루아지', '37.2484284' ,'127.07598613557741'],
#  ['최고당돈가스' ,'37.2499581', '127.07613935'],
#  ['하루텐동' ,'0.0', '0.0'],
#  ['하추다방', '37.24951805' ,'127.07748082207937'],
#  ['메콩타이', '37.2512093', '127.07490995215784'],
#  ['청년다방' ,'37.252776499999996' ,'127.07340930661805'],
#  ['떡군이네' ,'37.25014095', '127.07436773949851'],
#  ['두가지떡볶이', '37.24886585', '127.07633665'],
#  ['엽기떡볶이', '0.0' ,'0.0'],
#  ['신전떡볶이' ,'37.2494851','127.0694717'],
#  ['이모네중앙닭발' ,'37.2520962' ,'127.07636506512978'],
#  ['찌개지존' ,'37.275621799999996' ,'127.04524320354511'],
#  ['심가네감자탕' ,'37.2481526' ,'127.0791858'],
#  ['서울24시감자탕해장국', '37.25364485', '127.076719'],
#  ['전주청기와감자탕' ,'37.25040465' ,'127.07366648415095'],
#  ['화성옥' ,'37.251262499999996', '127.0745033921558'],
#  ['청진옥' ,'37.2499873', '127.0794295'],
#  ['꽃찬찜닭', '37.24849535' ,'127.07677232644994'],
#  ['락빈닭칼국수한마리', '37.25073055' ,'127.07522175'],
#  ['새마을식당', '37.25191005', '127.0744791109828'],
#  ['정통집' ,'37.25248365', '127.0757189102378'],
#  ['천보헌' ,'37.24930485' ,'127.07426218142271'],
#  ['장터밥상' ,'37.2501822', '127.07663915'],
#  ['이모찌마' ,'37.2478181', '127.0769124'],
#  ['찜생찜사', '37.248454100000004' ,'127.07721687513495']]
#
# test_now = [37.24886585, 127.07633665] #현재위치 - 두가지떡볶이 지점입니다
#
#
#
# a = np.array(a)
# print(distance_func(test_now,a))
# print(distance_func(test_now,a).shape)
#
# # 구글맵에서 키와마루아지 - 두가지떡볶이 거리를 쟀을때 오차가 거의 없긴한데 조금더 확인해봐야할 거 같습니다
#
# # test code2: search_func함수 테스트입니다. score는 일단 임의로 지정했습니다.
#
# test_data = ['두가지떡볶이', 37.24886585, 127.07633665]
# test_score = []
# for i in range(34):
#     test_score = np.append(test_score,i)
# test_score = np.array(test_score)
# test_score = test_score.reshape(-1,1)
# test_score = np.sort(test_score)[::-1]
# print(test_score)
# test_final = search_func(test_data,a,test_score)
# print(test_final)



