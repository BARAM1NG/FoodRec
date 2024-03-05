# 주소 받아오기
import googlemaps
from datetime import datetime

# Google Places API 키 설정
api_key = id_key

# Google Maps 클라이언트 생성
gmaps = googlemaps.Client(key=api_key)

def get_restaurant_address(restaurant_name):
    # 장소 검색 API 호출
    places_result = gmaps.places(query=restaurant_name, language='ko')

    # 결과에서 첫 번째 장소 선택
    place = places_result['results'][0]

    # 주소 추출
    address = place['formatted_address']

    return address


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