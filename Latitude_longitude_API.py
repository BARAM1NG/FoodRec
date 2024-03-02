# 주소 받아오기
import googlemaps
from datetime import datetime

# Google Places API 키 설정
api_key = 'AIzaSyD-3sRu9IQcs67ci_9cCrXoSa5a1URELuE'

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

# 테스트용 코드
restaurant_address = get_restaurant_address(input())

# 참고 URL
# https://velog.io/@ejc9501/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%84%EB%8F%84%EA%B2%BD%EB%8F%84-%EC%B0%BE%EA%B8%B0geocoder-geocoding-API-%EA%B5%AC%EA%B8%80-%EC%8A%A4%ED%94%84%EB%A0%88%EB%93%9C%EC%8B%9C%ED%8A%B8

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

geocoding(restaurant_address)