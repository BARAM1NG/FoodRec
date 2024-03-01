import googlemaps
from datetime import datetime

# Google Places API 키 설정
api_key = 'AIzaSyD-3sRu9IQcs67ci_9cCrXoSa5a1URELuE'

# Google Maps 클라이언트 생성
gmaps = googlemaps.Client(key=api_key)

def get_restaurant_address(restaurant_name):
    # 장소 검색 API 호출
    places_result = gmaps.places(query=restaurant_name)

    # 결과에서 첫 번째 장소 선택
    place = places_result['results'][0]

    # 주소 추출
    address = place['formatted_address']

    return address

# 테스트용 코드
restaurant_name = "츠쿠모"
restaurant_address = get_restaurant_address(restaurant_name)
restaurant_address.split(',')