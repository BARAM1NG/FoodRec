# 참고 URL
# https://velog.io/@ejc9501/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%84%EB%8F%84%EA%B2%BD%EB%8F%84-%EC%B0%BE%EA%B8%B0geocoder-geocoding-API-%EA%B5%AC%EA%B8%80-%EC%8A%A4%ED%94%84%EB%A0%88%EB%93%9C%EC%8B%9C%ED%8A%B8

#구글맵 api 로드
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

geocoding('대한민국 서울특별시 송파구 방이동 105-4번지 1층')