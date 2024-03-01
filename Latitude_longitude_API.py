# 참고 URL
# https://velog.io/@ejc9501/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9C%84%EB%8F%84%EA%B2%BD%EB%8F%84-%EC%B0%BE%EA%B8%B0geocoder-geocoding-API-%EA%B5%AC%EA%B8%80-%EC%8A%A4%ED%94%84%EB%A0%88%EB%93%9C%EC%8B%9C%ED%8A%B8

# import 라이브러리
from geopy.geocoders import Nominatim

# 위도, 경도 반환하는 함수
def geocoding(address):
    try:
        geo_local = Nominatim(user_agent='South Korea')  #지역설정
        location = geo_local.geocode(address)
        geo = [location.latitude, location.longitude]
        return geo

    except:
        return [0,0]

geocoding('경기 하남시 미사강변동로 121')[0]