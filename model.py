import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class RestaurantRecommendationSystem:
    def __init__(self, data, random_seed=42):
        np.random.seed(random_seed)

        # 음식점 데이터 생성
        self.prices = data.iloc[:, 2] # 1~5까지 저가, 중저가, 중가, 중고가, 고가
        self.ratings = data.iloc[:, 3] # 네이버 별점
        self.visitors_reviews = data.iloc[:, 4] # 방문자 리뷰수
        self.blog_reviews = data.iloc[:5] # 블로그 리뷰수

        # 실제 점수
        self.weights = data.iloc[:, 6] # 주관적 별점을 가중치로 설정
        self.true_scores = (self.ratings + self.visitors_reviews + self.blog_reviews + self.prices) / 4
        self.true_scores_scaled = MinMaxScaler().fit_transform(self.true_scores.reshape(-1, 1)).flatten()

        # MLP 모델 생성 및 학습
        self.X = np.column_stack((self.ratings, self.visitors_reviews, self.blog_reviews, self.prices))
        self.y = self.true_scores_scaled
        self.mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_seed)
        self.mlp.fit(self.X, self.y)

    def predict_score(self, new_data):
        # 새로운 음식점의 점수에 대한 예측
        predicted_score = self.mlp.predict(new_data)
        logistic_score = 100 / (1 + np.exp(-predicted_score)) # 로지스틱 함수 통해서 0~1값으로 반환
        return logistic_score

    def recommend_restaurants(self, new_data, num_recommendations=3):
        # KNN을 사용하여 음식점 추천
        knn_weights = np.column_stack((self.weights * self.ratings, self.weights * self.visitors_reviews, self.weights * self.blog_reviews))
        knn_data = np.column_stack((self.X, knn_weights))
        knn_data_scaled = MinMaxScaler().fit_transform(knn_data)

        new_data_weighted = np.column_stack((new_data[:, :3], np.array(self.weights)))
        new_data_scaled = MinMaxScaler().fit_transform(new_data_weighted)

        knn = NearestNeighbors(n_neighbors = num_recommendations)
        knn.fit(knn_data_scaled)

        # 가중치를 적용하여 거리 계산
        distances, indices = knn.kneighbors(new_data_scaled)

        # 추천 음식점 인덱스 출력
        recommended_indices = indices[0]
        return recommended_indices
    
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class DataProcessor:
    def __init__(self, data):
        self.prices = data.iloc[:, 2]
        self.ratings = data.iloc[:, 3]
        self.visitors_reviews = data.iloc[:, 4]
        self.blog_reviews = data.iloc[:5]
        self.weights = data.iloc[:, 6]
        self.true_scores = (self.ratings + self.visitors_reviews + self.blog_reviews + self.prices) / 4
        self.true_scores_scaled = MinMaxScaler().fit_transform(self.true_scores.reshape(-1, 1)).flatten()

    def get_features_and_labels(self):
        X = np.column_stack((self.ratings, self.visitors_reviews, self.blog_reviews, self.prices))
        y = self.true_scores_scaled
        return X, y

class MLPModel:
    def __init__(self, random_seed=42):
        self.mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_seed)

    def train(self, X, y):
        self.mlp.fit(X, y)

    def predict_score(self, new_data):
        predicted_score = self.mlp.predict(new_data)
        logistic_score = 100 / (1 + np.exp(-predicted_score))
        return logistic_score

class RestaurantRecommender:
    def __init__(self, data_processor, mlp_model):
        self.data_processor = data_processor
        self.mlp_model = mlp_model

    def recommend_restaurants(self, new_data, num_recommendations=3):
        knn_weights = np.column_stack((self.data_processor.weights * self.data_processor.ratings,
                                       self.data_processor.weights * self.data_processor.visitors_reviews,
                                       self.data_processor.weights * self.data_processor.blog_reviews))
        knn_data = np.column_stack((self.data_processor.get_features_and_labels()[0], knn_weights))
        knn_data_scaled = MinMaxScaler().fit_transform(knn_data)

        new_data_weighted = np.column_stack((new_data[:, :3], np.array(self.data_processor.weights)))
        new_data_scaled = MinMaxScaler().fit_transform(new_data_weighted)

        knn = NearestNeighbors(n_neighbors=num_recommendations)
        knn.fit(knn_data_scaled)

        distances, indices = knn.kneighbors(new_data_scaled)
        recommended_indices = indices[0]
        return recommended_indices

# 데이터 로딩과 전처리
data = pd.read_csv("train.csv")
data_processor = DataProcessor(data)

# MLP 모델 학습과 예측
X, y = data_processor.get_features_and_labels()
mlp_model = MLPModel()
mlp_model.train(X, y)

# 음식점 추천
recommender = RestaurantRecommender(data_processor, mlp_model)
new_data = np.array([[4.5, 100, 20, 3]])  # 예시: 평점 4.5, 방문자 리뷰수 100, 블로그 리뷰수 20, 가격 중간가
recommendations = recommender.recommend_restaurants(new_data)
print("Recommended restaurant indices:", recommendations)
