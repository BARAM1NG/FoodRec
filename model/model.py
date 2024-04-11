import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #torch.nn.functional.logsigmoid 함수 사용
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math

BATCH_SIZE = 10 #실험을 통해 수정

kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}

# 전처리 클래스
#가게이름 | 음식종류 | 가격 | 구글 지도 별점 | 방문자 리뷰 | 블로그 리뷰 | 주소 | 주관적 별점
class CustomDataset(Dataset):
  def __init__(self, dataframe):
    self.data = dataframe

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    _names = self.data.iloc[idx, 0]
    _address = self.data.iloc[idx, 1]
    _kinds = [kind[self.data.iloc[idx, 2]] for i in range(9)]
    _x = self.data.iloc[idx, 3:7].astype(np.float32)
    _target_score = self.data.iloc[idx, 7].astype(np.float32)

    return torch.tensor(_kinds), torch.tensor(_x), torch.tensor(_target_score)

# 매핑 함수
def Mapping(score):
  scaled_score= F.softmax(score)

  return scaled_score[:, 0].unsqueeze(1)

# MLP 클래스
class MLP(nn.Module):
  def __init__(self, node_number, device):
    super(MLP, self).__init__()
    self.MLP = nn.Sequential(
        nn.Linear(5, node_number),
        nn.Linear(node_number, node_number*2),
        nn.Linear(node_number*2, 2)
    )
    #음식 종류 기준 : 네이버
    #한식, 양식, 아시아음식, 일식, 중식, 분식, 카페, 뷔페, 기타
    self.weight_0 = nn.Parameter(torch.full((9,), 0.5, dtype=torch.float32))

  def forward(self, x):
    score = self.MLP(x)

    return score

# Kind 인코딩 함수  
def encode_category(df):
    category_mapping = {
        '한식': 0,
        '양식': 1,
        '아시아음식': 2,
        '일식': 3,
        '중식': 4,
        '분식': 5,
        '카페': 6,
        '뷔페': 7,
        '기타': 8
    }
    df[0] = df[0].map(category_mapping)
    return df


# 인코딩 함수  
def apply_log_to_columns(df):
    kind = {'한식':0, '양식':1, '아시아음식':2, '일식':3, '중식':4, '분식':5, '카페':6, '뷔페':7, '기타':8}
    columns_to_apply_log = [1, 3, 4]  # 원하는 열의 인덱스
    for col in columns_to_apply_log:
        df.iloc[:, col] = df.iloc[:, col].apply(lambda x: math.log(x))
    
    df = df.values.tolist()

    df = [item for sublist in df for item in sublist]
    return df