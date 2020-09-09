import numpy as np
import cupy as cp

class KNN:
    def __init__(self, K , X_train, y_train, y_name):
        self.K = K
        self.X = X_train
        self.y = y_train
        self.y_name = y_name
        self.idx = []
        self.distance_matrix = []
        
    def Calculate_distance(self, X1, X2):
        return ((X1 - X2)**2).sum(1)**0.5

    def get_nearest_k(self, X_test):
        # 거리계산된 배열, y와 1:1 매칭
        self.distance_matrix = self.Calculate_distance(self.X, X_test)
        # K번째로 작은 수
        KminusOne = (np.sort(self.distance_matrix)[self.K - 1])
        # 가장작은 k개의 인덱스 번호
        self.idx = np.where(self.distance_matrix <= KminusOne)
        
    def majority_vote(self):
        # label 배열로 변환
        label = np.bincount(self.y[self.idx], minlength = self.y_name.shape[0])
#         print(label)
        # 가장 많은 label 구하기
        Max = np.argmax(label)
        # 꽃 이름?
        return self.y_name[Max]
    
    def weighted_majority_vote(self):
        # 가중치 배열생성 1 / (distance_matrix + [1])
        arr = 1 / (self.distance_matrix[self.idx] + 1 )
        # arr에 각 idx에 매칭되는 y_train
        
#         yarr = self.y[self.idx]       
        yarr = cp.array(self.y)[self.idx].tolist()

        # 가중치들의 합을 계산
        result = np.zeros(self.y_name.shape[0])
        for i in range(self.K):
            result[yarr[i]] += arr[i]
        # 가중치 합이 가장 큰 값의 꽃 이름?    
#         print(result)
        return self.y_name[np.argmax(result)]
            
    def show_dim(self):
        print("Input Dimension: ",self.X.shape)
        print("Output Dimension: ",self.y.shape)
        
    def reset(self):
        self.idx = []
        self.distance_matrix = []