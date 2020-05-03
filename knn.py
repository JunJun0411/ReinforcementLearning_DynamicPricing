# coding: utf-8

import numpy as np

class KNN:
    def __init__(self, K , X_train, y_train, y_name):
        self.K = K
        self.X = X_train
        self.y = y_train
        self.y_name = y_name
        self.distance_matrix = []     # weighted_majority_vote에서 거리를 사용하기 위해 저장
        self.idx = []                 # 가장 작은 k개의 Index를 저장
        
    @staticmethod    
    # 두 배열(점)의 거리를 계산하는 함수
    def Calculate_distance(self, X1, X2):
        matrix = 0
        # 모든 feature에 대해서 차의 제곱 수행후 합한 뒤 제곱근을 씌운다.
        for i in range(X1.shape[1]):
            matrix += (X1[:,i] - X2[i])**2
        matrix = matrix**0.5
        return matrix
    
    # test data와 가까운 K개의 Index를 뽑아 저장
    def get_nearest_k(self, X_test):
        # train_data와 test_data의 거리계산된 배열을 self.distance_matrix에 저장
        self.distance_matrix = self.Calculate_distance(self.X, X_test)
        
        # 오름차순으로 정렬 후 K번째로 작은 수 뽑는다
        KminusOne = (np.sort(self.distance_matrix)[self.K - 1])
        
        # 가장작은 k개의 인덱스 번호를 뽑아 self.idx에 저장
        self.idx = np.where(self.distance_matrix <= KminusOne)
        
    # 가까운 k개의 속한 그룹을 보고 가장 많이 속해있는 그룹의 꽃이름을 return    
    def majority_vote(self):
        # train_y data 에 인덱싱 되는 개수를 각각 구한다.
        label = np.bincount(self.y[self.idx], minlength = self.y_name.shape[0])

        # 가장 많이 indexing되어있는 output 구하기
        Max = np.argmax(label)

        # 꽃 이름?
        return self.y_name[Max]

    # 가까운 k개의 가중치(1/1+distance)를 각각 구해 그 합이 가장 큰 그룹의 꽃이름을 return
    def weighted_majority_vote(self):
        # 가중치 배열생성 1 / (distance_matrix + [1]) 
        arr = 1 / (self.distance_matrix[self.idx] + [1] )

        # arr에 각 idx에 매칭되는 y_train 
        yarr = self.y[self.idx]

        # 가중치들의 합을 계산
        result = np.zeros(self.y_name.shape[0])
        for i in range(self.K):
            result[yarr[i]] += arr[i]

        # 가중치 합이 가장 큰 값의 꽃 이름?    
        return self.y_name[np.argmax(result)]
    
    # input, output 차원을 보여주는 함수        
    def show_dim(self):
        print("Input Dimension: ",self.X.shape)
        print("Output Dimension: ",self.y.shape)
    # test data가 새로 들어와 KNN알고리즘을 정상수행 할 수 있도록 초기화
    def reset(self):
        self.idx = []
        self.distance_matrix = []