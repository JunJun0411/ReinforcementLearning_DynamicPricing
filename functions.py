# coding: utf-8

# Q-learning table사용
import scipy
import scipy.stats
import cupy as np
from cupy import asarray, concatenate, zeros, where, unique, asnumpy, bincount, argmax, argpartition, argsort, unravel_index, max as cmax, min as cmin, random as crandom
import numpy
from random import choice
import pylab
from collections import defaultdict
import pickle as pkl
from tqdm import tqdm_notebook

INITIALPRICE = 3800

def data_load():
    with open('./Data/dclus1.pkl', 'rb') as f:
        dclus = pkl.load(f)
        dclus = [asarray(i.astype('float').values) for i in dclus]
    
    with open('./Data/pp.pkl', 'rb') as f:
        pp = pkl.load(f)
        pp = [asarray(i.astype('float')) for i in pp]

    with open('./Data/dd.pkl', 'rb') as f:
        dd = pkl.load(f)
        dd = [asarray(i.astype('float')) for i in dd]
        
    return dclus, pp, dd
    
# 완전 랜덤 샘플링
def randomSampling(dc, num):
    # lon 최대 최소값, lat 최대 최소값
    max_lon, min_lon = max(dc[:, 0]), min(dc[:, 0])
    max_lat, min_lat = max(dc[:, 1]), min(dc[:, 1])
    
    # lon, lat값 랜덤 생성 후 매칭 반환
    sample_lon = np.random.uniform(low=min_lon, high=max_lon, size=(num,1))
    sample_lat = np.random.uniform(low=min_lat, high=max_lat, size=(num,1))
    return np.concatenate([sample_lon, sample_lat], axis=1)

def trip_Distance(idx, pp):
    """ OD """
    return ((cmax(pp[idx][:, 4]) + cmin(pp[idx][:, 4])) - pp[idx][:, 4]).reshape(-1, 1)

def Calculate_distance(X1, X2):
    """ euclidean 거리계산 -> harvasion으로 변경 고려 """
    return ((X1 - X2)**2).sum(1)**0.5

def Calculate_Matrix(idx, dclus, pp, dd):
    """공차 샘플링"""
    # 85% 하차데이터 + 15% 랜덤테이터
    x1 = int(dd[idx].shape[0] * 0.85)
    x2 = dd[idx].shape[0] - x1

    # 15% 랜덤데이터 생성
    rS = randomSampling(dclus[idx], x2)
    x3 = np.random.choice(dd[idx].shape[0], x1, replace=False)
    x3 = np.concatenate([dd[idx][x3][:, :2], rS], axis = 0)

    """ PD """
    x = pp[idx][:, 0:2]     # 승객 위치
    y = x3[:, 0:2]         # 공차 위치
    matrix = np.zeros([x.shape[0],y.shape[0]]) # 매트릭스 초기화


    # d1 계산
    for idx, i in enumerate(x):
        matrix[idx] = Calculate_distance(i, y)

    # emin = 0이 아닌 최솟값
#     emin = np.min(np.where(matrix == 0.0, 1, matrix))
#     return 1 / (matrix + emin) * emin
    return (cmax(matrix)+cmin(matrix)) - matrix

def myHdbscan(train_data_robustScaled, cluster_size = 7):
    """auto HBSCAN"""
    import hdbscan
    test_data = train_data_robustScaled
    clusterer = hdbscan.HDBSCAN(min_cluster_size = cluster_size, gen_min_span_tree=False, prediction_data =True)
    clusterer.fit(test_data)
    return clusterer, unique(clusterer.labels_).shape[0] # label갯수 리턴

def ScaledHdbscan(idx, pp, cluster_size = 7):
    from sklearn.preprocessing import RobustScaler
    
    #outlier 잡기 : Scaling
    robustScaler = RobustScaler()
#     train_data = pp[idx][:, :2]
    train_data = asnumpy(pp[idx][:, :2])
    print(robustScaler.fit(train_data))
    
    train_data_robustScaled = robustScaler.transform(train_data)
    
    """ HDBSCAN """
    clusterer, labelNum  = myHdbscan(train_data_robustScaled, cluster_size) # 4 ~ 8
    print (clusterer, labelNum)
    """ Hdbscan Hyperparameter autoScaling"""
    cnt = 0
    while labelNum > 15:
        cnt += 1
        if cnt > 10:
            break

        # cluster_size 높여서
        cluster_size += 1
        clusterer, labelNum = myHdbscan(train_data_robustScaled, cluster_size)
        
    if labelNum < 3:
        cluster_size += 1
        clusterer, labelNum = myHdbscan(train_data_robustScaled, cluster_size)
        cluster_size -= 1
    
    while labelNum < 3:
        cnt += 1
        if cnt > 4:
            print("labelNum < 3")
            break
        cluster_size -= 1
        clusterer, labelNum = myHdbscan(train_data_robustScaled, cluster_size)
        
    # Hdbscan 적용
    return clusterer, labelNum

def dbcluster(idx, pp, clusterer, labelNum, k=3):
    """ DS """
    from knn import KNN

    X_train = pp[idx][:, 0:2]
    
    t_train = clusterer.labels_
    # 레이블이 -1부터가 아닌 0부터 시작시키기 위해 + 1
    if -1 in clusterer.labels_:
        t_train = clusterer.labels_ + 1
        
    X_test = pp[idx][:, 2:4]
    print("label : ", unique(t_train))
    K = 3 # K = 3

    knn_train = KNN(K, X_train, t_train, unique(t_train))
    knn_train.show_dim()

    y2 = zeros(X_test.shape[0], dtype = int)

    for i in range(X_test.shape[0]):
        knn_train.get_nearest_k(X_test[i])
        y2[i] = knn_train.weighted_majority_vote()
        knn_train.reset()
        
    if -1 not in clusterer.labels_:
        y2 += 1
        
    binc = bincount(y2)
    maxLabel = argmax(binc[1:]) + 1
    maxLabelcount = cmax(binc[1:])
    print("y2 : ", y2, "\ny2.bincount : ", binc)
    print("maxLabel: ", maxLabel, "maxLabelcount : ", maxLabelcount)
    # 확률로 변환, maxLabel => 1.2, 0번 label => 0.8, 나머지 label => 1 

    y2 = where(y2 == maxLabel, 1.2, where(y2 == 0, 0.8, 1))

    # 수 확인
    values, counts = unique(y2, return_counts=True)
    print("values: ",values,"counts: ", counts)

    y2 = y2.reshape(-1, 1)
    return y2

def mat2(PP, DP, next_state, PD, matrix):
    """ 승객, 운전자 시각에서의 각각 Matrix를 중앙값으로 구분 후 조합 """
    MIN = min(matrix.shape[0], matrix.shape[1])

    # 승객입장 Matrix : 거리, 가격을 따짐
    xPP = PP - next_state
    P_Matrix = PD * xPP
    # 운전자입장 Matrix : 거리, OD, DS, 가격을 따짐
    xDP = next_state - DP
    D_Matrix = matrix * xDP

    # 두 Matrix 0 ~ 1 Scaling
    P_Matrix -= P_Matrix.min() # 0 ~
    P_Matrix = P_Matrix / P_Matrix.max() # 0 ~ 1
    D_Matrix -= D_Matrix.min() # 0 ~
    D_Matrix = D_Matrix / D_Matrix.max() # 0 ~ 1

    # 중앙값 : outlier의 영향이 적기 때문,
    Pmid = numpy.median(np.asnumpy(P_Matrix))
    Dmid = numpy.median(np.asnumpy(D_Matrix))

    # 중앙값 미만의 확률들은 0으로 초기화
    P_Matrix = np.where(P_Matrix < Pmid, 0, P_Matrix)
    D_Matrix = np.where(D_Matrix < Dmid, 0, D_Matrix)
    
    newMatrix = P_Matrix * D_Matrix
    
    return newMatrix, MIN

def pick(newMatrix, MIN):
    """ Matrix에서 모든 수가 0인 행, 열 뺐을 경우 min(행, 열) """
    
    newMatrix = np.where(newMatrix == 0., 0, 1)
    
    Pa = newMatrix.shape[0]
    Dr = newMatrix.shape[1]

    if newMatrix.sum(1).min() == 0:
        Pa = np.where(newMatrix.sum(1) == 0, 0, 1).sum()

    if newMatrix.sum(0).min() == 0:
        Dr = np.where(newMatrix.sum(0) == 0, 0, 1).sum()
    return MIN - min(Pa, Dr)

def pick2(mat, threshold = 0.4):
    """ Matrix에서 중복없이 뽑는 알고리즘 """
    n = mat.size
    flat = mat.flatten()
    indices = argpartition(flat, -n)[-n:]
    indices = indices[argsort(-flat[indices])]
    x = unravel_index(indices, mat.shape)
#     print(x, mat.shape[0], mat.shape[1], n)
    xx = x[0]
    yy = x[1]
    
    x1 = zeros(mat.shape[0], dtype=int)
    y1 = zeros(mat.shape[0], dtype=int)
    x2 = zeros(mat.shape[1], dtype=int)
    y2 = zeros(mat.shape[1], dtype=int)

    idx=0
    for i in range(n):
        if(not x1[xx[i]] and not y1[yy[i]]):
    #         print(i, ": ",xx[i], yy[i])
    #   기준 점 잡기
            x1[xx[i]] = 1
            y1[yy[i]] = 1
            x2[idx]=xx[i]
            y2[idx]=yy[i]
            idx+=1
            
    return idx

# # Q-learning table사용
# from cupy import asarray, concatenate, zeros, where, unique, asnumpy, bincount, argmax, \
# argpartition, argsort, unravel_index, max as cmax, min as cmin, random as crandom
# from numpy import median
# from random import choice
# from pylab import plot, savefig, title
# from collections import defaultdict
# from pickle import load
# from tqdm import tqdm_notebook
# from functions import data_load, randomSampling, trip_Distance, Calculate_distance,\
# Calculate_Matrix, myHdbscan, ScaledHdbscan, dbcluster, mat2, pick

INITIALPRICE = 3800

# 환경
class Env:
    def __init__(self):
        self.action_space = ['u', 'l', 'c', 'r', 'd'] # 행동 순서대로 -80, -20, 0, +20, +80
        self.n_actions = len(self.action_space)  # 5 actions
        self.marketPrice = INITIALPRICE          # marketPrice 
        self.PP = None    #Passenger의 선호 가격 리스트
        self.passengers = None #승객의 수 초기 500명
        self.DP = None    #Driver의 선호 가격 리스트
        self.drivers = None    #운전자 수 초기 500명
        self.matrix = None # matching matrix
        self.PD = None
        
    def step(self, action):
        if   action == 0:   # up : price-=80
            self.marketPrice-=80
            
        elif action == 1:   # left : price-=20
            self.marketPrice-=20
            
        elif action == 2:   # center : Nothing happen
            pass
        
        elif action == 3:   # right : price+=20
            self.marketPrice+=20

        elif action == 4:   # down : price+=80
            self.marketPrice+=80


        next_state = self.marketPrice
        reward = 0
        match = 0
        threshold = 0.4   # 기준점 어떻게?
        
        #보상 가격허용치를 넘으면 -100
        if next_state < (INITIALPRICE / 2) or next_state > (INITIALPRICE * 2):
            reward = -100
            
        else:
            """ 방법 2 Matrix 생성 후 뽑기"""
            newMatrix = mat2(self.PP, self.DP, next_state, self.PD, self.matrix)
            match = pick(newMatrix)
            reward = (match / 1000) * (next_state / 1000)

        return next_state, reward
        
    def reset(self, PD, matrix):
        self.passengers = matrix.shape[0]
        self.drivers = matrix.shape[1]
        self.PP = crandom.normal(self.marketPrice, self.marketPrice / 20, self.passengers).reshape(-1, 1) #Passenger의 선호 가격 리스트
        self.DP = crandom.normal(self.marketPrice, self.marketPrice / 20, self.drivers) #Driver의 선호 가격 리스트
        # 매칭 매트릭스 초기화
        self.PD = PD
        self.matrix = matrix
            
        return self.marketPrice
        
# 가격 Agent
class priceActionModel:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3, 4] 순서대로 -80, -20, 0, +20, +80
        self.load_model = False
        self.actions = actions
        self.learning_rate = None
        self.discount_factor = None
        self.epsilon = 1.  # exploration
        self.epsilon_decay = None
        self.epsilon_min = None
        self.q_table = None # 5열
        
    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)
    
    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if crandom.rand() < self.epsilon:
            
            action = choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            print(state, self.q_table[state])
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action
        
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return choice(max_index_list)
    
    def reset(self):
        self.load_model = False
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0]) # 5열
        
if __name__ == "__main__":
    # Data_load
    dclus, pp, dd = data_load()
    
    env = Env()
    agent = priceActionModel(actions=list(range(env.n_actions)))            # [0, 1, 2, 3, 4]
    
    for episode in tqdm_notebook(range(len(pp))):
        global_step = 0
        scores, episodes = [], []
    
        # 매칭 매트릭스 계산
        PD = Calculate_Matrix(episode, dclus, pp, dd)
        OD = trip_Distance(episode, pp)
        clusterer, labelNum = ScaledHdbscan(episode, pp, cluster_size = 7)
        DS = dbcluster(episode, pp, clusterer, labelNum, k = 3)
        matrix = PD * OD * DS
        print(episode, "시간, 승객 수: ", matrix.shape[0], ", 운전자 수: ",matrix.shape[1], "\n")
        state = env.reset(PD, matrix)
        agent.reset()
        for i in tqdm_notebook(range(50000)):
            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(str(state))
            
            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
            next_state, reward = env.step(action)

            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            # 모든 큐함수를 화면에 표시
            #env.print_value_all(agent.q_table)
            
            scores.append(state)
            episodes.append(episode*50000 + i)
            if((i % 100) == 0):
                plot(episodes, scores, 'b')
#                 if i > 49800:
                title('{}Time Price: {}'.format(episode, state))
                savefig("./DP_graph0{}.png".format(episode))
                print("episode:", episode, "  score:", state, " global_step:",
                      global_step, "  epsilon:", agent.epsilon)
            