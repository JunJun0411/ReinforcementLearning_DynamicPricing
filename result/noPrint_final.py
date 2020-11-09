# coding: utf-8

# Q-learning table사용
from cupy import asarray, concatenate, zeros, where, unique, asnumpy, bincount, argmax, argpartition, argsort, unravel_index, max as cmax, min as cmin, random as crandom
from numpy import median
from random import choice
from pylab import plot, savefig, title, clf
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from functions import data_load, randomSampling, trip_Distance, Calculate_distance,Calculate_Matrix, myHdbscan, ScaledHdbscan, dbcluster, mat2, pick

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
        
        # S' < 시장가격 절반가격 이하로 내려갈 경우
        if next_state < (INITIALPRICE / 2):
            reward = next_state - (INITIALPRICE * 3) / 2
            
         # S' < 시장가격 2배 가격 이상으로 올라갈 경우
        elif next_state > (INITIALPRICE * 2):
            reward = INITIALPRICE - next_state

        else:
            """ 방법 2 Matrix 생성 후 뽑기"""
            newMatrix, MIN = mat2(self.PP, self.DP, next_state, self.PD, self.matrix)
            match = pick(newMatrix, MIN)
            reward = (match) * (next_state / 1000)

        return next_state, reward
        
    def reset(self, PD, matrix):
        self.passengers = matrix.shape[0]
        self.drivers = matrix.shape[1]
        self.PP = crandom.normal(self.marketPrice, self.marketPrice / 20, self.passengers).reshape(-1, 1) #Passenger의 선호 가격 리스트
        self.DP = crandom.normal(self.marketPrice, self.marketPrice / 20, self.drivers) #Driver의 선호 가격 리스트
        # 매칭 매트릭스 초기화
        self.PD = PD
        self.matrix = matrix
        self.marketPrice = INITIALPRICE

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
            #print(state, self.q_table[state])
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
    
    for episode in tqdm(range(len(pp))):
        global_step = 0
        scores, episodes = [], []
    
        # 매칭 매트릭스 계산
        PD = Calculate_Matrix(episode, dclus, pp, dd)
        OD = trip_Distance(episode, pp)
        clusterer, labelNum = ScaledHdbscan(episode, pp, cluster_size = 7)
        DS = dbcluster(episode, pp, clusterer, labelNum, k = 3)
        matrix = PD * OD * DS

        Passenger = matrix.shape[0]
        Driver = matrix.shape[1]
#        print(episode, "시간, 승객 수: ", matrix.shape[0], ", 운전자 수: ",matrix.shape[1], "\n")

        state = env.reset(PD, matrix)
        agent.reset()

        for i in tqdm(range(50000)):
            #learningRate 조정
            agent.learning_rate = 500 / (i + 501)

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
            episodes.append(i)

        plot(episodes, scores, 'b')
        title('{}Time Price: {}, Passenger: {}, Driver: {}'.format(episode, state, Passenger, Driver))
        savefig("./Grape/DP_graph_{}Time.png".format(episode))
#               global_step += 1
#               print("episode:", episode, "  score:", state, " global_step:",
#                      global_step, "  epsilon:", agent.epsilon)
        clf() 

