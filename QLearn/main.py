import numpy as np

# Reward matrix
R = np.matrix([ [-1,-1,-1,-1,0,-1], 
        [-1,-1,-1,0,-1,100],
        [-1,-1,-1,0,-1,-1],
        [-1,0,0,-1,0,-1],
        [-1,0,0,-1,-1,100],
        [-1,0,-1,-1,0,100] ])

# Quality matrix
Q = np.matrix(np.zeros([6,6]))

gamma = 0.8

# ---------------------------------------------------------------------------------

def get_next_action(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    next_action = int(np.random.choice(av_act,1))
    return next_action


def update(current_state, action, gamma):
    
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * max_value

# --------------------------------------------------------------------------------

for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    action = get_next_action(current_state)
    update(current_state,action,gamma)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

# --------------------------------------------------------------------------------

current_state = 2 #
steps = [current_state]

while current_state != 5:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1] #büyük olanı seçiyor
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1)) #en büyük değerden iki tane varsa rasgele seciyor
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index) #seçilen state i list e yazıyr
    current_state = next_step_index 

# Print selected sequence of steps
print("Selected path:")
print(steps)
