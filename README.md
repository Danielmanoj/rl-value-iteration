# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## POLICY ITERATION ALGORITHM
The policy iteration algorithm is a method for finding the optimal policy in a Markov Decision Process (MDP). It alternates between policy evaluation (finding the value function for a fixed policy) and policy improvement (updating the policy using the new value function).

## VALUE ITERATION FUNCTION

The value iteration algorithm is another method for solving MDPs. It iteratively updates the value function using a Bellman optimality equation until the value function converges. Once the value function converges, the optimal policy can be derived by choosing the action that maximizes the expected value of the next state.
### Name: MANOJ G
### Register Number:212222240060
```
# Creating the Frozen Lake environment
envdesc  = ['SFFH','FHFF','FHFF', 'GFFF']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 12 
P = env.env.P
```
```
# Value Iteration Algorithm
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q = np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a] += prob*(reward + gamma * V[next_state] * (not done))
      if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
        break
      V = np.max(Q, axis=1)
    pi = lambda s : {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi
```

## OUTPUT:


![image](https://github.com/user-attachments/assets/5ac41044-9805-4021-9b33-f55f53569b97)

![image](https://github.com/user-attachments/assets/2e84a7c4-4f89-4963-b86b-23cfeca02c4a)

![image](https://github.com/user-attachments/assets/e4a7383b-e873-4ed0-bbc2-3589e6ed843b)



## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
