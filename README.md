# Rainforcement-Learning-CliffWalking

Example 6.6: Cliff Walking This gridworld example compares Sarsa and Q-learning, highlighting the dierence between on-policy (Sarsa) and off-policy (Q-learning) methods. 
Consider the gridworld shown in the upper part of Figure 6.13. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left.
Reward is -1 on all transitions except those into the the region marked "The Cliff." Stepping into this region incurs a reward of 􀀀100 and sends the agent instantly back to the start. 
The lower part of the gure shows the performance of the Sarsa and Q-learning methods with e-greedy action selection, e = 0.1.
After an initial transient, Q-learning learns values for the optimal policy, that which travels right along the edge of the cliff. Unfortunately, this results in its occasionally falling off the cliff because of the e-greedy action selection.
Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path through the upper part of the grid. Although Qlearning actually learns the values of the optimal policy, its on-line performance is worse than that of Sarsa, which learns the roundabout policy. 
Of course, if epsilon were gradually reduced, then both methods would asymptotically converge to the optimal policy.

![Untitled](https://github.com/MortezaGhoddousi/Rainforcement-Learning-CliffWalking/assets/143504966/2b05b5d2-38b4-4021-afb1-559a9d5de679)
