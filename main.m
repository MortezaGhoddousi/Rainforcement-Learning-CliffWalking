clc; clear; close all;

CW = CliffWalking_RL;
[statespace, cliff] = CW.createstatespace();
CW.observation_space = statespace;
CW.cliff = cliff;
CW.alpha = 0.1;
CW.gamma = 0.9;
pi = 0.9;

% Q = zeros(numel(statespace), 4);
load Q.mat

f = waitbar(0, 'Starting');
s = 100;
for iteration = 1:s
    Q = CW.runepisode(Q, pi);
    waitbar(iteration/s, f, sprintf('Progress: %d %%', floor(iteration/s*100)));
    pause(0.1);
end
close(f)

save Q.mat

chain = CW.rungreedy(Q, 0.1);
CW.result(chain);




