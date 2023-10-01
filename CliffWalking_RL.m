classdef CliffWalking_RL
    properties
        observation_space {mustBeNumeric}
        cliff {mustBeNumeric}
        alpha {mustBeNumeric}
        gamma {mustBeNumeric}
    end
    methods
        function r = reset(~) % Reset Episode
            r = 37;
        end
        function action = softmax(~, Q, state, pi)
            p = exp(Q(state, :)./pi)./sum(exp(Q(state, :)./(pi)));
            p = cumsum(p);
            p(end) = 1;
            action = find(rand<=p, 1, 'first');
        end
        function [statespace, cliff] = createstatespace(~)
            cliff = zeros(4, 12);
            cliff(4, 2:11) = 1;
            cliff(end) = 2;
            statespace = 1:numel(cliff);
            statespace = reshape(statespace, size(cliff'))';
        end
        function [i, j] = findstate(obj, state)
            [i, j] = find(obj.observation_space==state);
        end
        function [new_state, reward, Terminal] = step(obj, action, state)
            [i, j] = obj.findstate(state);
            switch action
                case 1
                    i = i-1; % Up
                case 2
                    j = j+1; % Right
                case 3
                    i = i+1; % Down
                case 4
                    j = j-1; % Left
            end
            i = min(i, size(obj.observation_space, 1));
            i = max(i, 1);
            j = min(j, size(obj.observation_space, 2));
            j = max(j, 1);
            if obj.cliff(i, j) == 1
                new_state = obj.reset();
                reward = -100;
                Terminal = false;
            elseif obj.cliff(i, j) == 0
                new_state = obj.observation_space(i, j);
                reward = -1;
                Terminal = false;
            elseif obj.cliff(i, j) == 2
                new_state = obj.observation_space(i, j);
                reward = 0;
                Terminal = true;
            end
        end
        function Q = runepisode(obj, Q, pi)
            state = obj.reset();
            nappar = true;
            while nappar
                action = obj.softmax(Q, state, pi);
                [new_state, reward, Terminal] = obj.step(action, state);
                Q(state, action) = Q(state, action) + ...
                    obj.alpha*(reward + ...
                    obj.gamma*max(Q(new_state, action)) - ...
                    Q(state, action));
                state = new_state;
                if Terminal
                    break
                end
                pi = pi*0.9;
            end
        end
        function chain = rungreedy(obj, Q, pi)
            state = obj.reset();
            nappar = true;
            iter = 1;
            while nappar
                action = obj.softmax(Q, state, pi);
                [new_state, ~, Terminal] = obj.step(action, state);
                chain(iter) = state; %#ok
                iter = iter+1;
                state = new_state;
                if Terminal
                    chain(iter) = obj.observation_space(end); %#ok
                    break
                end
            end
        end
        function [] = result(obj, chain)
            I = uint8(obj.cliff)*255;
            I(end) = 0;
            for it = 1:numel(chain)
                c = 125;
                [i, j] = obj.findstate(chain(it));
                I(i, j) = c;
            end
            I = imresize(I, 5);
            figure
            imshow (I)
            title('Cliff Walking')
            xlabel('Gray squeres represent the direction of movement')
        end
    end
end