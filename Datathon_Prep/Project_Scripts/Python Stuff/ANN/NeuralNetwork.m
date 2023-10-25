classdef NeuralNetwork
    properties
        tol
        input
        target
        approx
        layers
        weights
        error_history
    end
    
    methods
        function obj = NeuralNetwork(input, target, hidden)
            obj.tol = 0.01;
            obj.input = input;
            obj.target = target;
            obj.approx = zeros(size(target));
            obj.layers = {obj.input};
            obj.weights = {};
            layer_structure = [size(input, 2), hidden, 1];

            for i = 1:length(layer_structure) - 1
                obj.weights{i} = rand(layer_structure(i), layer_structure(i + 1));
            end
        end
        
        function obj = feedforward(obj)
            sigmoid = @(x) 1 ./ (1 + exp(-x));
            obj.layers = {obj.input};
            for i = 1:length(obj.weights) - 1
                obj.layers{end+1} = sigmoid(obj.layers{end} * obj.weights{i});
            end
            obj.layers{end+1} = sigmoid(obj.layers{end} * obj.weights{end});
            obj.approx = obj.layers{end};
        end
        
        function obj = backprop(obj)
            sigmoid_derivative = @(x) x .* (1 - x);
            error = 2 * (obj.target - obj.approx);
            for i = length(obj.weights):-1:1
                delta = error .* sigmoid_derivative(obj.layers{i + 1});
                error = delta * obj.weights{i}';
                obj.weights{i} = obj.weights{i} + obj.layers{i}' * delta;
            end
        end
        
        function obj = train(obj, epochs)
            if nargin < 2
                epochs = 10000;
            end
            for i = 1:epochs
                obj = obj.feedforward();
                obj = obj.backprop();
                er = mean((obj.target - obj.approx).^2);
                obj.error_history = [obj.error_history, er];
                if er < obj.tol
                    fprintf('Tolerance reached at iteration %d\n', i);
                    break;
                end
            end
        end
    end
end
