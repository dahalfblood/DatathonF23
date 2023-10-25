rand('twister', 1973)   % equivalent to np.random.seed(1973)
input = [rand(5, 2); -rand(5, 2)];
target = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0];
hidden = [5,5,5,5,5,5,5,5,5,5,5,5,5];

nn = NeuralNetwork(input, target, hidden);
nn.tol = 0.01;
nn = nn.train();

figure('Position', [100, 100, 600, 600]);
plot(nn.error_history);
xlabel('Iteration');
ylabel('Error');

disp(nn.approx);
