function nn()
    %set non-random seed
    rng(1);
    global ANNType eta
    ANNType = 'sigmoid'; % 'ReLU'; % 'tanh'; % 'SiLU'; % 'softplus'; % 'Gaussian'; % 'linear'; % 'sin'; % 'quadratic'; % 
    fTol = 0.01; % Tolerance
    eta = 0.1;
    % set neurons per layer, for example [20, 20, 20] means three layers of
    % 20 neurons each
    nHidLayer =  [5 7 5 3]; 
%% JBG ANN implementation
    nExample = 0.1;
    switch nExample 
        case 0
            mTrainInput = [1; 2; -1; -2]; 
            %mTrainInput = (mTrainInput - mean(mTrainInput) ) ./ std(mTrainInput);
            mTrainOut = [1; 1; 0; 0];
            mTestInput = [3; 4; -3; -4];
            %mTestInput = (mTestInput / max(mTestInput) - 1)  * 3 ;
            mTestOut = [1; 1; 0; 0];
        case 0.1
            mTrainInput = [1, 1; 2, 1; -1, -1; -2, -1]; 
            %mTrainInput = (mTrainInput / max(mTrainInput) - 1) * 3  ;
            mTrainOut = [1; 1; 0; 0];
            mTestInput = [3, 1; 4, 1; -3, -1; -4, -1];
            %mTestInput = (mTestInput / max(mTestInput) - 1)  * 3 ;
            mTestOut = [1; 1; 0; 0];
        case 0.2
            mTrainInput = [1, 1, 1; 2, 1, 1; -1, 1, 1; -2, 1, 1]; 
            %mTrainInput = (mTrainInput / max(mTrainInput) - 1) * 3  ;
            mTrainOut = [1; 1; 0; 0];
            mTestInput = [3, 1, 1; 4, 1, 1; -3, 1, 1; -4, 1, 1];
            %mTestInput = (mTestInput / max(mTestInput) - 1)  * 3 ;
            mTestOut = [1; 1; 0; 0];
        case 1
            mTrainInput = [1 3; -1 -5]; 
            mTrainOut = [1; 0];
            mTestInput = [2 4; -1 -3];
            mTestOut = [1; 0];
        case 2
            m=5; n = 10; 
            % Normal random numbers:  normrnd(mu, sigma,...)
            mTrainInput = [normrnd(0,1,[m,n]); normrnd(10,1,[m,n])]; 
            mTrainOut = [ones(m,1); zeros(m,1)];
            mTestInput = [normrnd(0,1,[m,n]); normrnd(10,1,[m,n])];
            mTestOut = [ones(m,1); zeros(m,1)];
        case 3
            [mTrainInput, mTrainOut, mTestInput, mTestOut] = natalityData();
    end
    % Configure ANN
    nInputSize = size(mTrainInput,2);
    nOutSize = size(mTrainOut,2);
    mArchitecture = [nInputSize,...
                    nHidLayer,...
                    nOutSize];
    cSyn = funSynapseInitialization(mArchitecture);
    funTestAccuracy(cSyn,mTrainInput,mTrainOut,mTestInput,mTestOut)
         
    er=[]
    for i=[1:100000]
        %disp('----------------------')
        layer = funFeedForward(cSyn,mTrainInput);
        [o, fErr] = funBackPropagation(cSyn,layer,mTrainOut);
        cSyn = o;
        er = [er; fErr];
        if fErr<fTol
            fprintf("Stopping at error: %f error\n", fErr)
            break
        end  
        %print out debug data
        if i==1 || mod(i,100000) == 0
            fprintf("\titer=%.0f, Error: %f\n", i, fErr)
            disp(layer{end})
        end 
    end
    plot(er)
    funTestAccuracy(cSyn,mTrainInput,...
        mTrainOut,mTestInput,mTestOut)

    if fErr>fTol
        fprintf("Value Below Tolerance not found, please adjust alpha\n\n")
    else
        fprintf("Stopped at iteration: %f \n", i)
        fprintf("Value Below Tolerance found: %f\n\n", fErr)
    end
%% MATLAB ANN toolbox
    %singleNeuron;
    %customNeuron;
    %linearPerceptronSeparator()
    %fourClassPerceptronClassification()
end

function cSyn = funSynapseInitialization(mArchitecture)
    cSyn = {};
    for i = 1:length(mArchitecture)-1
        %initialize random synapse weights with a mean of 0 in [-1,1]
        cSyn{i} = 2*rand(mArchitecture(i),mArchitecture(i+1)) - 1;
    end
end

function layer = funFeedForward(cSyn,mTrainInput)
    layer = {mTrainInput};
    %disp('Feed forward layer')
    for i = 1:length(cSyn)
        layer{i+1} = funActivation(layer{i}, cSyn{i}); 
    end    
end

function [cSyn, fErr] = funBackPropagation(cSyn,layer,mTrainOut)
    global eta 
    layer_error = {};
    layer_delta = {};
    for i = length(layer):-1:1
        if i== length(layer)
            layer_error{i} = layer{end} - mTrainOut;
        else
            layer_error{i} = layer_delta{i+1}*cSyn{i}.';
        end
        layer_delta{i} = layer_error{i}.*funActivationDerivative(layer{i});
    end
    %adjust values via gradient descent
    for i = length(cSyn):-1:1
        cSyn{i} = cSyn{i} - eta.*(layer{i}.'*layer_delta{i+1});
    end
    %Record error
    fErr = mean(abs(layer_error{end}));
end

function funTestAccuracy(cSyn,mTrainInput,...
        mTrainOut,mTestInput,mTestOut)
    %check for accuracy w.r.t. training data
    layer = funFeedForward(cSyn,mTrainInput);
    disp("Test  Output                          :"); 
    disp(layer{end});
    err = immse(layer{end}, mTrainOut);
    fprintf("Mean Squared Error with Training data : %f\n", err)
    %check for accuracy w.r.t. training data
    layer = funFeedForward(cSyn,mTestInput);
    err = immse(layer{end}, mTestOut);
    fprintf("Mean Squared Error with Testing data  : %f\n", err)
end



function out = funActivation(layer, synapse)
    global ANNType 
    x = layer*synapse;
    switch ANNType
        case 'linear'
            out = x;
        case 'quadratic'
            out = x.^2;
        case'sin'
            out = sin(x); 
        case 'sigmoid'
            out = 1 ./ (1+exp(-x)); 
        case'tanh'
            out = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x)); 
        case 'SiLU'
            out = x./(1 + exp(-x)); 
        case 'Gaussian'
            out = exp(-x.^2);
        case 'softplus'
            out = log(1+exp(x));
        case 'ReLU'
            out = relu(x);
    end    
end

function g = relu(z)
    g = max(0,z);
end

function out = funActivationDerivative(x)
    global ANNType 
    switch ANNType
        case 'linear'
            out = 1;
        case 'quadratic'
            out = 2*x;
        case 'sin'
            out = cos(x); 
        case 'sigmoid'
            out = (exp(-x) ./ (exp(-x) + 1).^2); 
        case'tanh'
            out = 1- ((exp(x) - exp(-x)) ./ (exp(x) + exp(-x))) .^2; 
        case 'SiLU'
            out = (1 + exp(-x) + x.*exp(-x)) ./ (1 + exp(-x)).^2; 
        case 'Gaussian'
            out = -2.*x.*exp(-x.^2);
        case 'softplus'
            out = 1./(1+exp(-x));
        case 'ReLU'
            out = reluGradient(x);
    end    
end

function z = reluGradient(z)
    z(z>=0) = 1;
    z(z<0) = 0;
end
%% Natality data
function [mTrainInput, mTrainOut, ...
    mTestInput, mTestOut] = natalityData()
%{
USING FERTILITY DATASET
1 - Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1)
2 - Age at the time of analysis. 18-36 (0, 1)
3 - Childhood diseases (ie , chicken pox, measles, mumps, polio) 1) yes, 2) no. (0, 1)
4 - Accident or serious trauma 1) yes, 2) no. (0, 1)
5 - Surgical intervention 1) yes, 2) no. (0, 1)
6 - High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1)
7 - Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1)
8 - Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1)
9 - Number of hours spent sitting per day ene-16 (0, 1)
10 - Output: Diagnosis normal (N-->0), altered (O-->1)
%}
% input data
    filename = 'fertility_Diagnosis.csv';
    delimiterIn = ',';
    Data = importdata(filename,delimiterIn);
    
    % create training and testing matrices
    [entries, attributes] = size(Data);
    entries_breakpoint = round(entries*.50); %set breakpoint for training and testing data at 90% of dataset
    nInputSize=9;
    nOutSize=attributes-nInputSize;
    trainingdata = Data(1:entries_breakpoint,:); %truncate first 90% entries for training data
        mTrainInput = trainingdata(:,1:nInputSize); %90%x9 matrix input training data
        mTrainOut = trainingdata(:,nInputSize+1:end); %90:1 matrix output training data
    testingdata = Data(entries_breakpoint:end,:); %truncate last 10 entries for testing data
        mTestInput= testingdata(:,1:nInputSize); %10:9 matrix input testing data
        mTestOut= testingdata(:,nInputSize+1:end); %10:1 matrix output testing data  
     
end
%%
function fourClassPerceptronClassification ()
    % Define data
    close all, clear all, clc, format compact
    % number of samples of each class
    K = 30;
    % define classes
    q = .5; % offset of classes
    A = [rand(1,K)-q; rand(1,K)+q];
    B = [rand(1,K)+q; rand(1,K)+q];
    C = [rand(1,K)+q; rand(1,K)-q];
    D = [rand(1,K)-q; rand(1,K)-q];
    % plot classes
    figure(2); plot([A(1,:);B(1,:);C(1,:);D(1,:)],[A(2,:);B(2,:);C(2,:);D(2,:)],'*')
    figure(1)
    plot(A(1,:),A(2,:),'bs')
    hold on
    grid on
    plot(B(1,:),B(2,:),'r+')
    plot(C(1,:),C(2,:),'go')
    plot(D(1,:),D(2,:),'m*')
    % text labels for classes
    text(.5-q,.5+2*q,'Class A')
    text(.5+q,.5+2*q,'Class B')
    text(.5+q,.5-2*q,'Class C')
    text(.5-q,.5-2*q,'Class D')
    % define output coding for classes
    a = [0 1]';
    b = [1 1]';
    c = [1 0]';
    d = [0 0]';
    % % Why this coding doesn't work?
    % a = [0 0]';
    % b = [1 1]';
    % d = [0 1]';
    % c = [1 0]';
    % % Why this coding doesn't work?
    % a = [0 1]';
    % b = [1 1]';
    % d = [1 0]';
    % c = [0 1]';

    %Prepare inputs & outputs for perceptron training
    % define inputs (combine samples from all four classes)
    P = [A B C D];
    % define targets
    T = [repmat(a,1,length(A)) repmat(b,1,length(B)) ...
    repmat(c,1,length(C)) repmat(d,1,length(D)) ];
    %plotpv(P,T);

    %Create a perceptron
    net = perceptron;

    % Train a perceptron 
    E = 1;
    net.adaptParam.passes = 1;
    linehandle = plotpc(net.IW{1},net.b{1});
    n = 0;
    while (sse(E) & n<10000)
    n = n+1;
    [net,Y,E] = adapt(net,P,T);
    linehandle = plotpc(net.IW{1},net.b{1},linehandle);
    drawnow;
    end
    % show perceptron structure
    view(net);

    % How to use trained perceptron
    % For example, classify an input vector of [0.7; 1.2]
    p = [0.7; 1.2]
    y = net(p)
    % compare response with output coding (a,b,c,d)
end

function linearPerceptronSeparator()
    % Define input and output data
    close all, clear all, clc, format compact
    % number of samples of each class
    N = 20;
    % define inputs and outputs
    offset = 5; % offset for second class
    x = [randn(2,N) randn(2,N)+offset]; % inputs
    y = [zeros(1,N) ones(1,N)]; % outputs
    % Plot input samples with PLOTPV (Plot perceptron input/target vectors)
    figure(1)
    plotpv(x,y);

    % Create and train perceptron
    net = perceptron;
    net = train(net,x,y);
    view(net);

    % Plot decision boundary
    figure(1)
    plotpc(net.IW{1},net.b{1});
end

function customNeuron()
    %Define one sample: inputs and outputs
    close all, clear all, clc, format compact
    inputs  = [1:6]' % input vector (6-dimensional pattern)
    outputs = [1 2]' % corresponding target output vector

    %Define and customize network
    % create network
    net = network( ...
    1,          ... % numInputs,    number of inputs,
    2,          ... % numLayers,    number of layers
    [1; 0],     ... % biasConnect,  numLayers-by-1 Boolean vector,
    [1; 0],     ... % inputConnect, numLayers-by-numInputs Boolean matrix,
    [0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
    [0 1]       ... % outputConnect, 1-by-numLayers Boolean vector
    );
    % View network structure
    view(net)

    %Define topology and transfer function
    % number of hidden layer neurons
    net.layers{1}.size = 5;
    % hidden layer transfer function
    net.layers{1}.transferFcn = 'logsig';
    view(net);
    
    %Configure network
    net = configure(net,inputs,outputs);
    view(net);
    
    %Train net and calculate neuron output
    % initial network response without training
    initial_output = net(inputs)
    % network training
    net.trainFcn = 'trainlm';
    net.performFcn = 'mse';
    net = train(net,inputs,outputs);
    % network response after training
    final_output = net(inputs)
end

function singleNeuron()
    % Neuron weights
    w = [4 -2]
    % Neuron bias
    b = -3
    % Activation function
     func = 'tansig'
    % func = 'purelin'
    % func = 'hardlim'
    % func = 'logsig'

    % Define input vector
    p = [2 3]
    
    % Calculate neuron output
    activation_potential = p*w'+b
    neuron_output = feval(func, activation_potential)
    
    %lot neuron output over the range of inputs 
    [p1,p2] = meshgrid(-10:.25:10);
    z = feval(func, [p1(:) p2(:)]*w'+b );
    z = reshape(z,length(p1),length(p2));
    plot3(p1,p2,z)
    grid on
    xlabel('Input 1')
    ylabel('Input 2')
    zlabel('Neuron output')
end

