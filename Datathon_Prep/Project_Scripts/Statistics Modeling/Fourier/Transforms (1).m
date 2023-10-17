function Transforms()
    %-----------------------------------------------
    % Author: Prof. Juan B. Gutiérrez - juan.gutierrez3@utsa.edu
    % Date: May 2020
    % Shared under Creative Commons Attribution CC BY 3.0 US
    % You are free to share or adapt.
    % You must give appropriate credit, provide a link to the license,
    % and indicate if changes were made
    %-----------------------------------------------
    clear;
    addpath('regtools');
    % First, produce a random time series with trend
    t = (0:.1:pi)';
    n = size(t,1);
    s = std( cos(t).*t)/rand(); 
    x = (cos(t).*t + s*rand(n,1));
    %t = -1:0.01:1; x = t.^2;
    % Set parameters for smoothing
    b = 5;         % Number of basis functions
    nLambda  = 0; % HP filter smoothing parameter.
    % Now, produce smooth time series 

    x1 = DFT(b,t,x); 
    %x2 = SmoothFourier(m,x); 
    x3 = funHodrickPrescott(x,nLambda);  
    %b = 100;          % Number of basis to preserve. 
    x4 = IntegralTransform(t, x, b, 0); 
    %b = 50;          % Number of basis to preserve. 
    x5 = IntegralTransform(t, x, b, 1);
    %b = 50;          % Number of basis to preserve. 
    x6 = IntegralTransform(t, x, b, 2);
    %[x4, nLambda, R] = funRegularization(t,x, b);
    % Visualize original function
    figure(1);clf;     
    title('Function regularization comparisons'); 
    hold on
    plot(t, x, 'color',[.5 .5 .5]); 
    %plot(t, x1, 'color', [0 77 64]/256, 'linewidth',2);
    plot(t, x3, 'b', 'linewidth',2.5);
    plot(t, x4, 'color',[30 136 229]/256, 'linewidth',2);
    plot(t, x5, 'color', [216 27 96]/256, 'linewidth',2);
    plot(t, x6, 'color', [126 132 107]/256, 'linewidth',2);
    legend('Raw', ... %'DFT', ...
        'cos((k-1)*pi*x/l)', 'x^{(k-1)}*exp((k-1)*x/l)', 'x^{(k-1)}', 'location', 'southwest');
    axis tight;
    hold off
    
    return 
    
    % Visualize multiple regularizations
    figure(2);clf;     
    title('Function regularization comparisons'); 
    hold on
    plot(t, x, 'color',[.5 .5 .5]); 
    plot(t, R(:,1), 'color', [230 159 0]/256, 'linewidth',2);
    plot(t, R(:,2), 'color', [86 180 233]/256, 'linewidth',2);
    plot(t, R(:,3), 'color', [0 158 115]/256, 'linewidth',2);
    plot(t, R(:,4), 'color', [240 228 66]/256, 'linewidth',2);
    plot(t, R(:,5), 'color', [0 114 178]/256, 'linewidth',2);
    plot(t, R(:,6), 'color', [213 121 167]/256, 'linewidth',2);
    plot(t, R(:,7), 'color', [216 27 96]/256, 'linewidth',2);
    legend('Raw','Exponential', 'Monomials', 'Legendre', ...
        'Chebyshev 1st','Chebyshev 2nd','Gegenbauer','Jacobi', 'southwest');
    axis tight;
    hold off
end

function y1 = DFT(m,x,f)
    n = length(x);
    % Calculate the elements of the discrete Fourier Transform
    % This can be done with MATLAB's fft function
    w = exp(2*pi/n * j);
    for i=0:n-1
        for k=i:n-1
            F(i+1,k+1) = w^(i*k);
            F(k+1,i+1) = F(i+1,k+1);
        end
    end
    T = F*f;
    
    % Now filter out frequencies. Leave only m frequencies
    B = sort(T,'descend');
    [~,uIdx] = ismember(T,B); % Index to reverse sort: f = B(uIdx)
    idx = m+1:n;
    B(idx) = 0;
    T = B(uIdx);  
    
    % Calculate the inverse Fourier transform
    % This can be done with MATLAB's ifft function; this is just an exercise
    w = exp(-2*pi*j/n);
    for i=0:n-1
        for k=i:n-1
            iF(i+1,k+1) = w^(i*k);
            iF(k+1,i+1) = iF(i+1,k+1);
        end
    end
    
    % Filter before inverting the transform
    y1 = real(iF*T)/n;
end

function y2 = SmoothFourier(m,x)
    % x = raw data vector - equal spacing is assumed
    % m = frequencies that we want to keep
    y2 = fft(x);
    n = size(x,1);
    B = sort(y2,'descend');
    [~,uIdx] = ismember(y2,B); % Index to reverse sort: f = B(uIdx)
    idx = m+1:n;
    B(idx) = 0;
    y2 = real(ifft(B(uIdx)));
end

function s = IntegralTransform(x, y, b, nBasis)
    syms sx sy
    x = x-x(1); % guaranteed start at the origin
    l = x(end)-x(1); % length of the interval
    alpha = rand; beta = rand;
    for k=1:b 
       % Try multiple basis, except cBasis{k} = cos((k-1)*pi*t/l); 
        switch nBasis
            case 0
                cBasis{k} = cos((k-1)*pi*x/l);  
            case 1
                cBasis{k} = x.^(k-1).*exp((k-1)*x/l);  
            case 2
                cBasis{k} = x.^(k-1);
            case 3
                alpha = 0; beta = alpha; % Legendre polynomials = legendreP(k,sx)
                sy = jacobiP(k,alpha,beta,sx);
                cBasis{k} = double(subs(sy, sx, x));
            case 4
                alpha = -0.5; beta = alpha; % Chebyshev pol. 1st kind = chebyshevT(k,sx)
                sy = jacobiP(k,alpha,beta,sx);                
                cBasis{k} = double(subs(sy, sx, x));
            case 5
                alpha = 0.5; beta = alpha; % Chebyshev pol. 2nd kind = chebyshevU(k,sx)
                sy = jacobiP(k,alpha,beta,sx);                
                cBasis{k} = double(subs(sy, sx, x));
            case 6
                beta = alpha; % Gegenbauer polynomials= gegenbauerC(k,a,sx)
                sy = jacobiP(k,alpha,beta,sx);                
                cBasis{k} = double(subs(sy, sx, x));
            case 7
                % Jacobi polynomials
                sy = jacobiP(k,alpha,beta,sx);                
                cBasis{k} = double(subs(sy, sx, x));
        end
    end     
    for i = 1:b
        for j = 1:b
            A(i,j) = Integrator(x, cBasis{i} .* cBasis{j});
        end
        w(i,1) = Integrator(x, y .* cBasis{i});
    end
    % Find coefficients
    c =  A \ w; % This notation solves the the system Ac = w
    % Now create the series with the basis functions
    s = 0;
    for i=1:b 
       s = s + c(i) *  cBasis{i};
    end
end

function I = Integrator(t, x)
    I = 0;
    n = length(x);
    dx = (t(end)-t(1))/n;
    for i=2:n
        I = I + (x(i-1)+x(i))/2*dx;
    end
end

function s = IntegralTransform2(t, x, b, cBasis)
    for i = 1:b
        for j = 1:b
            A(i,j) = Integrator(t, cBasis{i} .* cBasis{j});
        end
        w(i,1) = Integrator(t, x .* cBasis{i});
    end
    % Find coefficients
    c =  A \ w;
    % Now create the series with the basis functions
    s = 0;
    for i=1:b
       s = s + c(i) *  cBasis{i};
    end
end


function ann()
    % Define the number of inputs, hidden layers and outputs
    inputs = 2;
    hidden = 2;
    outputs = 1;
    
    % Define the weight matrices and bias
    weights1 = rand(inputs+1, hidden);
    weights2 = rand(hidden+1, outputs);
    
    % Define the input vector
    input_vector = [1 2];
    
    % Add bias to the input
    input_vector = [input_vector 1];
    
    % Perform the feedforward calculation
    hidden_layer = sigmoid(input_vector * weights1);
    hidden_layer = [hidden_layer 1];
    output = sigmoid(hidden_layer * weights2);
    
    % Define the target output
    target = 0.5;
    
    % Calculate the error
    error = target - output;
    
    % Perform backpropagation
    delta_output = error .* sigmoid_derivative(output);
    error_hidden = delta_output * weights2';
    delta_hidden = error_hidden(1:end-1) .* sigmoid_derivative(hidden_layer(1:end-1));
    
    % Update the weights
    learning_rate = 0.1;
    weights2 = weights2 + hidden_layer' * delta_output * learning_rate;
    weights1 = weights1 + input_vector' * delta_hidden * learning_rate;
    
    disp(output);
    
    function y = sigmoid(x)
        y = 1 ./ (1 + exp(-x));
    end
    
    function y = sigmoid_derivative(x)
        y = x .* (1 - x);
    end

end