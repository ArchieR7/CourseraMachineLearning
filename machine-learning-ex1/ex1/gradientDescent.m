function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cXost function (computeCost) and gradient here.
    %
    % temp_theta = theta;
    % s = 0;
    % for i = 1:m
    %    s += theta(1) + theta(2) * X(i,2) - y(i);
    % end
    % temp_theta(1) -= alpha / m * s;
    % s = 0;
    % for i = 1:m
    %    s += (theta(1) + theta(2) * X(i,2) - y(i)) * X(i,2);
    % end
    % temp_theta(2) -= alpha / m * s;
    h = X*theta;
    a = X'*(h-y)/m;
    theta = theta - alpha.*a;


    % theta = temp_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
