function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% First, compute the sigmoid for easy reuse
h = sigmoid(X*theta);

% Compute each part of the cost using the previously found  sigmoid
% function, which should allow for easier code readability and debugging.
h1 = y' * log(h);
h2 = (1-y)' * log(1 - h);

% Finalize computing the cosrt and assign to the appropriate variable.
J = (-1 / m) * (h1 + h2);

% Previously calulated sigmoid allows for easy computation of the gradient.
grad = (X' * (h - y)) / m;

% =============================================================

end
