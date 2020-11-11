function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = sigmoid(X*theta);
%J = sum(-y.*log(h) - (1-y).*log(1-h)) + (lambda/2)*sum(theta.^2);
%J = (1/m)*J;
 z = X * theta;      % m x 1
  h_x = sigmoid(z);  % m x 1 
  
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  
  J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term;
temp = (1/m)* (X'*(h-y));
k = (lambda/m)*(theta);
[a ,b] = size(theta);
for i = 2:a
    for j=1:b
        p(i,j) = temp(i,j) + k(i,j);
    end 
end
p(1,1) = temp(1,1);
grad = p;




% =============================================================

end
