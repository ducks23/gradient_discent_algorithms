function J = computeCost(X, y, theta)

m = length(y); % number of training examples

predictions = X * theta;

sqr_differences = (predictions-y).^2;

J = 1/(2*m) * sum(sqr_differences);

end
