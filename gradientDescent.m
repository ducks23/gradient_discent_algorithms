function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta_temp = theta;
for iter = 1:num_iters
    
    for i = 1:length(theta)
        hypothesis_vector = ((X * theta_temp) - y)';
        theta(i) = theta_temp(i) - alpha * 1/m * hypothesis_vector * X(:,i);
    end
    theta_temp = theta;
    J_history(iter) = computeCost(X, y, theta);
    
end

end
