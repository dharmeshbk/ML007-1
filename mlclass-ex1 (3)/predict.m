function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1)
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X_with_bias = [ones(m,1) X] ; %[X,1] ;% Add one bias 1 as column , now vector dimension is 1 by 401
%size(X_with_bias)
%Theta1; %% dimension is 25 by 401

a_one = X_with_bias * Theta1';%% Dimension is 1 by 401
%size(a_one)


z_two = sigmoid(a_one);
%size(z_two)


a_two = [ones(size(z_two),1),z_two]; %% Add column for bias 1
%size(a_two);

%a2 = [ones(size(z2),1) sigmoid(z2)]

z_three = a_two * Theta2';
%size(z_three)

hyp = sigmoid(z_three);

[predict_max, index_max] = max(hyp, [], 2);

p = index_max

% =========================================================================


end
