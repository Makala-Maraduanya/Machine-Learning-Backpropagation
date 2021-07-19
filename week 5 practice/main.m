%this is the driver function
data=load('seeds_dataset.txt');

%setting the data from the file
X=data(:,1:7);
y=data(:,8);

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

%initializing the required variables
input_size=7; %values from the input layer minus the bias unit
hidden_units=5; %number of units in the hidden layer 
output_layer=3; %number of units in the output layer

%randomly getting values for Theta1 and Theta2 (weights)
Theta1=initializeWeights(input_size,hidden_units);
Theta2=initializeWeights(hidden_units,output_layer);

%placing the values of the weights in a single variable
params=[Theta1(:);Theta2(:)];

%displaying the cost
J=costFunction(params,input_size,hidden_units, output_layer, X, y, lambda);

fprintf('The cost function: %f\n',J);
                          
fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = initializeWeights(input_size, hidden_units);
initial_Theta2 = initializeWeights(hidden_units, output_layer);

% Unroll parameters
initial_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('Program paused. Press enter to continue.\n');
pause;

%Optimizing the cost function
fprintf('Optimizing the cost function i.e training the network');

%setting the necessary options for using fminunc
options = optimset('GradObj','on','MaxIter', 50);
costFunc = @(p) costFunction(p,input_size,hidden_units,output_layer, X, y, lambda);

%running fminunc
[final_params, cost] = fminunc(costFunc, initial_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(final_params(1:hidden_units * (input_size + 1)), ...
                 hidden_units, (input_size + 1));

Theta2 = reshape(final_params((1 + (hidden_units * (input_size + 1))):end), ...
                 output_layer,(hidden_units + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%predict the values of Theta1 and Theta2 are now obtained after training
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
