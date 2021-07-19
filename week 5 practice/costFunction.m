function [J grad]=costFunction(params,input_size,hidden_units,output_layer,X,y,lambda)
  
  % Setup some useful variables
  m = size(X, 1);
  
  
  %unrolling values for theta1 and theta2
  Theta1=reshape(params(1:hidden_units*(input_size+1)),hidden_units,(input_size+1));
  Theta2=reshape(params((1+(hidden_units*(input_size+1))):end),output_layer,(hidden_units+1))
  
  %important variables that need to be returned
  J=0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  %placing the bias values in X
   X=[ones(m,1) X]; %this is a1 as well 210*8
   
   %calculating activation for the first layer
   z2=Theta1*X'; %5*8 multiplied by 8*210
   a2=sigmoid(z2);
   
   %adding ones in a2 (Note that a2' for easier addition of the bias)
   a2=[ones(size(a2,2),1) a2']; %210*5 plus the bias values to become 210*6
   
   %calculating values for the output layer
   z3=Theta2*a2'; %3*6 multiplied by 6*210
   a3=sigmoid(z3);
   
   %creating a matrix where y=0 or 1
   y_new = zeros(output_layer, m); % 3*210
   for i=1:m,
      y_new(y(i),i)=1;
   end

   J=-(y_new).*log(a3)-(1-y_new).*log(1-a3);
   
   J=sum(sum(J))*(1/m);
   
   
   %getting the values of the weights minus the first one(bias)
   t1 = Theta1(:,2:size(Theta1,2));
   t2 = Theta2(:,2:size(Theta2,2));
   
   %regularization
   Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

   J=J+Reg;
   
   
   
   %---------------backpropagation
   for t=1:m,
     %---------------step 1
     a_1=X(t,:); %setting a row of X in the first activation layer/input layer
     
     %forward propagating
     a_1=a_1';
     %first layer multiplication
     z_2=Theta1*a_1;
     
     %first layer translating to values for the second layer/hidden layer
     a_2=sigmoid(z_2);
     
     %adding the bias values for the second layer
     a_2=[1; a_2];
     
     %calculating from the hidden layer to the outputlayer
     z_3=Theta2*a_2;
     a_3=sigmoid(z_3);
     
     %-----------------step 2
     %subtracting actual y from the predicted value
     delta3=a_3-y_new(:,t);
     
     %----------------step3
     %backpropagation
     z_2=[1;z_2];
     
     delta2=(Theta2'*delta3).*gradientSigmoid(z_2);
     
     %getting rid of the bias
     delta2 = delta2(2:end);
     
     %-----------------step4
     %updating the DELTAS
     Theta2_grad = Theta2_grad + delta3 * a_2';
     Theta1_grad = Theta1_grad + delta2 * a_1';
   endfor
   
   % Step 5
   Theta2_grad = (1/m) * Theta2_grad; 
   Theta1_grad = (1/m) * Theta1_grad; 


% -------------------------------------------------------------
%regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

       
endfunction
