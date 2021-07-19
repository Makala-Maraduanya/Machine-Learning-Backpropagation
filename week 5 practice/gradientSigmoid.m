function g=gradientSigmoid(z)
   %this function is key in back propagation
  g = zeros(size(z));
  
  g=sigmoid(z).*(1-sigmoid(z));
endfunction
