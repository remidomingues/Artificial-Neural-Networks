x = (0:0.1:2*pi)';
units = 15;
f = square(2 * x);

makerbf;

Phi = calcPhi(x,m,var);
w = Phi\f; 
y = Phi*w;
rbfplot1(x,y,f,units)