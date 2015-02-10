plotinit
data=read('cluster');
units=20;
% Plot data and place rbf units randomly
vqinit;
singlewinner=1;

%emstepb; %One iteration
emiterb; %Multiple iterations
