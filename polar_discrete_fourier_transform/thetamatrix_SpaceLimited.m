%% A-1. Theta matrix for space limited function
% Reviewer Hans Feichtinger is acknowledged for the suggested code modifications
% N1 sample size in radial direction
% N2 sample size in angular direction
function theta=thetamatrix_SpaceLimited(N2,N1)
	b2 = 2*pi/N2;
	progr2= -pi + b2/2 : b2 : pi - b2/2;
	theta = progr2(:)*ones(1,N1-1);
