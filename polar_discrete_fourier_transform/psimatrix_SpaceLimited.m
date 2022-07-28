%% A-3. Psi matrix for space limited function
% N1 sample size in radial direction
% N2 sample size in angular direction
function psi=psimatrix_SpaceLimited(N2,N1)
	b2 = 2*pi/N2;
	progr2= -pi + b2/2 : b2 : pi - b2/2;
	psi = progr2(:)*ones(1,N1-1);
