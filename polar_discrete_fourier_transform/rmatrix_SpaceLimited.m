%% A-2. r matrix for space limited function
% N1 sample size in radial direction
% N2 sample size in angular direction
% R effective space limit
% zeromatrix is precalculated Bessel zeros
function r=rmatrix_SpaceLimited(N2,N1,R,zeromatrix)
	M=(N2-1)/2;
	for ii=1:N2;
		p=ii-1-M;
		for k=1:N1-1;
			% zero2=zeromatrix(5001-abs(p),:);
			zero2=zeromatrix(end-abs(p),:);
			jpk=zero2(k);
			jpN1=zero2(N1);
			r(ii,k)=(jpk/jpN1)*R;
		end
	end
