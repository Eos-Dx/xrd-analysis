%% A-2. r matrix for space limited function
% N1 sample size in radial direction
% N2 sample size in angular direction
% R effective space limit
% zeromatrix is precalculated Bessel zeros
function r=rmatrix_SpaceLimited_vectorized(N2,N1,R,zeromatrix)
	M=(N2-1)/2;
	iirange = 1:N2;
	prange=iirange-1-M;
	krange=1:N1-1;

	zero2_combined = zeromatrix(end-abs(prange),1:N1);

	jpk=zero2_combined(:, krange);
	jpN1=zero2_combined(:, end);

	r(iirange,krange)=(jpk./jpN1)*R;
end
