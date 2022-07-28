%%A-4. Rho matrix for space limited function
% N1 sample size in radial direction
% N2 sample size in angular direction
% R effective space limit
% zeromatrix precalculated Bessel zeros
function rho=rhomatrix_SpaceLimited_vectorized(N2,N1,R,zeromatrix)
	M=(N2-1)/2;
	iirange = 1:N2;
	qrange = iirange-1-M;
	lrange = 1:N1-1;

	jql=zeromatrix(end-abs(qrange),lrange);
	rho(iirange, lrange) = jql/R;
end
