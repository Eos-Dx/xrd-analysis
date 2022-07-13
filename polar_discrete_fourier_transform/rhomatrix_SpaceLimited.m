%%A-4. Rho matrix for space limited function
% N1 sample size in radial direction
% N2 sample size in angular direction
% R effective space limit
% zeromatrix precalculated Bessel zeros
function rho=rhomatrix_SpaceLimited(N2,N1,R,zeromatrix)
	M=(N2-1)/2;
	for ii=1:N2;
		q=ii-1-M;
		for l=1:N1-1;
			% zero2=zeromatrix(5001-abs(q),:);
			zero2=zeromatrix(201-abs(q),:);
			jql=zero2(l);
			rho(ii,l)=jql/R;
		end
	end
