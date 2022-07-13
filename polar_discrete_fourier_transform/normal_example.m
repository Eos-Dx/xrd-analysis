%%A-5. Forward transform of Gaussian function
N2=15; %number of sample points in angular direction
% N1=383; %number of sample points in radial direction
N1=30; %number of sample points in radial direction
M=(N2-1)/2; %highest order of bessel function
R=40;% space limit
Wp=30; % band limit
a=0.1;
load('zeromatrix.mat');
load('normal_polar.mat');
theta=thetamatrix_SpaceLimited(N2,N1); %Sample point in angular direction in space domain.
r=rmatrix_SpaceLimited(N2,N1,R,zeromatrix);%Sample point in radial direction in space domain.
psi=psimatrix_SpaceLimited(N2,N1);%Sample point in angular direction in frequency domain.
rho=rhomatrix_SpaceLimited(N2,N1,R,zeromatrix);%Sample point in radial direction in frequency domain.
[x,y]=pol2cart(theta,r); %sample points in Cartesian coordinates in space domain
[x1,y1]=pol2cart(psi,rho); %sample points in Cartesian coordinates in frequency domain


%Discretizing the function
% gau = @(x) exp(-(x).^2); 
% f=gau(r);
f = normal_polar;

% DFT
fnk=circshift(fft(circshift(f,M+1,1),N2,1),-(M+1),1);
% DHT
for n=-M:M
	ii=n+M+1;
	% zero2=zeromatrix(5001-abs(n),:);
	zero2=zeromatrix(201-abs(n),:);
	jnN1=zero2(N1);
	if n<0
		Y=((-1)^abs(n))*YmatrixAssembly(abs(n),N1,zero2);
	else
		Y=YmatrixAssembly(abs(n),N1,zero2);
	end
	fnl(ii,:)=(Y*fnk(ii,:)')';
	Fnl(ii,:)=fnl(ii,:)*(2*pi*(i^(-n)))*(R^2/jnN1);
end
% IDFT
TwoDFT=circshift(ifft(circshift(Fnl,M+1,1),N2,1),-(M+1),1);

writematrix(TwoDFT, "normal_TwoDFT.txt");
