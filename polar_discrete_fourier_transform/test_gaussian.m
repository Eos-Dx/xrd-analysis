%%A-5. Forward transform of Gaussian function
N2=5; %number of sample points in angular direction
N1=4; %number of sample points in radial direction
M=(N2-1)/2; %highest order of bessel function
R=10;% space limit
Wp=10; % band limit
a=0.1;
load('zeromatrix.mat')
theta=thetamatrix_SpaceLimited(N2,N1); %Sample point in angular direction in space domain.
r=rmatrix_SpaceLimited(N2,N1,R,zeromatrix);%Sample point in radial direction in space domain.
psi=psimatrix_SpaceLimited(N2,N1);%Sample point in angular direction in frequency domain.
rho=rhomatrix_SpaceLimited(N2,N1,R,zeromatrix);%Sample point in radial direction in frequency domain.
[x,y]=pol2cart(theta,r); %sample points in Cartesian coordinates in space domain
[x1,y1]=pol2cart(psi,rho); %sample points in Cartesian coordinates in frequency domain


%Discretizing the function
gau = @(x) exp(-(a*x).^2); 
f=gau(r);

% DFT
fnk=circshift(fft(circshift(f,M+1,1),N2,1),-(M+1),1);
% DHT
for n=-M:M
	ii=n+M+1;
	% zero2=zeromatrix(5001-abs(n),:);
	zero2=zeromatrix(end-abs(n),:);
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

figure(1);
title('\fontsize{24}Discrete Forward Transform [dB]');
imagesc(20*log10(abs(real(TwoDFT))));

%creating a discrete 2D Fourier transform
gau2 = @(x) pi/a^2*exp((-(x/a).^2)/4); 
trueFunc=gau2(rho);


%calculating the dynamic error from transform and true function
calc_error= 20*log10(abs(trueFunc- TwoDFT)/max(max(abs(TwoDFT))));

mean1=mean(mean(calc_error)); % Average dynamic error
max1=max(max(calc_error)); % Maximum dynamic error
