%%A-5. Y matrix Assembly Function
% Y is the N-1 x N-1 transformation matrix to be assembled
% n is the order of the bessel function
% N is the size of the transformation matrix
%zeros are the bessel zeros passed to the function

function Y = YmatrixAssembly(n,N,zero)
	%tic


	for l=1:N-1

		for k=1:N-1

			jnk=zero(k);
			jnl=zero(l);
			jnN=zero(N);
			jnplus1=besselj(n+1, jnk);

			Y(l,k)=(2*besselj(n,(jnk*jnl/jnN)))/(jnN*jnplus1^2);


		end
	end

	%toc

end
