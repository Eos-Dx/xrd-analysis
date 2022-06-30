function TwoDFastFourierTransform(dataFilePath)
    %% Code to import raw data and take 2D FFT
    
    %%% ________________________________________________________________________________________________________________________________ %%%
    %%% File paths setup
    % Set files directory
    % Read in all data files (fullfile returns OS-agnostic full path)
    filenameFormat = "*A*.txt";
    fullFilePath = fullfile(dataFilePath, filenameFormat);
    filesStruct = dir(fullFilePath);
    
    %%% ________________________________________________________________________________________________________________________________ %%%
    
    % Create an object to store the data and the fft
    sampleStore = zeros(numel(filesStruct),256,256);
    fftStore = zeros(size(sampleStore));
    
    % Extract the matrix
    for sampleIndex = 1 : numel(filesStruct)
        % Read sample matrix %
    
        pathToFile = fullfile(dataFilePath, filesStruct(sampleIndex).name);
        sampleMatrix = readmatrix(pathToFile);
        sampleStore(sampleIndex,:,:) = sampleMatrix;
    
        %%% ________________________________________________________________________________________________________________________________ %%%
        % Fourier calculations
    
        % For the fast Fourier transform, documentation is here: https://www.mathworks.com/help/matlab/ref/fft2.html
        fast2DFourierTransform = fft2(sampleMatrix);
        fftStore(sampleIndex,:,:) = fftshift(fast2DFourierTransform);
    
        %%% ________________________________________________________________________________________________________________________________ %%%
    end
    
    % The imagesc command display image with scaled colors. Documentation is here: https://www.mathworks.com/help/matlab/ref/imagesc.html
    
    % Y = fftshift(X) rearranges a Fourier transform X by shifting the zero-frequency component to the center of the array.
    % More about fftshift here: https://www.mathworks.com/help/matlab/ref/fftshift.html
    % We can consider this like a centering. Without this centering, our transform will unreadable on the image.
    % I use log2 to enhance regions with small amplitude.
    
    % Display the image and its 2D FFT
    % Use 20*log10+1 (denoted as [dB+1]) where dB is decibels
    % The +1 is so we do not take the log of zero
    
    % Set figure properties
    fig = figure("Name", "2D Discrete Fast Fourier Transform");
    
    tileColumns = 2;
    tileRows = 1;
    tiledlayout(tileRows, tileColumns);
    nexttile;
    originalImage = reshape(sampleStore(sampleIndex,:,:),256,256);
    originalImagedB1 = 20*log10(1+originalImage);
    originalImage = imagesc(originalImagedB1);
    title("Original Image [dB+1]");
    
    nexttile;
    fft2Image = reshape(fftStore(sampleIndex,:,:),256,256);
    fft2ImageMagnitudedB1 = 20*log10(1+abs(fft2Image));
    result2dFourierTransform = imagesc(fft2ImageMagnitudedB1);
    title("2D FFT Magnitude [dB+1]");
    
    % Set the figure proportions according to tiledlayout size
    fig.Position(3) = tileColumns*fig.Position(3);
    fig.Position(4) = tileRows*fig.Position(4);
end