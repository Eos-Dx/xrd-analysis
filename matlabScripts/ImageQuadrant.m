% The next section is autogenerated by Matlab
%%% Set up the Import Options and import the data %%%

clear;
clc;

opts = delimitedTextImportOptions("NumVariables", 256);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29", "VarName30", "VarName31", "VarName32", "VarName33", "VarName34", "VarName35", "VarName36", "VarName37", "VarName38", "VarName39", "VarName40", "VarName41", "VarName42", "VarName43", "VarName44", "VarName45", "VarName46", "VarName47", "VarName48", "VarName49", "VarName50", "VarName51", "VarName52", "VarName53", "VarName54", "VarName55", "VarName56", "VarName57", "VarName58", "VarName59", "VarName60", "VarName61", "VarName62", "VarName63", "VarName64", "VarName65", "VarName66", "VarName67", "VarName68", "VarName69", "VarName70", "VarName71", "VarName72", "VarName73", "VarName74", "VarName75", "VarName76", "VarName77", "VarName78", "VarName79", "VarName80", "VarName81", "VarName82", "VarName83", "VarName84", "VarName85", "VarName86", "VarName87", "VarName88", "VarName89", "VarName90", "VarName91", "VarName92", "VarName93", "VarName94", "VarName95", "VarName96", "VarName97", "VarName98", "VarName99", "VarName100", "VarName101", "VarName102", "VarName103", "VarName104", "VarName105", "VarName106", "VarName107", "VarName108", "VarName109", "VarName110", "VarName111", "VarName112", "VarName113", "VarName114", "VarName115", "VarName116", "VarName117", "VarName118", "VarName119", "VarName120", "VarName121", "VarName122", "VarName123", "VarName124", "VarName125", "VarName126", "VarName127", "VarName128", "VarName129", "VarName130", "VarName131", "VarName132", "VarName133", "VarName134", "VarName135", "VarName136", "VarName137", "VarName138", "VarName139", "VarName140", "VarName141", "VarName142", "VarName143", "VarName144", "VarName145", "VarName146", "VarName147", "VarName148", "VarName149", "VarName150", "VarName151", "VarName152", "VarName153", "VarName154", "VarName155", "VarName156", "VarName157", "VarName158", "VarName159", "VarName160", "VarName161", "VarName162", "VarName163", "VarName164", "VarName165", "VarName166", "VarName167", "VarName168", "VarName169", "VarName170", "VarName171", "VarName172", "VarName173", "VarName174", "VarName175", "VarName176", "VarName177", "VarName178", "VarName179", "VarName180", "VarName181", "VarName182", "VarName183", "VarName184", "VarName185", "VarName186", "VarName187", "VarName188", "VarName189", "VarName190", "VarName191", "VarName192", "VarName193", "VarName194", "VarName195", "VarName196", "VarName197", "VarName198", "VarName199", "VarName200", "VarName201", "VarName202", "VarName203", "VarName204", "VarName205", "VarName206", "VarName207", "VarName208", "VarName209", "VarName210", "VarName211", "VarName212", "VarName213", "VarName214", "VarName215", "VarName216", "VarName217", "VarName218", "VarName219", "VarName220", "VarName221", "VarName222", "VarName223", "VarName224", "VarName225", "VarName226", "VarName227", "VarName228", "VarName229", "VarName230", "VarName231", "VarName232", "VarName233", "VarName234", "VarName235", "VarName236", "VarName237", "VarName238", "VarName239", "VarName240", "VarName241", "VarName242", "VarName243", "VarName244", "VarName245", "VarName246", "VarName247", "VarName248", "VarName249", "VarName250", "VarName251", "VarName252", "VarName253", "VarName254", "VarName255", "VarName256"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

%%% ________________________________________________________________________________________________________________________________ %%%
%%% File paths

pathToFileList = "C:\Custom\data\ArionData\June9Preprocessed\";
fileList = dir(pathToFileList);

%%% ________________________________________________________________________________________________________________________________ %%%
% Constants for quadratic avg calculations

quadrantPoints = []; 
counterRow = 128;

% Assume that (128, 128) is our center
centerCoord = [128, 128];

for sampleFileIndex = 3 : numel( fileList )
    % Path to the file
    path = fullfile( fileList( sampleFileIndex ).folder, fileList( sampleFileIndex ).name );
    % Get the matrix
    matrix = readmatrix( path );

    %%% ________________________________________________________________________________________________________________________________ %%%

    % Here we will do calculations for the quadratic average
    % For instance and for this task, the center is geometrically defined
    % for indexRow = centerCoord(1):1:254 % We fix the row...
    %     counterCol = 128;
    %     for indexCol = centerCoord(2):1:254 % ...and for each column...

    %         if ( matrix( indexRow, indexCol ) == 0 )

    %             quadrantPoints = [

    %                 matrix( indexRow, counterCol );
    %                 matrix( counterRow, indexCol );
    %                 matrix( counterRow, counterCol )
    %             ];

    %             % Apply the average values to our matrix
    %             avg = mean(quadrantPoints);
                
    %             matrix( indexRow, counterCol ) = avg;
    %             matrix( counterRow, indexCol ) = avg;
    %             matrix( counterRow, counterCol ) = avg;

    %         elseif ( matrix( counterCol, indexRow ) == 0 )

    %             quadrantPoints = [
    %                 matrix( indexRow, indexCol );

    %                 matrix( counterRow, indexCol);
    %                 matrix( counterRow, counterCol )
    %             ];

    %             % Apply the average values to our matrix
    %             avg = mean(quadrantPoints);
    %             matrix( indexRow, indexCol ) = avg;

    %             matrix( counterRow, indexCol ) = avg;
    %             matrix( counterRow, counterCol ) = avg;

    %         elseif ( matrix( indexCol, counterRow ) == 0 )

    %             quadrantPoints = [
    %                 matrix( indexRow, indexCol );
    %                 matrix( indexRow, counterCol);

    %                 matrix( counterRow, counterCol )
    %             ];

    %             % Apply the average values to our matrix
    %             avg = mean(quadrantPoints);
    %             matrix( indexRow, indexCol ) = avg;
    %             matrix( indexRow, counterCol ) = avg;
                
    %             matrix( counterRow, counterCol ) = avg;
                
    %         elseif ( matrix( counterCol, counterRow ) == 0 )
    %             quadrantPoints = [
    %                 matrix( indexRow, indexCol );
    %                 matrix( indexRow, counterCol );
    %                 matrix( counterRow, indexCol );

    %             ];

    %             % Apply the average values to our matrix
    %             avg = mean(quadrantPoints);
    %             matrix( indexRow, indexCol ) = avg;
    %             matrix( indexRow, counterCol ) = avg;
    %             matrix( counterRow, indexCol ) = avg;
                
    %         else
    %             quadrantPoints = [
    %                 matrix( indexRow,                 indexCol );
    %                 matrix( indexRow, counterCol );
    %                 matrix( counterRow, indexCol );
    %                 matrix( counterRow, counterCol )
    %             ];

    %             % Apply the average values to our matrix
    %             avg = mean(quadrantPoints);
    %             matrix( indexRow, indexCol ) = avg;
    %             matrix( indexRow, counterCol ) = avg;
    %             matrix( counterRow, indexCol ) = avg;
    %             matrix( counterRow, counterCol ) = avg;
    %         end
    %         counterCol = counterCol - 1;
    %         if ( counterCol == 0 )
    %             counterCol = 1;
    %         end
    %     end
    %     counterRow = counterRow - 1;
    %     if ( counterRow == 0 )
    %         counterRow = 1;
    %     end
            
    % end

    matrixQuadrantAverage = ( matrix + flip(matrix) + flip(matrix, 2) + flip( flip(matrix), 2 ) ) / 4;

    %%% ________________________________________________________________________________________________________________________________ %%%

    % Extract the quadrant
    matrixQuadrant = matrixQuadrantAverage( 128:256, 128:256 );

    quadrantMatrixList(sampleFileIndex, :, :) = matrixQuadrant;
    %%% ________________________________________________________________________________________________________________________________ %%%
end

%%% ________________________________________________________________________________________________________________________________ %%%

% Distance calculations
for matrixIndex = 1 : 1 : size(quadrantMatrixList, 1)
    for matrixIndexToCompareWith = 1 : 1 : size(quadrantMatrixList, 1)
        
        A = quadrantMatrixList(matrixIndex, :, :);
        A = reshape(A, size(A, 2), size(A, 3) );
        B = quadrantMatrixList(matrixIndexToCompareWith, :, :);
        B = reshape(B, size(B, 2), size(B, 3) );
        
        % Doc: https://www.mathworks.com/help/stats/pdist2.html
        Distance = pdist2(A, B, 'euclidean');
        distance( matrixIndexToCompareWith, :, :) = Distance;
    end
    distanceList( matrixIndex, :, :, : ) = distance;
end



