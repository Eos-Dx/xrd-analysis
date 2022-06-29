clear 

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


%basepathcancer="C:\Users\aplaz\Desktop\01_BUSINESS (APL)\ARION\XRD DATA\April21May04\";
%basepathcancer="C:\Users\aplaz\Desktop\01_BUSINESS (APL)\ARION\XRD DATA\05-09-2022\processed\textfiles\";

basepathcancer="C:\Custom\data\ArionData\June9Preprocessed\";

DirlistC=dir(basepathcancer);
filelist=ls(basepathcancer);

%basepathnormal="C:\Users\aplaz\Desktop\01_BUSINESS (APL)\ARION\XRD DATA\ARION 04212022\PROCESSED files\normal_txt\";
%DirlistN=dir(basepathnormal);

% --- --- --- %

%Start by running through  cancer files:
for Ncancer=3:numel(DirlistC)
    path=fullfile(DirlistC(Ncancer).folder,DirlistC(Ncancer).name);
    Ac = readtable(path, opts);

    Ac=table2array(Ac);

    maxedge=165;
    minedge=85;
    innercut=50;
    band=128-innercut/2-minedge;
    
    band54=4;     %must be even number   
    width54=78;     %width of 5-4 peak integration
    
    bandM=4;       %meridinal scatter integration band width

    %integrate meridinal scattering 
    meridinalcount=1; %pixel number in the integral
    meridinal9C=0;

    for column=128-bandM/2:128+bandM/2     
        for row=128-innercut/2-band:128-innercut/2                 %top patch                  
                meridinal9C=meridinal9C+Ac(row,column); %intergral of 9-meridinal scattering
                meridinalcount=meridinalcount+1;
    
        end
    
        for row=128+innercut/2:128+innercut/2+band                 %bottom patch
                meridinal9C=meridinal9C+Ac(row,column);
                meridinalcount=meridinalcount+1;
        end
    end


    integralmerC(Ncancer-2)=meridinal9C;

    horizontalcount=1;
    horizontal9C=0;

        for row=128-bandM/2:128+bandM/2   
            for column=128-innercut/2-band:128-innercut/2                 %left eye
                horizontal9C=horizontal9C+Ac(row,column);
                horizontalcount=horizontalcount+1;
            end

            for column=128+innercut/2:128+innercut/2+band                 %right eye
                horizontal9C=horizontal9C+Ac(row,column);
                horizontalcount=horizontalcount+1;
            end
        end


integralhorC(Ncancer-2)=horizontal9C;
scatterratioCANCER(Ncancer-2)=integralhorC(Ncancer-2)/integralmerC(Ncancer-2);


% integrate the 5-4A scattering  band -4 pixels wide
    integral54=0;
    c1=1;
    for column=128-band54/2:128+band54/2
        r1=1;
        for row=6:width54
        
            integral54=integral54+Ac(row,column);
            peak54C(r1,c1)=Ac(row,column);                              % build 54 peak pixel array
            r1=r1+1 ;
        end
            
        for row=256-width54:250
            
                integral54=integral54+Ac(row,column);
    
        end
        c1=c1+1;
    end

peak54C=mean(peak54C,2);            %peak54 curve is average of rows
[M,I]=max(peak54C);                  %value of max of peak 54
max54locationC(Ncancer-2)=I;
max54C(Ncancer-2)=M;                %position of maximum of peak 54


integral54CANCER(Ncancer-2)=integral54;     %integral of 54 band

ratio54to9C(Ncancer-2)=integral54CANCER(Ncancer-2)/integralmerC(Ncancer-2);

end

%T=table(filelist,scatterratioCANCER');

%Ncancer

% % Write results to a Spreadship
% filename = 'ResultsJune9.xlsx';
% 
% writematrix(Ac,filename,'Sheet','Ac');
% 
% % writematrix(band,filename,'Sheet','Band');
% % writematrix(band54,filename,'Sheet','Band54');
% % writematrix(bandM,filename,'Sheet','bandM');
% % writematrix(c1,filename,'Sheet','c1');
% % writematrix(column,filename,'Sheet','column');
% % writematrix(horizontal9C,filename,'Sheet','horizontal9C');
% % writematrix(horizontalcount,filename,'Sheet','horizontalcount');
% % writematrix(I,filename,'Sheet','I');
% % writematrix(innercut,filename,'Sheet','innercut');
% % writematrix(integral54,filename,'Sheet','integral54');
% 
% writematrix(integral54CANCER,filename,'Sheet','integral54CANCER');
% writematrix(integralhorC,filename,'Sheet','integralhorC');
% writematrix(integralmerC,filename,'Sheet','integralmerC');
% 
% % writematrix(M,filename,'Sheet','M');
% 
% writematrix(max54C,filename,'Sheet','max54C');
% writematrix(max54locationC,filename,'Sheet','max54locationC');
% 
% % writematrix(maxedge,filename,'Sheet','maxedge');
% % writematrix(meridinal9C,filename,'Sheet','meridinal9C');
% % writematrix(meridinalcount,filename,'Sheet','meridinalcount');
% % writematrix(minedge,filename,'Sheet','minedge');
% % writematrix(Ncancer,filename,'Sheet','Ncancer');
% 
% writematrix(peak54C,filename,'Sheet','peak54C');
% 
% % writematrix(r1,filename,'Sheet','r1');
% 
% writematrix(ratio54to9C,filename,'Sheet','ratio54to9C');
% 
% % writematrix(row,filename,'Sheet','row');
% 
% writematrix(scatterratioCANCER,filename,'Sheet','scatterratioCANCER');
% 
% % writematrix(width54,filename,'Sheet','width54');

