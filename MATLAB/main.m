%% Generate Point Clouds for HAR
%==========================================================================
% Authors #1 Simin Zhu
% Authors #3 Chakir
% Authors #2 Mujtaba
% Version 1.0
%==========================================================================

%% Extract all the files
Dir = pwd;
rootdir = dir(fullfile(Dir, '/Dataset_848')).folder;
myFiles = dir(fullfile(rootdir, '*/*.dat'));

%% Loop through all data files in all folders
for k = 1:length(myFiles)
    % Path of the current data file
    path = strcat(myFiles(k).folder, '/', myFiles(k).name);
    
    % Extract the data sequence
    fileID = fopen(path, 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);
    radarData = dataArray{1};
    clear fileID dataArray ans;

    % Extract radar parameters
    fc = radarData(1); % Center frequency
    Tsweep = radarData(2)/1000; % Sweep time in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
    Data = radarData(5:end); % raw data in I+j*Q format
    fs = NTS/Tsweep; % sampling frequency ADC
    record_length = length(Data)/NTS*Tsweep; % length of recording in s
    nc = record_length/Tsweep; % number of chirps

    %plot the range profile?
    is_plot = 1;

    %% Range-time processing
    Data_range_MTI = RT_Generation(Data,NTS,nc,is_plot);
    %Time axis
    axis_RT_time = linspace(Tsweep,Tsweep*size(Data_range_MTI,2),size(Data_range_MTI,2))';
    %Range axis

    %% Doppler-time processing
    TimeWindowLength = 200;
    [Data_spec_MTI2,idx_r] = Spec_Generation(Data_range_MTI,TimeWindowLength,is_plot);
    %Time axis
    axis_spec_time = linspace(Tsweep*TimeWindowLength,Tsweep*TimeWindowLength*size(Data_spec_MTI2,2),size(Data_spec_MTI2,2))';
    %Velocity axis
    
    %% Block from Mujtaba: Detector
    CFAR_winv = 100;
    CFAR_winh = 1;
    CFAR_wingv = 25;
    CFAR_wingh = 0;
    pfa = 5e-3;
    CFAR_2D_out = CA_CFAR_2D_fast(Data_spec_MTI2,CFAR_winv,CFAR_wingv,CFAR_winh,CFAR_wingh,pfa,1);
    %% Save Point Cloud and Labels

    %% Visualizations
    
end

