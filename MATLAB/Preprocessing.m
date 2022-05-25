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

    % Calculate axis

    %plot the range profile?
    is_plot = 1;

    %% Range-time processing
    Data_range_MTI = RT_Generation(Data,NTS,nc,is_plot);

    %% Doppler-time processing
    [Data_spec_MTI2,idx_r] = Spec_Generation(Data_range_MTI,is_plot);

    %% Block from Mujtaba: Detector
    
    %% Save Point Cloud and Labels

    %% Visualizations
    
end

