function [Data_spec_MTI2,idx_r] = Spec_Generation(Data_range_MTI,TimeWindowLength,is_plot)

%% Target localization
Ns = size(Data_range_MTI,2);
% Index of the largest value along range 
Data_range_MTI_temp = Data_range_MTI;
Data_range_MTI_temp(1:3,:) = 0;
[~,idx_r] = max(Data_range_MTI_temp,[],1);
% Apply median filter
idx_r = round(medfilt1(idx_r,100));

%% Generating the Spectrogram
% Parameters for STFT
OverlapFactor = 0;
OverlapLength = round(TimeWindowLength*OverlapFactor);
Pad_Factor = 1;
FFTPoints = Pad_Factor*TimeWindowLength;
Data_spec_MTI2=0;

% range intervals
intervals = 5;
win = hamming(intervals*2+1);

%STFT
for i = 1:1:2*intervals+1
    idx_r_temp = idx_r - intervals -1 + i;
    idx_r_temp(idx_r_temp<=0) = 1;
    Data_before_fft = Data_range_MTI(sub2ind(size(Data_range_MTI),idx_r_temp,1:Ns));
    Data_MTI_temp = fftshift(spectrogram(Data_before_fft,hamming(TimeWindowLength),OverlapLength,FFTPoints),1);
    Data_spec_MTI2 = Data_spec_MTI2 + win(i)*abs(Data_MTI_temp);
end
Data_spec_MTI2=flipud(Data_spec_MTI2);

% Plot Spectrogram
if is_plot == 1
    figure(5)
    imagesc(20*log10(Data_spec_MTI2)); colormap('jet'); axis xy;
    colormap;
    clim = get(gca,'CLim');
    set(gca, 'CLim', clim(2)+[-40,0]);
    xlabel('Time[s]', 'FontSize',16);
    ylabel('Velocity [m/s]','FontSize',16);
    set(gca, 'FontSize',16);
end
