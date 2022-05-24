function Data_range_MTI = RT_Generation(Data,NTS,nc,is_plot)

%% Reshape the data into fast and slow time
Data_time=reshape(Data, [NTS nc]);

%% Prepare windowing function, hamming window
win = repmat(hamming(NTS),1,nc);

%% First FFT along fast time
tmp = fftshift(fft(Data_time.*win),1);
%discard negtive frequency
Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);

%% Filtering out stationary targets across slow-time
Data_range_MTI = zeros(size(Data_range,1),nc);
% high-pass filter
[b,a] = butter(4, 0.0075, 'high');
for k=1:size(Data_range,1)
  Data_range_MTI(k,:) = filter(b,a,Data_range(k,:));
end

%% Plot the range profile of the data
if is_plot == 1
    figure(6);
    colormap(jet);
    imagesc(20*log10(abs(Data_range_MTI)));
    xlabel('Sweep Index');
    ylabel('Range Index');
    title('Range Profiles');
    clim = get(gca,'CLim'); axis xy; ylim([1 size(Data_range_MTI,1)]);
    set(gca, 'CLim', clim(2)+[-60,0]);
end