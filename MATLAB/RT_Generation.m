function Data_range_MTI = RT_Generation(Data,NTS,nc)

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
