function [CFAR_2D_out] = CA_CFAR_2D_fast(spectro,CFAR_winv,CFAR_wingv,CFAR_winh,CFAR_wingh,pfa,is_plot)
% CFAR_winh = 7;
% CFAR_winv = 7;
% CFAR_wingh = 2;
% CFAR_wingv = 2;
% pfa = 1e-5;
% offset = 5;
CFAR_2D_out = zeros(size(spectro));

Num_all_cells = (2*(CFAR_winv + CFAR_wingv)+1)*(2*(CFAR_winh + CFAR_wingh)+1);
Num_guard_cells = (2*(CFAR_wingv)+1)*(2*(CFAR_wingh)+1);
Num_training_cells = Num_all_cells - Num_guard_cells;

threshold_factor = (Num_training_cells * (pfa^(-1/Num_training_cells)-1));
%threshold_factor = 10^(threshold_factor/20)
%threshold_factor = offset;
conv_window = ones(2*(CFAR_winv + CFAR_wingv)+1,2*(CFAR_winh + CFAR_wingh)+1);
conv_window(CFAR_winv+1: CFAR_winv + 2*CFAR_wingv + 1,CFAR_winh+1: CFAR_winh + 2*CFAR_wingh + 1) = 0;
total_noise=conv2(spectro,conv_window,'same');
Noise_power = 1/(Num_training_cells) * total_noise;
Detection_threshold = threshold_factor * Noise_power;
CFAR_2D_out((spectro > Detection_threshold) & (spectro > 0))=1;

if is_plot == 1
    figure(8)
    mesh(1:size(CFAR_2D_out,2),1:size(CFAR_2D_out,1),CFAR_2D_out);
    view([0,90])
    xlabel('Time[s]', 'FontSize',16);
    ylabel('Velocity [m/s]','FontSize',16);
end