clc;
close all;
clearvars;

[y, Fs] = audioread("vowels_90817.wav");

zero = 0;
for i = 1:size(y,1)
    if y(i) == 0
        zero = zero+1;
    end
    if zero == 12
        break
    end
end
i
x = y(1:i);
r = xcorr(x);

r_norm = r/max(r)

figure;
plot(r_norm);

