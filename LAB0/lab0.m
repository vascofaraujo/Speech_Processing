clc;
close all;
clearvars;



[x, Fs] = audioread("anos.wav");

delay = 0.001*Fs
lag = 10

y = zeros(length(x));

lag = round(delay*Fs)

y(1:lag) = x(1:lag)

for i = lag+1:length(y)
    y(i) = x(i) + gain * y(i-lag)
end



audiowrite("anos_new.wav", y, Fs)
