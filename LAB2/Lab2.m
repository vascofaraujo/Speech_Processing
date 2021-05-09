clc;
close all;
clearvars;

%% Part 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 1


[y,Fs] = audioread('vowels_90817.wav');


%soundsc(y,Fs);

%plot the audio signal
Ts = 1/Fs;
t = 0:Ts:(length(y)*Ts)-Ts;

figure;
plot(t,y);
title("Audio signal vowels 90817.wav");

%We can see that the first vowel can be found 
% between 0.8 and 1.8s, so we will extract that
start = 0.8*Fs;
finish = 1.8*Fs;
y2 = y(start:start+finish-1);

%Confirm we isolated a single vowel
t2 = 0:Ts:(length(y2)*Ts)-Ts;
figure;
plot(t2,y2);
title("Isolated vowel from vowels 90817.wav");
%soundsc(y2, Fs);

%Autocorrelate function
R = xcorr(y2);

figure;
plot(R);
title("Autocorrelation function");

[M,I] = max(R);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 2

[y, Fs] = audioread("birthdate_90817.wav");

file_f0 = fopen("birthdate_90817.myf0","w");

time = length(y);
interval = 0.01*Fs; 
size = 0.02*Fs;

loop_time = time/interval;

for i = 1:loop_time-1
    y2 = y( (i-1)*interval+1 : (i+1)*interval );
    
    %If you want to check each window uncomment next lines
    %figure;
    %plot(y2);
    %w = waitforbuttonpress;
    
    R = xcorr(y2);

    %normalize R
    [M,I] = max(R);
    R_norm = R/M;
    
    
    %Finds max value of correlation
    [M1,I1] = max(R_norm(size+round(Fs/400):size+round(Fs/60)));
    % % Obtaining the fundamental period
    F0(i) = Fs/(I1 + round(Fs/400));     
    
    if M1 <= 0.3 
        F0(i) = 0;
    end
    
    %Writes F0 to file
    fprintf(file_f0,'%f\n',F0(i));   
end

figure;
plot(F0);
title("F0");

%Calculate average without 0 values
sum = 0;
num = 0;
for i = 1:length(F0)
    if F0(i) ~= 0
        sum = sum + F0(i);
        num = num+1;
    end
end
averageF0 = sum/num;


%% Part 2

%Question 1

[y,Fs] = audioread("birthdate_90817.wav");

%These values are for the first question. In order to change the results,
%simply change the values for prediction and/or window
prediction = 16;
window = 0.02;

time = window*Fs;
interval = (window/2)*Fs;

ak = v_lpcauto(y, prediction, [interval, time, 0]);
zi=zeros([1 prediction]);

for i=1:length(ak)

    [res((i-1)*interval+1:(i)*interval), zf] = filter(ak(i,:), 1, y((i-1)*interval+1:(i)*interval), zi);
    zi = zf;   
end

figure;
plot(res);
title("Residual");

%soundsc(res, Fs);
audiowrite('birthdate_90817_res.wav',res,Fs);


%Now to recosntruct the signal
zi=zeros([1 prediction]);

for i = 1:length(ak)
    [y_reconstuct((i-1)*interval+1:(i)*interval), zf] = filter(1, ak(i,:), res((i-1)*interval+1:(i)*interval), zi);
    zi = zf;   
end


%soundsc(y_reconstuct, Fs);
audiowrite('birthdate_90817_syn.wav',y_reconstuct,Fs);















