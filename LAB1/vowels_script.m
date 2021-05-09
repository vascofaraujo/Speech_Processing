clc;
close all;
clearvars;

F1 = [295.5334442512733  , 416.56824843029017 , 536.7368723904365 , 761.9499953116365 , 587.3401501989579 , 317.8215113401321 , 627.8411104750736 , 506.30531378265385 , 338.1472744500272 ];
    
F2 = [2298.843381605319  , 2038.7410873681358 , 1889.830895432693 , 1329.854654578722 , 1496.7722307282454 , 1752.0635667679228 , 1107.3312753542532 , 976.660067982939 , 754.3243029090357 ];

triangle_side1X = [F1(1), F1(9)];
triangle_side1Y = [F2(1), F2(9)];

triangle_side2X = [F1(9), F1(4)];
triangle_side2Y = [F2(9), F2(4)];

triangle_side3X = [F1(1), F1(4)];
triangle_side3Y = [F2(1), F2(4)];

figure;
hold all;
for i = 1:size(F1,2)
    scatter(F1(i), F2(i));
end

plot(triangle_side1X, triangle_side1Y, "k");
plot(triangle_side2X, triangle_side2Y, "k");
plot(triangle_side3X, triangle_side3Y, "k");

legend('i','e','E','a','6','@','O','o','u');

title("Vowel Triangle of Portuguese vowels")
xlabel('F1')
ylabel('F2')




