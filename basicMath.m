clear;
clc;
disp("basta");
pkg load symbolic

% Update the range of x1 and x2 to go from -100 to 100
x1 = -100:.1:100;
x2 = -100:.1:100;

w1 = 2;
w2 = 3;
b = 6;

z = w1 * x1 + w2 + x2 + b;

a = 1 ./ (1 + e .^ -z);  % Elementwise division

% Plot the function with the updated x1 range
% figure;             % Opens a new figure window
% plot(z, a); 
% title('Plot of function a against x1');
% xlabel('z');
% ylabel('a');        
% xlim([-10 10]);   
% grid on;            


% L = @(y) y*log(a)+(1-y)*log(1-a);

% sum(L([1:5]))
pause;              

