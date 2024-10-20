% Load symbolic package
pkg load symbolic;

% Define symbolic variable x
syms x;

% Define the function f(x)
f = log(1-x);

% Compute the derivative of f(x)
df = diff(f, x);

% Display the derivative
disp("The derivative of log(x) is:");
pretty(df);  % Displays the derivative in a readable format

g = (e.^-x)/(1+e.^-x).^2

dg = diff(g, x);
dgg = simplify(dg)
pretty(dgg);
