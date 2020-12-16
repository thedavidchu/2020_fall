% ECE367 - PS4 - Q6

% ============================== PART A ============================== %

% Define y = Ax
% Define n = h*w
h = 50;
w = 50;
n = h * w;
m = 1950;

A = zeros(m, n);
M = zeros(h, w);

% Construct A matrix
for i = 1:n
    M(i) = 1;
    % The fact that M is an h x w MATRIX should be made clear.
    % The instructions make it out to be an (h*w) VECTOR!
    A(:, i) = scanImage(M);
    M(i) = 0;
end

% Plot A
hold on;
imshow(A, []);
title("Plot of Matrix A");
hold off;

% ============================== PART B ============================== %

% Get scanned beam info
y = scanImage;
[U, S, V] = svd(A);

% Plot S
s = diag(S);
hold on
figure;
plot(log10(s))
title("Log Plot of Singular Values in Decreasing Order")
hold off;

% Use r = 1200
r = 1200;

% Reduce dimension of U, S, V to (m x r), (r x r), (n x r)
U_r = U(:, 1:r);
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

% Multiply matrices
A_r = U_r * S_r * V_r.';

% Solve for x with Moore-Penrose Pseudoinverse, filtering out below 0.1
x = pinv(A_r, 0.1) * y;

% Reshape x
M_un = reshape(x, 50, 50);

% Plot M_un

hold on;
figure;
image(M_un);
title("Message M_u_n");
hold off;

% Message
% "Winter is coming". Very funny Stark.

% Dear Professor Draper,
% Fun fact, I sincerely dislike Matlab. Everything about it is just
% horrible. Okay, fine. It runs really quickly compared to NumPy, but it is
% really rather opaque.
% David
