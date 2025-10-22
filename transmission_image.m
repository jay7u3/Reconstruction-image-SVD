function transmission_image
    % Ouverture de l'image
    [im, map] = imread('image.png');
    im = rgb2gray(im);
    
    % Choix de r (le rang de precision)
    r = 100;

    A = double(im);
    estimer_image2(A, r);
    %estimer_image1(A, r, 1e-9);
end


function imres = estimer_image2(A, r)
    [U, S, V] = svd(A);   % Décomposition en valeurs singulières
    
    imres = zeros(size(A));
    figure;
    
    for i = 1:r
        imres = imres + S(i,i) * U(:,i) * V(:,i)';  % Ajout du i-ème terme
        image(imres);
        colormap gray;
        title(['Approximation de rang ', num2str(i)]);
        pause(0.3);
    end
end


function imres = estimer_image1(A, r, tolerance)
    B1 = A' * A;
    B2 = A * A';
    imres = zeros(size(A));
    
    for i = 1:r
        [V, lambda1] = methode2(B1, tolerance);
        [U, lambda2] = methode2(B2, tolerance);
        
        % Normalisation
        U = U / norm(U);
        V = V / norm(V);
        
        % Valeur singulière moyenne
        lambda = (lambda1 + lambda2) / 2;
        sigma = sqrt(abs(lambda));
        
        % Mise à jour de l'image estimée
        imres = imres + sigma * (U * V');
        
        % Déflation des matrices
        B1 = B1 - lambda1 * (V * V') / (norm(V)^2);
        B2 = B2 - lambda2 * (U * U') / (norm(U)^2);
        
        % Affichage progressif (facultatif)
        imagesc(imres);
        colormap gray;
        title(['Approximation de rang ', num2str(i)]);
        pause(0.1);
    end
end

function erreur = calculer_erreur_image(A1, A2)
    erreur = norm(A1-A2, 2);
end

% Methode de la deflation
function [X, lambda] = deflation(A, erreur_max)
    H = size(A, 2);
    X = rand(H, 1);
    lambda_old = 0;
    lambda = 1;

    while abs(lambda - lambda_old) > erreur_max
        lambda_old = lambda;
        XTEMP = A * X;
        X = XTEMP / norm(XTEMP, 2);
        lambda = (X' * A * X) / (X' * X);
    end
end