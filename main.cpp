#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <opencv2/opencv.hpp>

#define ld long double

using namespace std;
using namespace cv;

// -------- Génération d'une matrice aléatoire --------
vector<vector<ld>> creer_matrice_aleatoire(int n, int m)
{
    vector<vector<ld>> M(n, vector<ld>(m));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            M[i][j] = (ld)rand() / RAND_MAX; // valeur entre 0 et 1
        }
    }
    return M;
}

// -------- Génération d'un vecteur aléatoire --------
vector<ld> creer_vecteur_aleatoire(int n)
{
    vector<ld> v(n);
    for (int i = 0; i < n; i++)
        v[i] = (ld)rand() / RAND_MAX;
    return v;
}

// -------- Affiche une matrice --------
void afficher_matrice(vector<vector<ld>>& A)
{
    for (int l = 0; l < A.size(); l++)
    {
        for (int c = 0; c < A[0].size(); c++)
        {
            cout << A[l][c] << ' ';
        }
        cout << '\n';
    }
}

// -------- Affiche un vecteur--------
void afficher_vecteur(vector<ld>& X)
{
    for (int i = 0; i < X.size(); i++)
    {
        cout << X[i] << ' ';
    }
    cout << '\n';
}

// -------- Produit matrice-vecteur --------
vector<ld> produit_matrice_vecteur(const vector<vector<ld>>& A, const vector<ld>& X)
{
    int n = A.size();
    vector<ld> result(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < X.size(); j++)
            result[i] += A[i][j] * X[j];
    }
    return result;
}

// -------- Norme 2 d'un vecteur --------
ld norme2(const vector<ld>& v)
{
    ld somme = 0.0;
    for (ld x : v)
        somme += x * x;
    return sqrt(somme);
}

// -------- Produit scalaire de deux vecteurs --------
ld produit_scalaire(const vector<ld>& X, const vector<ld>& Y)
{
    ld somme = 0.0;
    for (int i = 0; i < X.size(); i++)
        somme += X[i] * Y[i];
    return somme;
}

// -------- Produit matriciel complet --------
vector<vector<ld>> produit_matrice(const vector<vector<ld>>& A, const vector<vector<ld>>& B)
{
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<ld>> C(n, vector<ld>(m, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < p; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// -------- Transposition d'une matrice --------
vector<vector<ld>> transposee(const vector<vector<ld>>& A)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> T(m, vector<ld>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            T[j][i] = A[i][j];
    return T;
}

// -------- Soustraction de matrices : A - B --------
vector<vector<ld>> soustraction_matrice(const vector<vector<ld>>& A, const vector<vector<ld>>& B)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> C(n, vector<ld>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// -------- Multiplication d'une matrice par un scalaire --------
vector<vector<ld>> matrice_scalaire(const vector<vector<ld>>& A, ld s)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> C(n, vector<ld>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = s * A[i][j];
    return C;
}

// -------- Produit vecteur * vecteur' (matrice) --------
vector<vector<ld>> produit_vecteur_transpose(const vector<ld>& U, const vector<ld>& V)
{
    int n = U.size();
    int m = V.size();
    vector<vector<ld>> C(n, vector<ld>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = U[i] * V[j];
    return C;
}

// -------- Erreur relative entre deux matrices --------
ld erreur_relative(const vector<vector<ld>>& A, const vector<vector<ld>>& B)
{
    int n = A.size();
    int m = A[0].size();
    ld somme_diff2 = 0.0;
    ld somme_orig2 = 0.0;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            ld diff = A[i][j] - B[i][j];
            somme_diff2 += diff*diff;
            somme_orig2 += A[i][j]*A[i][j];
        }
    }
    return sqrt(somme_diff2 / somme_orig2); // norme relative de Frobenius
}


// -------- Méthode de la puissance itérée --------
pair<vector<ld>, ld> puissance_iteree(const vector<vector<ld>>& A, ld erreur_max)
{
    int n = A.size();
    vector<ld> X = creer_vecteur_aleatoire(n);

    ld lambda = 1.0;
    ld lambda_ancien = 0.0;

    while (abs(lambda - lambda_ancien) > erreur_max)
    {
        lambda_ancien = lambda;

        vector<ld> XTEMP = produit_matrice_vecteur(A, X);

        // Normalisation
        ld norm = norme2(XTEMP);
        for (int i = 0; i < n; i++)
            X[i] = XTEMP[i] / norm;

        lambda = produit_scalaire(X, produit_matrice_vecteur(A, X)) / produit_scalaire(X, X);
    }

    return {X, lambda};
}

// -------- Conversion OpenCV -> Matrice --------
vector<vector<ld>> image_to_matrix(const Mat& img)
{
    vector<vector<ld>> M(img.rows, vector<ld>(img.cols));
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            M[i][j] = img.at<uchar>(i,j) / 255.0; // normalisé [0,1]
    return M;
}

// -------- Conversion Matrice -> OpenCV --------
Mat matrix_to_image(const vector<vector<ld>>& M)
{
    int n = M.size(), m = M[0].size();
    Mat img(n, m, CV_8UC1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            ld val = M[i][j];
            if(val < 0) val = 0;
            if(val > 1) val = 1;
            img.at<uchar>(i,j) = static_cast<uchar>(val * 255);
        }
    return img;
}

// -------- Estimation de l'image par SVD approchée --------
vector<vector<ld>> estimer_image(const vector<vector<ld>>& A, int r, ld tolerance)
{
    int n = A.size();
    int m = A[0].size();
    vector<vector<ld>> imres(n, vector<ld>(m, 0.0));

    vector<vector<ld>> B1 = produit_matrice(transposee(A), A);
    vector<vector<ld>> B2 = produit_matrice(A, transposee(A));

    for (int i = 0; i < r; i++)
    {
        auto [V, lambda1] = puissance_iteree(B1, tolerance);
        auto [U, lambda2] = puissance_iteree(B2, tolerance);

        // Normalisation
        ld normU = norme2(U);
        ld normV = norme2(V);
        for (ld& x : U) x /= normU;
        for (ld& x : V) x /= normV;

        // Valeur singulière moyenne
        ld lambda = (lambda1 + lambda2) / 2.0;
        ld sigma = lambda>0 ? sqrt(lambda) : 0;

        // Mise à jour de l'image estimée
        vector<vector<ld>> UVt = produit_vecteur_transpose(U, V);
        vector<vector<ld>> UVt_scaled = matrice_scalaire(UVt, sigma);
        for (int ii = 0; ii < n; ii++)
            for (int jj = 0; jj < m; jj++)
                imres[ii][jj] += UVt_scaled[ii][jj];

        // Déflation
        vector<vector<ld>> deflationV = matrice_scalaire(produit_vecteur_transpose(V, V), lambda1 / (normV * normV));
        vector<vector<ld>> deflationU = matrice_scalaire(produit_vecteur_transpose(U, U), lambda2 / (normU * normU));
        B1 = soustraction_matrice(B1, deflationV);
        B2 = soustraction_matrice(B2, deflationU);

        // Affichage de l'image reconstruite
        Mat img_show = matrix_to_image(imres);
        cout << (i+1) << '/' << r << '\n';
        imshow("Reconstruction progressive", img_show);
        waitKey(100); // pause 100 ms
    }

    return imres;
}

int main()
{
    int r;
    ld tolerance;
    cout << "Entrer r (la precision comprise entre 0 et la taille de la plus petite dimension de l'image)" << '\n';
    cin >> r;
    cout << "Entrer la tolerance (par exemple 0.001)" << '\n';
    cin >> tolerance;

    Mat img = imread("image.png", IMREAD_GRAYSCALE);
    vector<vector<ld>> A = image_to_matrix(img);

    vector<vector<ld>> approximation = estimer_image(A, r, tolerance);

    // Calcul et affichage de l'erreur relative
    ld erreur = erreur_relative(A, approximation);
    cout << "Erreur relative entre l'image originale et reconstruite : " << erreur << endl;


    return 0;
}
