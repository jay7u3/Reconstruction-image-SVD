# Reconstruction-image-SVD

Ce projet illustre la reconstruction progressive d'une image à l’aide de la décomposition en valeurs singulières (SVD).  
Deux implémentations sont proposées : une en MATLAB et une en C++ avec OpenCV.

---

## Principe

Une image est représentée par une matrice A contenant les intensités des pixels.  
La décomposition en valeurs singulières permet d’écrire :
A = U Σ Vᵀ  
où U et V sont des matrices orthogonales et Σ contient les valeurs singulières décroissantes.

En ne gardant que les r premières valeurs singulières, on obtient une approximation de rang r :
Aᵣ = Σ (σᵢ uᵢ vᵢᵀ)

Cela permet de reconstruire progressivement l’image en ajoutant chaque composante une à une.

---

## Partie MATLAB

La version MATLAB propose deux approches :
- Une version simple utilisant la fonction svd().
- Une version manuelle utilisant la méthode des puissances itérée pour calculer les vecteurs propres.

L’image est affichée à chaque itération, montrant l’amélioration visuelle au fur et à mesure que le rang augmente.

---

## Partie C++

La version C++ utilise OpenCV pour lire l’image et afficher la reconstruction en temps réel. Dans cette version, la méthode des puissances itérée est également utilisé.

---

## Résultats

- L’image floue au début (rang faible)
- L’image de plus en plus nette (rang élevé)


![](gifs/reconstruction2.gif)


![](gifs/reconstruction1.gif)


---

## Intérêt pratique

- **Compression** : stocker uniquement les r plus grandes valeurs singulières permet de réduire fortement la taille des données.
- **Réduction de bruit** : les valeurs singulières faibles contiennent souvent le bruit de l’image.
- **Visualisation pédagogique** : permet de comprendre comment la SVD reconstruit une image à partir de composantes fondamentales.

---

Projet pédagogique combinant théorie des matrices et traitement d’images, pour montrer concrètement le pouvoir de la décomposition en valeurs singulières.
