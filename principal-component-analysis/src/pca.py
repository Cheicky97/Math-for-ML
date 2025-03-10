# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7  2025

@author: cheic
"""
import numpy as np

class PCA:
    """
    This class provides methods for Principal Components Analysis
    """
    def __init__(self, X:np.ndarray, normalized:bool=False):
        """
        Parameters:
        X:  numpy.ndarray
            (N, D) dataset, N is the number of sample and D the dimension of the dataset
            i.e. the number of the features.
            Note : X must contains only numerical values.
        normalized: bool
            if True, you indicate that X is already normalized (z-score)
            The default is False.
        """
        # checking if inputs types match our expectation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be an np.ndarray")
        elif not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        
        self.X=X
        self.N, self.D = X.shape
        self.mean = np.mean(X, axis=0) # the mean values of data features
        # Z-score normalization of the data set X
        # i.e. data centered and variance in the interval [0, 1]
        self.X_normalized = (self.X - self.mean) / np.std(X, axis=0)
        # From now we will work with the normalized dataset
        pass

    def cov(self)-> np.ndarray:
        """
        Computes and returns the covariance matrix.
        """
        return (self.X_normalized.T @ self.X_normalized) / self.N

    def projection_matrix(self, B:np.ndarray, orthogonal:bool=True) -> np.ndarray:
        """
        Calculate the projection matrix i.e. operator used to project a D-dimensional vectors space
        onto a the subspace U spanned by b1,---,bM, columns vector of B.
        ----------
        Parameters:
        
        B: numpy.array
            transformation matrix to pass from the subspace U to the original scpace 
            B = [b1|...|bM]

            in one 1D B is just a vector.
        orthogonal: boolean, optional
            If the input matrix (or input vector) is orthogonal i.e. orthonormal column vectors,
            this parameter must be setted to True.
            The default is True.
        ----------
        return:
            np.array of the projection matrix
        """
        if orthogonal: 
            return B @ B.T
        else:
            return B @ np.linalg.inv(B.T @ B) @ B.T
        
    
    def eigen(self, Matrix:np.ndarray)-> tuple:
        """
        diagonalize Matrix and sort eigen values and vectors by ascending
        value of eigen values
        ----------
        Parameters:
        Matrix:np.ndarray
            The matrix to be diagonalized.
        ----------
        return
            (eig_vals, eig_vecs)
            with eig_vals the np.ndarray of eigen values 
            and eig_vecs the matrix consisting of eigen vectors as columns
        """
        # diagonalisation
        eig_vals, eig_vecs = np.linalg.eig(Matrix)
        #sorting
        sorted_indx = np.argsort(eig_vals)[::-1]
        return eig_vals[sorted_indx], eig_vecs[:,sorted_indx]

    def normalize_vec(self, vec):
        """
        normalize a set of vectors
        """
        return vec / np.linalg.norm(vec, axis=0)

    def pca(self, nb_components:int) -> tuple:
        """
        Perform pca and select a number of principal components, i.e.
        select nb_components axis where the variance are the highest.  
        ---------
        Parameters:
        nb_components: integer
            number of principal components to be selected.
        """
        # computation of the covariance matrix S
        S = self.cov()
        # computation of the eigen values and vectors of S
        eig_vals, eig_vecs = self.eigen(S)
        # construction of the principal subspace regarding nb_components
        principal_vals, principal_components = (eig_vals[:nb_components],
                                                 self.normalize_vec(eig_vecs[:,:nb_components])
                                                 )
        # projection of data point onto this subspace
        X_reconst = self.X_normalized @ projection_matrix(principal_components) + self.mean # type: ignore
        return X_reconst, principal_vals, principal_components 

    def hd_pca(self, nb_compoents:int) -> tuple:
        """
        Perform high dimensional PCA, suitable when N << D so that the computation of the
        covariance matrix and subsequently its diagonalisation become too much heavy.
        -----------
        Parameters:
        nb_components: integer
            number of principal components to be selected.     
        """
        # first we compute X.T @ X matrix we denote by M
        M = self.X_normalized @ self.X_normalized.T # NxN matrix
        # then, diagonalized it (eig_vals should be same as for the covariance matrix)
        eig_vals, vecs = self.eigen(M) # denote vecs the eig. vec. of M (eig_vecs reserved for S)
        # compute eigen_vectors of the covariance matrix and select only first D vectors from it
        # and finally normalized them
        eig_vecs = self.normalize_vec((self.X_normalized.T @ vecs)[:,:self.D])
        # construct the principal subspace
        principal_vals, principal_components = eig_vals[:nb_compoents], eig_vecs[:,:nb_compoents]
        # project normalized data points onto this subspace
        X_reconst = (self.X_normalized @ self.projection_matrix(principal_components)) + self.mean
        return X_reconst, principal_vals, principal_components

if __name__ == "__main__":
    pass