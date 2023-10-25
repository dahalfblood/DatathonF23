import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import psycopg2 as pg
import traceback

def perform_lda(X, y):
    nClass = len(np.unique(y))
    nFeat = X.shape[1]
    S_w = np.cov(X.T)
    S_b = np.zeros((nFeat, nFeat))
    X_mean = np.mean(X, axis=0)

    for i in range(nClass):
        class_indices = np.where(y == i)[0]
        if len(class_indices) > 0:
            d_mean = np.mean(X[class_indices], axis=0) - X_mean
            S_b += len(class_indices) * np.outer(d_mean, d_mean)

    eva, eve = np.linalg.eig(np.linalg.inv(S_w) @ S_b)

    eva = eva[np.isfinite(eva)]
    eve = eve[:, np.isfinite(eva)]

    idxs = eva.argsort()[::-1]
    eve = eve[:, idxs]
    W_lda = eve[:, :nClass - 1]

    X_lda = X @ W_lda

    return X_lda

# Database connection parameters
db_params = {
    'dbname': 'USA',
    'user': 'postgres',
    'password': 'BigBrain69420!',
    'host': 'localhost',
    'port': '5432'
}

# Load data from the database
try:
    conn = pg.connect(**db_params)
    print("Connected to the database. It's up to you now. Godspeed")
    cur = conn.cursor()

    cur.execute('''Select cast(id_m_edu as int), cast(id_f_edu as int), cast(id_sex as int)  FROM "USA"."natalityConf" limit 10000''')
    rows = cur.fetchall()

    cur.close()
    conn.close()

    y = np.array([x[2] for x in rows])
    X = np.array([[x[0], x[1]] for x in rows])

    # Standardize the data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    # ------------------- LDA Happens Here -------------------
    X_lda = perform_lda(X_scaled, y)

    # ------------------- PCA Happens Here -------------------
    # Calculate the covariance matrix of LDA-transformed data
    covX_lda = np.cov(X_lda.T)

    # Ensure that the covariance matrix is 2D
    covX_lda_re = covX_lda.reshape(2,2)

    # Eigenvalues and eigenvectors of the covariance matrix
    eva_pca, eve_pca = np.linalg.eig(covX_lda_re)

    # Sort eigenvalues and eigenvectors in descending order
    idxs_pca = eva_pca.argsort()[::-1]
    eve_pca = eve_pca[:, idxs_pca]
    W_pca = eve_pca[:, :2]  # Assuming you want to keep the top 2 principal components

    X_pca = X_lda @ W_pca

    # Plot the data in the PCA space
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('LDA + PCA on Data')
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.show()

except (Exception, pg.Error) as error:
    traceback.print_exc()
    print("Error:", error)