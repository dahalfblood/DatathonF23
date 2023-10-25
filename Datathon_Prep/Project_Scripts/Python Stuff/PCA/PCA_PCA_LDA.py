import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import psycopg2 as pg
import pandas as pd

def perform_pca(X, num_components):
    # Standardize X
    N_x = X - X.mean(axis=0)

    # Covariance of X
    covX = np.cov(N_x.T)

    # Eigenvalues and eigenvectors of the covariance matrix
    eva, eve = LA.eig(covX)

    # Sort eigenvalues and eigenvectors in descending order
    ind = eva.argsort()[::-1]
    eva = eva[ind]
    eve = eve[:, ind]

    # Keep the top 'num_components' principal components
    W_pca = eve[:, :num_components]

    # Project data onto the selected principal components
    X_pca = X @ W_pca

    return X_pca

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

    idxs = np.argsort(eva)[::-1]
    eve = eve[:, idxs]
    W_lda = eve[:, :nClass - 1]

    X_lda = X @ W_lda

    return X_lda
db_params = {
    'dbname': 'USA',
    'user': 'postgres',
    'password': 'BigBrain69420!',
    'host': 'localhost',
    'port': '5432'
}

try:
    oConn = pg.connect(**db_params)

    print("Connected to the database. It's up to you now. Godspeed")

    oCur = oConn.cursor()

    # Execute the SQL query
    oCur.execute('''SELECT id_m_edu, id_f_edu, am_post_full_con, am_lunar_month_dob, id_sex FROM "USA"."natalityConf" limit  100000 ''')

    # Fetch all rows as a list of tuples
    rows = oCur.fetchall()

    # Get the column names from the cursor description
    col_names = [desc[0] for desc in oCur.description]

    oCur.close()
    oConn.close()

    # Convert the rows to a Pandas DataFrame
    df = pd.DataFrame(rows, columns=col_names)
    y = np.array([[x[0], x[1], x[2]] for x in rows])
    

    # Convert non-numeric columns to numeric data type
    df[['id_m_edu', 'id_f_edu',]] = df[['id_m_edu', 'id_f_edu', ]].apply(pd.to_numeric)

    # Assuming 'am_lunar_month_dob' is one of the numeric columns in X_numeric
    X_numeric = df[['am_lunar_month_dob', 'id_m_edu', 'id_f_edu', 'am_post_full_con']].values

    # ------------------- First PCA Happens Here -------------------
    X_pca_1 = perform_pca(X_numeric, num_components=2)

    # ------------------- Second PCA Happens Here -------------------
    X_pca_2 = perform_pca(X_pca_1, num_components=2)
    
    X_pca_3 = perform_pca(X_pca_2, num_components=2)
    
    
    X_lda = perform_lda(X_pca_3,y)
    #X_lda = perform_lda(X_pca_2,y)

    # Plot the data in the second PCA space
    plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], cmap = 'viridis')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('PCA+PCA+LDA')
    plt.grid(True)
    plt.show()

except (Exception, pg.Error) as error:
    print("Error, computer go boom now", error)
