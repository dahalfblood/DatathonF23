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
    oCur.execute('''SELECT id_m_edu, id_f_edu, am_post_full_con, am_lunar_month_dob, id_sex FROM "USA"."natalityConf" limit  1000 ''')

    # Fetch all rows as a list of tuples
    rows = oCur.fetchall()

    # Get the column names from the cursor description
    col_names = [desc[0] for desc in oCur.description]

    oCur.close()
    oConn.close()

    # Convert the rows to a Pandas DataFrame
    df = pd.DataFrame(rows, columns=col_names)

    # Convert non-numeric columns to numeric data type
    df[['id_m_edu', 'id_f_edu',]] = df[['id_m_edu', 'id_f_edu', ]].apply(pd.to_numeric)

    # Assuming 'am_lunar_month_dob' is one of the numeric columns in X_numeric
    X_numeric = df[['am_lunar_month_dob', 'id_m_edu', 'id_f_edu', 'am_post_full_con']].values

    # ------------------- First PCA Happens Here -------------------
    X_pca_1 = perform_pca(X_numeric, num_components=3)

    # ------------------- Second PCA Happens Here -------------------
    X_pca_2 = perform_pca(X_pca_1, num_components=3)

    # Plot the data in the second PCA space
    plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Twice on Data')
    plt.grid(True)
    plt.show()

except (Exception, pg.Error) as error:
    print("Error, computer go boom now", error)
