import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import psycopg2 as pg
import pandas as pd

db_params = {
    'dbname': 'USA',
    'user': 'postgres',
    'password': 'BigBrain69420!',
    'host': 'localhost',
    'port': '5432'
}

try:
    oConn = pg.connect(**db_params)

    print("We got you into the mainframe. It's up to you now. Godspeed")

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

    # Convert non-numeric columns to numeric data type
    df[['id_m_edu', 'id_f_edu',]] = df[['id_m_edu', 'id_f_edu', ]].apply(pd.to_numeric)

    # Assuming 'am_lunar_month_dob' is one of the numeric columns in X_numeric
    X_numeric = df[['am_lunar_month_dob', 'id_m_edu', 'id_f_edu','am_post_full_con']]

    # Standardize X
    N_x = X_numeric - X_numeric.mean(axis=0)

    # Covariance of X
    covX = np.cov(N_x.T)

    # eval/evec of our covariance matrix
    eva, eve = LA.eig(covX)

    # indices
    ind = eva.argsort()[::-1]
    eva = eva[ind]
    eve = eve[:, ind]

    # Separate 'X' and 'Z' data points using boolean masks
    is_X = (df['id_sex'] == '1')    
    is_Z = (df['id_sex'] == '2')
    X_data = X_numeric[is_X]
    Z_data = X_numeric[is_Z]


    # Ensure arrays are at least 2-dimensional for matrix multiplication
    X_data = X_data.values.reshape(-1, X_data.shape[1])
    Z_data = Z_data.values.reshape(-1, Z_data.shape[1])

    # Calculate the corresponding 'x', 'y', and 'z' coordinates for 'X' and 'Z' data points
    x_X = X_data @ eve[:, 0].reshape(-1, 1)
    y_X = X_data @ eve[:, 1].reshape(-1, 1)
    z_X = X_data @ eve[:, 2].reshape(-1, 1)

    x_Z = Z_data @ eve[:, 0].reshape(-1, 1)
    y_Z = Z_data @ eve[:, 1].reshape(-1, 1)
    z_Z = Z_data @ eve[:, 2].reshape(-1, 1)


    fig = plt.figure(1)
    plt.scatter(x_X, y_X, label='X', marker='d')
    plt.scatter(x_Z, y_Z, label='Z', marker='X')
    plt.xlim(-250, 250)  # Set appropriate x-axis limits
    plt.ylim(-250, 250)  # Set appropriate y-axis limits
    plt.show()

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_X, y_X, z_X, label='X', marker='d')
    ax.scatter(x_Z, y_Z, z_Z, label='Z', marker='X')
    ax.set_xlim(-250, 250)  # Set appropriate x-axis limits
    ax.set_ylim(-250, 250)  # Set appropriate y-axis limits
    ax.set_zlim(-250, 250)  # Set appropriate z-axis limits
    plt.show()

except (Exception, pg.Error) as error:
    print("Error, computer go boom now", error)
