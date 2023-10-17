import numpy as np
import matplotlib.pyplot as plt
import psycopg2 as pg
import traceback

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
    print("We got you into the mainframe. It's up to you now. Godspeed")
    cur = conn.cursor()

    cur.execute('''Select cast(id_m_edu as int), cast(id_f_edu as int), cast(id_sex as int)  FROM "USA"."natalityConf" limit 1000''')
    rows = cur.fetchall()

    cur.close()
    conn.close()
    
    y = np.array([[x[0], x[1], x[2]] for x in rows])
    
   

    # Standardize the data
    X = y[:, :3]  # Extract features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    # Perform LDA
    nClass = len(np.unique(y[:, 2]))
    nFeat = X_scaled.shape[1]
    S_w = np.cov(X_scaled.T)
    S_b = np.zeros((nFeat, nFeat))
    for i in range(nClass):
        class_indices = np.where(y[:, 2] == i)[0]
        if len(class_indices) > 0:
            d_mean = np.mean(X_scaled[class_indices], axis=0) - X_mean
            S_b += len(class_indices) * np.outer(d_mean, d_mean)
    
    eva, eve = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
    
    # Filter out infinite and NaN values
    eva = eva[np.isfinite(eva)]
    eve = eve[:, np.isfinite(eva)]
    
    idxs = np.argsort(eva)[::-1]
    eve = eve[:, idxs]
    W = eve[:,nClass - 1]

    x = X @ eve[:,0]
    z = X @ eve[:,1]  
    class_labels = y[:, 2].astype(int)
    # Plot the data in the LDA space
    plt.scatter(x, z, c=class_labels, cmap='viridis')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA on Data')
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.show()

except (Exception, pg.Error) as error:
    traceback.print_exc()
    print("Error:", error)
