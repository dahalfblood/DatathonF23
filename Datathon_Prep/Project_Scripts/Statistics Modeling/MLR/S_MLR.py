import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import psycopg2 as pg


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

    oCur.close()
    oConn.close()
    
    # Extract the relevant columns from the 'rows' variable
    X = [
        [row[0], row[2], row[4]] 
        for row in rows
        ]  
    Y = [row[1] for row in rows]
    
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    Y = Y.reshape(-1, 1)


    def lin_reg(X,Y):
        #Mean of X & Y
        mX = np.mean(X, axis = 1, keepdims = True)
        mY = np.mean(Y)
        #Deviation of X&Y
        sig_X = X - mX
        sig_Y = Y - mY
        #slope from 
        slope = np.sum(sig_X * sig_Y) / np.sum(sig_X ** 2)
        intercept = mY - slope * mX
        
        return slope, intercept

    slope, intercept = lin_reg(X,Y)


    print("SLOPE:", slope)
    print("INTERCEPT:", intercept)
    
    '''need to get the graph established somehow. idk how at this point'''
    '''dimensionality issue bc X= (MxN) and Y = (Mx1)l'''
    
    for i in range(X.shape[1]):
        x_var = X[:,i]
    

    # Create a scatter plot of the data points
    plt.scatter(x_var, Y, label='Data Points', color='blue')

    # Create the linear regression line
    regression_line = slope * X + intercept

    # Plot the regression line
    plt.plot(x_var, regression_line, color='red', label='Linear Regression')

    # Add labels and a legend
    plt.xlabel('X (Independent Variable)')
    plt.ylabel('Y (Dependent Variable)')
    plt.legend()

    # Display the plot
    plt.show()

except (Exception, pg.Error) as error:
    print("Error, computer go boom now", error)