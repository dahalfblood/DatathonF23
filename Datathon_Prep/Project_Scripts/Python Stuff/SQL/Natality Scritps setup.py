import psycopg2 as pg
import numpy as np
'''
Prefixes

o - instance of a class/ object
s - string
n - integer
m - matrix
cls - class names
dt- date
d - dictionairy 
'''

oConn = pg.connect("dbname=USA user=postgres password=BigBrain69420!")
oCur = oConn.cursor()
x = oCur.execute('''SELECT dt_year, id_state, id_area, id_cert, in_resident,id, state, county FROM "USA"."natalityConf" limit  10 ''')

rows = oCur.fetchall()

n = 2
y = [int(x[n]) for x in rows]
z = np.asarray(y)
oConn.close()

print('end')