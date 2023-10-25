import psycopg2 as pg
import numpy as np

'''
PREFIXES
o - Instance of a class/object
s - String
n - Integer
f - Floating point number
m - Matrix
cls - Class names
dt - Date
d - Dictionary
'''

oConn = pg.connect("dbname=USA user=postgres password=SARS-CoV-2")
oCur = oConn.cursor()
oCur.execute(''' SELECT dt_year, cast(am_birthweight as int), cast(am_gestation as int) FROM "USA"."natalityConf" limit 100 ''')
rows = oCur.fetchall()

n = 2
y = [[x[0],x[1],x[2]] for x in rows]
z = np.asarray(y)
oConn.close()

print('end')



