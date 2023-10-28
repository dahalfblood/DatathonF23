f = open('LEA_Characteristics.csv', "r", encoding="cp1252")
counter = 0
for x in f:
    counter = counter + 1
print(counter)