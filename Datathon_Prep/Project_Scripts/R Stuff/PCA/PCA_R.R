library(odbc)

con <- dbConnect(odbc(),
                 Server = "PostegreSQL 15",
                 Database = "usa",
                 UID = "postgres",
                 PWD = rstudioapi::askForPassword("BigBrain69420!"),
                 port = 5432)


