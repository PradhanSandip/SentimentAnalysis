from ProcessData import ProcessData
d = ProcessData("reviews.csv")
print(d.get(0)["review"])
