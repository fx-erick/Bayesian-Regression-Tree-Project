options(java.parameters = "-Xmx2500m")
library("bartMachine")
set_bart_machine_num_cores(2)

db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")

X = data.frame(db[,2:11])
y = db$y
bart_machine <- bartMachine(X, y)
bart_machine

k_fold_cv(X, y, k_folds = 10)
rmse_by_num_trees(bart_machine, num_replicates = 20)

bart_machine_cv <- bartMachineCV(X, y)
k_fold_cv(X, y, k_folds = 10, k = 3, nu = 3, q = 0.99, num_trees = 200)
yhat = predict(bart_machine_cv, X)
check_bart_error_assumptions(bart_machine_cv)