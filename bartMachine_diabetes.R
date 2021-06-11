library("bartMachine")
library("bart")

options(java.parameters = "-Xmx2500m")
set_bart_machine_num_cores(4)

db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")
load("bart_machine.RData")

#' 75% of the sample size
smp_size <- floor(0.75 * nrow(db))

#' set the seed to make your partition reproducible
set.seed(123)

train_ind <- sample(seq_len(nrow(db)), size = smp_size)

train <- db[train_ind, ]
test <- db[-train_ind, ]

x_train <- train[,-c(1)]
y_train <- train[,1]

x_test <- test[,-c(1)]
y_test <- test[,1]

#train bart_machine from train datasets
bart_machine <- bartMachine(x_train, y_train)
bart_machine
#check error asumptions and convergence diagnostics of the bart model
check_bart_error_assumptions(bart_machine)
plot_convergence_diagnostics(bart_machine)
plot_y_vs_yhat(bart_machine, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)
#evaluate k_fold cross validation
k_fold_cv(x_train, y_train, k_folds = 10)
#plot rmse by number of trees
rmse_by_num_trees(bart_machine, num_replicates = 10)
#perform test set prediction of trained model
oos_perf = bart_predict_for_test_data(bart_machine, x_test, y_test)
print(oos_perf$rmse)

#perform hyperparameter optimization to get the best
#performing bart machine model according to k fold cv value
bart_machine_cv <- bartMachineCV(x_train, y_train)
bart_machine_cv
#check error asumptions and convergence diagnostics of 
#the bart model with optimized hyperparameter
check_bart_error_assumptions(bart_machine_cv)
plot_convergence_diagnostics(bart_machine_cv)
#evaluate k_fold cross validation
k_fold_cv(x_train, y_train, k_folds = 10, k = 3, nu = 3, q = 0.99, num_trees = 200)
#perform test set prediction of trained model
oos_perf = bart_predict_for_test_data(bart_machine_cv, x_test, y_test)
print(oos_perf$rmse)
plot_y_vs_yhat(bart_machine_cv, credible_intervals = TRUE)

plot(oos_perf$y_hat,y_test)
investigate_var_importance(bart_machine_cv, num_replicates_for_avg = 20)

