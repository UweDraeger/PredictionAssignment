library(tidyverse)
library(readr)
library(tidymodels)
library(ranger)
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

# read the data files, set missing data and errors to NA
pml_training <-
        as_tibble(read_csv("pml-training.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)

# to avoid confusion, the "testing" set is called newdata 
pml_newdata <-
        as_tibble(read_csv("pml-testing.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)

# find all variables with usable data in testing (i.e. remove all cols with NA only)
vars_with_data <- pml_newdata %>% select(which(colMeans(is.na(.)) == 0))
prednames <- colnames(vars_with_data)

# reduce training set to variables actually available in test set
# predname[60] is problem_id (copy of X1)
# predname[1:7] - descriptive variables, unsuited for forecasting
# add back the independent variable
train <- pml_training %>% 
        select(prednames[-60]) %>% 
        select(-(X1:num_window)) %>%
        mutate(classe = pml_training$classe)

# split train and test, create cross validation object 
set.seed(12345)

train_split <- initial_split(train, prop = 0.75)
pml_train <- training(train_split)
pml_test <- testing(train_split)

train_cv <- vfold_cv(pml_train)


# preprocess (just normalize the numeric data and deal with near zero variance 
# data) and define recipe
pml_recipe <-
        recipe(classe ~ ., data = train_split) %>%
        step_normalize(all_numeric()) %>%
        step_zv(all_predictors())

# spec the model to be random forest, tune mtry and trees, use ranger 
# implementation of random forest 
pml_model <- rand_forest() %>%
        set_args(mtry = tune(), trees = tune()) %>%
        set_engine("ranger") %>%
        set_mode("classification")

# build the workflow 
pml_workflow <- workflow() %>%
        add_recipe(pml_recipe) %>%
        add_model(pml_model)


# tune mtry and trees parameters
pml_grid <- expand.grid(mtry = c(3, 5, 7), trees = c(20, 50, 100))
pml_tune_results <- pml_workflow %>%
        tune_grid(
                resamples = train_cv,
                grid = pml_grid,
                metrics = metric_set(accuracy, sens, spec)
        )
results <- pml_tune_results %>% 
        collect_metrics()

accuracies <- results %>% 
        filter(.metric == "accuracy") %>% 
        select(mtry, trees, mean, std_err) %>% 
        arrange(mtry, trees)
ggplot(data = accuracies) +
        geom_point(aes(x = mtry, y = mean, color = trees)) 

 
# set parameters in final workflow
# choice of accuracy 
tuning_result <- pml_tune_results %>%
        select_best(metric = "accuracy")

pml_workflow <- pml_workflow %>%
        finalize_workflow(tuning_result)
        
# evaluate model on validation set
pml_fit <- pml_workflow %>%
        last_fit(train_split)

train_perf <- pml_fit %>%
        collect_metrics()

validat_pred <- pml_fit %>%
        collect_predictions()

validat_pred %>%
        conf_mat(truth = classe, estimate = .pred_class)


# fitting final model
pml_final <- fit(pml_workflow, pml_training)

# predict test data

predict(pml_final, new_data = pml_newdata)

stopCluster(cluster)
registerDoSEQ()
