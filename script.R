library(tidyverse)
library(readr)
library(tidymodels)
library(workflows)
library(tune)
library(ranger)
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

# read the data files, set missing data and errors to NA
pml_training <-
        as_tibble(read_csv("pml-training.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)

pml_testing <-
        as_tibble(read_csv("pml-testing.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)

# find all variables with usable data in testing (i.e. remove all cols with NA only)
predictors <- pml_testing %>% select(which(colMeans(is.na(.)) == 0))
prednames <- colnames(predictors)

# reduce training set to variables actually available in test set
# predname[60] is problem_id (copy of X1)
# predname[1:7] - descriptive variables, unsuited for model
# add back the independent variable
train <- pml_training %>% 
        select(prednames[-60]) %>% 
        select(-(X1:num_window)) %>%
        mutate(classe = pml_training$classe)

# split train and validation, create cv object 
set.seed(12345)

train_split <- initial_split(train, prop = 0.75)
train_cv <- vfold_cv(train)


# preprocess (just normalize the numeric data) and define recipe
pml_recipe <-
        recipe(classe ~ ., data = train) %>%
        step_normalize(all_numeric())

# spec the model to be random forest, tune mtry, ranger
pml_model <- rand_forest() %>%
        set_args(mtry = tune()) %>%
        set_engine("ranger", importance = "impurity") %>%
        set_mode("classification")

# build the workflow
pml_workflow <- workflow() %>%
        add_recipe(pml_recipe) %>%
        add_model(pml_model)


# tune mtry parameter
pml_grid <- expand.grid(mtry = c(7, 8, 9))
pml_tune_results <- pml_workflow %>%
        tune_grid(
                resamples = train_cv,
                grid = pml_grid,
                metrics = metric_set(accuracy, roc_auc)
        )

# set mtry parameter in final workflow
# choice of accuracy over roc_auc since roc_auc = 1 for all params
mtry_final <- pml_tune_results %>%
        select_best(metric = "accuracy")

pml_final_workflow <- pml_workflow %>%
        finalize_workflow(mtry_final)
        
# evaluate model on validation set
pml_fit <- pml_final_workflow %>%
        last_fit(train_split)

train_perf <- pml_fit %>%
        collect_metrics()

validat_pred <- pml_fit %>%
        collect_predictions()

validat_pred %>%
        conf_mat(truth = classe, estimate = .pred_class)



# fitting final model

# predict test data

stopCluster(cluster)
registerDoSEQ()
