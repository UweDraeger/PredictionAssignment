library(tidyverse)
library(tidymodels)
library(readr)

# read the data file, set error to NA
pml_training <-
        as_tibble(read_csv("pml-training.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)


pml_testing <-
        as_tibble(read_csv("pml-testing.csv", na = c("#DIV/0!", "NA"))) %>%
        mutate_if(is.character, as.factor)

#find all variables with data in testing (i.e. remove all cols with NA only)
predictors <- pml_testing %>% select(which(colMeans(is.na(.)) == 0)) 
