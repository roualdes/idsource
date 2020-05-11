library(FOCI)
library(xgboost)
library(tidymodels)
library(readr)
library(tune)
library(workflow)
library(doParallel)
registerDoParallel(cores=3)

## rsample's bootstrap in general and for confidence intervals
## need extract predicted probabilities

df <- read.csv("data/sources.csv")
M <- ncol(df)
nms <- names(df)[3:M]

## determine important predictors
f <- foci(as.integer(df$group), df[, 3:M])
predictors <- nms[f]

## prep data
sources_s <- initial_split(df[, c(predictors, "group")])
sources_tr <- training(sources_s)
sources_te <- testing(sources_s)

sources_r <- recipe(group ~ ., data = sources_tr) %>%
    step_normalize(all_predictors())

sources_pr <- prep(sources_r, training = sources_tr)

train_data <- bake(sources_pr, new_data = sources_tr)
test_data <- bake(sources_pr, new_data = sources_te)

## prep model
sources_m <- boost_tree(trees = tune(),
                        tree_depth = tune(),
                        learn_rate = tune()) %>%
    set_engine("xgboost") %>%
    set_mode("classification")

sources_w <- workflow() %>%
    add_recipe(recipe(group ~ ., data = train_data)) %>%
    add_model(sources_m)

sources_cv <- vfold_cv(train_data, v = 5)

## tune
parameters_grid <- expand.grid(trees = round(seq(15, 50, by = 5)), # num_round
                               tree_depth = seq(3, 7, by = 1), # max_depth
                               learn_rate = seq(0.1, 0.5, by = 0.05)) # eta

sources_tr <- sources_w %>%
    tune_grid(resamples = sources_cv,
              grid = parameters_grid,
              metrics = metric_set(accuracy, mn_log_loss))

## pick best parameters
sources_tr %>%
    collect_metrics() %>%
    filter(.metric == "accuracy") %>%
    arrange(desc(mean))

parameters_b <- sources_tr %>%
    select_by_one_std_err(metric = "mn_log_loss", mean)

## final fit
sources_fit <- sources_w %>%
    finalize_workflow(parameters_b) %>%
    fit(data = train_data)

## validate model on test_data
sources_fit %>%
    predict(test_data) %>%
    bind_cols(test_data) %>%
    metrics(truth = group, estimate = .pred_class)

pdf <- sources_fit %>%
    predict(test_data, type="prob")

## predict test1 data
ndf <- read.csv("data/test1.csv")
newpred <- bake(sources_pr, new_data = ndf) %>% # predict based on normalized data
    mutate(group = predict(sources_fit, new_data = .) %>%
               pull(.pred_class))

ndf <- cbind(ndf, newpred) # align with non-normalized data


## save new predictions
write_csv(ndf, "data/test2_guesses.csv")

## compare against old predictions
t1 <- read.csv("data/test1_guesses.csv")
t2 <- read.csv("data/test2_guesses.csv")
