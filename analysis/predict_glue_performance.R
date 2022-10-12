library(tictoc)
library(caret)

df <- read.csv("../reports/probing_results_1200_per_class/task1_predict_task_performance.csv")
dim(df)

all_glue_tasks = c("rte", "cola", "mrpc", "sst2", "qnli", "qqp")
all_probe_tasks = c("bigram_shift", "coordination_inversion", "obj_number", "odd_man_out", "past_present", "subj_number", "tree_depth")

# All layers in one task
all_layers_from_one_task <- function(glue_task, probe_task) {
  layers=1:12
  features = paste(paste(probe_task, "_layer_", sep=""), layers, sep="")
  x_y_features = c(glue_task, features)
  formula = as.formula(paste(glue_task, "~ ."))
  trcontrol <- trainControl(method="cv", number=5)
  model <- train(formula, data=df[x_y_features], method="lm", trControl=trcontrol)
  rmse <- sqrt(mean(summary(model)$residuals^2))
  
  ctrl_features <- matrix(rnorm(length(features) * nrow(df), 0, 0.1), 
                          nrow=nrow(df), ncol=length(features))
  ctrl_label <- df[glue_task]
  Z <- as.data.frame(cbind(ctrl_label, ctrl_features))
  ctrl_model <- train(
    as.formula(sprintf("%s ~ .", glue_task)), data=Z, method="lm", trControl=trcontrol)
  ctrl_rmse <- sqrt(mean(summary(ctrl_model)$residuals^2))
  
  SST <- var(df[glue_task]) * (length(df)-1)
  SSE <- deviance(model)
  return(list("RMSE"=rmse,
              "ctrl_RMSE"=ctrl_rmse,
              "RMSE_reduction"=(ctrl_rmse-rmse)/ctrl_rmse*100,
              "explained_var"=(SST-SSE) / SST * 100 ))
}

set.seed(1234)
for (gt in all_glue_tasks) {
  print(sprintf("Predict %s", gt))
  for (pt in all_probe_tasks) {
    ret = all_layers_from_one_task(gt, pt)
    print(sprintf("probing task %s. RMSE %.4f. ctrl_RMSE %.4f RMSE_reduction %.2f", pt, ret$RMSE, ret$ctrl_RMSE, ret$RMSE_reduction))
  }
}

# Which features are significant?
probing_from_one_task <- function(glue_task, probe_task) {
  layers=1:12
  features = paste(paste(probe_task, "_layer_", sep=""), layers, sep="")
  x_y_features = c(glue_task, features)
  formula = paste(glue_task, "~ .")
  model <- lm(formula, data=df[x_y_features])
  anova_result <- anova(model)
  rmse <- sqrt(mean(summary(model)$residuals^2))
  sig_features <- features[anova_result[,5]<0.05]
  
  ctrl_features <- matrix(rnorm(length(features) * nrow(df), 0, 0.1), 
                          nrow=nrow(df), ncol=length(features))
  ctrl_label <- df[glue_task]
  Z <- as.data.frame(cbind(ctrl_label, ctrl_features))
  ctrl_model <- lm(sprintf("%s ~ .", glue_task), data=Z)
  ctrl_rmse <- sqrt(mean(summary(ctrl_model)$residuals^2))
  
  SST <- var(df[glue_task]) * (length(df)-1)
  SSE <- deviance(model)
  return(list("anova_result"=anova_result, 
              "sig_features"=sig_features,
              "RMSE"=rmse,
              "RMSE_reduction"=(ctrl_rmse-rmse)/ctrl_rmse*100,
              "explained_var"=(SST-SSE) / SST * 100 ))
}

set.seed(1234)
for (gt in all_glue_tasks) {
  print(sprintf("Predict %s", gt))
  for (pt in all_probe_tasks) {
    ret = probing_from_one_task(gt, pt)
    anova_result = ret$anova_result
    sig_features = ret$sig_features
    print(sprintf("probing task %s", pt))
    print(sprintf(sig_features))
  }
}

# Some layers some tasks
probing_some_layers_some_ptasks <- function(glue_task, features) {
  x_y_features = c(glue_task, features)
  formula = as.formula(paste(glue_task, "~ ."))
  # Need to convert to formula; otherwise caret throws error
  
  trctrl <- trainControl(method="cv", number=5)
  model <- train(formula, 
                 data=df[x_y_features], 
                 trControl=trctrl, 
                 method="lm")
  
  summary_result <- summary(model)
  rmse <- sqrt(mean(summary_result$residuals^2))
  
  ctrl_features <- matrix(rnorm(length(features) * nrow(df), 0, 0.1), 
                          nrow=nrow(df), ncol=length(features))
  ctrl_label <- df[glue_task]
  Z <- as.data.frame(cbind(ctrl_label, ctrl_features))
  ctrl_model <- train(
    as.formula(sprintf("%s ~ .", glue_task)), 
    data=Z, method="lm", 
    trControl=trainControl(method="cv", number=5))
  ctrl_rmse <- sqrt(mean(summary(ctrl_model)$residuals^2))
  if (ctrl_rmse == 0) {
    reduction = 0
  } else {
    reduction = (ctrl_rmse-rmse)/ctrl_rmse*100
  }
  
  return(list(
    "summary_result"=summary_result, 
    "RMSE"=rmse,
    "RMSE_reduction"=reduction ))
}

for (gt in all_glue_tasks) {
  features = c(
    "bigram_shift_layer_5",
    "coordination_inversion_layer_6",
    "obj_number_layer_1", 
    "odd_man_out_layer_5",  
    "past_present_layer_1",
    "subj_number_layer_1",
    "tree_depth_layer_1"  
  )
  ret <- probing_some_layers_some_ptasks(gt, features)
  print(sprintf("GLUE task %s, RMSE %.5f, RMSE_reduction %.2f", 
                gt, ret$RMSE, ret$RMSE_reduction))
}

# Only three features per task
probing_some_layers_some_ptasks_fast <- function(glue_task, features) {
  x_y_features = c(glue_task, features)
  formula = as.formula(paste(glue_task, "~ ."))
  
  model <- lm(formula,data=df[x_y_features])
  
  summary_result <- summary(model)
  rmse <- sqrt(mean(summary_result$residuals^2))
  
  ctrl_features <- matrix(rnorm(length(features) * nrow(df), 0, 0.1), 
                          nrow=nrow(df), ncol=length(features))
  ctrl_label <- df[glue_task]
  Z <- as.data.frame(cbind(ctrl_label, ctrl_features))
  ctrl_model <- lm(sprintf("%s ~ .", glue_task), data=Z)
  ctrl_rmse <- sqrt(mean(summary(ctrl_model)$residuals^2))
  if (ctrl_rmse == 0) {
    reduction = 0
  } else {
    reduction = (ctrl_rmse-rmse)/ctrl_rmse*100
  }
  
  return(list(
    "summary_result"=summary_result, 
    "RMSE"=rmse,
    "RMSE_reduction"=reduction ))
}

all_probe_features <- outer(all_probe_tasks, paste0("_layer_", 1:12), FUN="paste0")

find_best_features <- function(glue_task) {
  best_features = NA
  smallest_rmse = 10000
  for (i in 1:(length(all_probe_features)-2)) {
    for (j in (i+1):(length(all_probe_features)-1)) {
      for (k in (j+1):length(all_probe_features)) {
        feats <- c(all_probe_features[i], all_probe_features[j], all_probe_features[k])
        ret <- probing_some_layers_some_ptasks_fast(glue_task, feats)
        if (ret$RMSE < smallest_rmse) {
          smallest_rmse = ret$RMSE
          best_features = feats
        }
      }
    }
  }
  ret <- probing_some_layers_some_ptasks(glue_task, best_features)
  return(list(
    "max_rmse_reduction"=ret$RMSE_reduction,
    "best_features"=best_features,
    "rmse"=ret$RMSE
  ))
}

for (gt in all_glue_tasks) {
  tic("find_best_features")
  retval = find_best_features(gt)
  toc()
  print(sprintf("Glue task %s, RMSE %.4f, max rmse reduction %.2f, achieved using %s",
                gt, retval$rmse, retval$max_rmse_reduction, retval$best_features))
}