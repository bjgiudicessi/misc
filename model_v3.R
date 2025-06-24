# Load required libraries
library(dplyr)
library(readr)
library(corrplot)
library(ggplot2)
library(effsize)
library(Information)
library(car)
library(vcd)
library(randomForest)
library(gridExtra)
library(knitr)
library(kableExtra)
library(rmarkdown)
library(xgboost)
library(pROC)
library(caret)
library(ROSE)
library(lightgbm)

# ---- Helper: safe PDF writer ---------------------------------------------
safe_pdf <- function(path, plot_call) {
  tryCatch({
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    pdf(path)
    plot_call()
    dev.off()
  }, error = function(e) {
    message("❌ Could not write PDF to ", path, ": ", e$message)
  })
}

# ---- Load data ------------------------------------------------------------
path <- "G:/Clients/Strategic America/Pella/PPM/Data/variables_1.csv"
df <- read_csv(path)

# Log original variable types
orig_types <- sapply(df, class)
tryCatch(write.csv(data.frame(Variable = names(orig_types), Type = orig_types),
                   "G:/Clients/Strategic America/Pella/PPM/Data/original_variable_types.csv", row.names = FALSE),
         error = function(e) message("Log original types failed: ", e$message))

# ---- Pre‑processing -------------------------------------------------------

df <- df %>%
  mutate(response_bin = ifelse(response == "Y", 1, 0)) %>%
  select(-luid)

# Mosaic binning
mosaic_bins <- df %>% count(mosaic) %>% mutate(bin = ntile(n, 10))
df <- df %>% left_join(mosaic_bins %>% select(mosaic, mosaic_bin = bin), by = "mosaic") %>% select(-mosaic)

# One‑hot encode hh_comp
hh_mat <- model.matrix(~ hh_comp - 1, data = mutate(df, hh_comp = as.factor(hh_comp))) %>% as.data.frame()
df <- bind_cols(df %>% select(-hh_comp), hh_mat)

# Scale continuous vars
scale_vars <- c("nat_inc", "est_hh_income")
df <- df %>% mutate(across(all_of(scale_vars), scale))

# Missing & NZV report
missing_pct <- sapply(df, \(x) mean(is.na(x)) * 100)
nzv_flag   <- sapply(df, \(x) length(unique(x)) == 1)
qual_df <- data.frame(Variable = names(df), MissingPercent = round(missing_pct,2), NearZeroVariance = nzv_flag)
tryCatch(write.csv(qual_df, "G:/Clients/Strategic America/Pella/PPM/Data/variable_quality_summary.csv", row.names = FALSE),
         error = \(e) message("Quality summary write failed: ", e$message))

# Processed types log
proc_types <- sapply(df, \(x) paste(class(x), collapse=", "))
tryCatch(write.csv(data.frame(Variable = names(proc_types), Type = proc_types),
                   "G:/Clients/Strategic America/Pella/PPM/Data/processed_variable_types.csv", row.names = FALSE),
         error = \(e) message("Processed types write failed: ", e$message))

# ---- Step 1: Spearman + Mann‑Whitney -------------------------------------
num_vars <- df %>% select(where(is.numeric)) %>% select(-response_bin)
spear_df <- lapply(names(num_vars), \(v) {
  sp <- cor.test(df$response_bin, df[[v]], method = "spearman")
  mw <- wilcox.test(df[[v]] ~ df$response_bin)
  data.frame(variable=v, spearman_rho=as.numeric(sp$estimate), spearman_p=sp$p.value, mw_p=mw$p.value)
}) |> bind_rows()

strong_vars <- spear_df %>% filter(abs(spearman_rho) > .3) %>% arrange(desc(abs(spearman_rho)))
if(nrow(strong_vars)==0) strong_vars <- spear_df %>% arrange(desc(abs(spearman_rho))) %>% slice_head(n=5)

top_vars <- strong_vars$variable

# ---- Visualisations -------------------------------------------------------
plot1 <- ggplot(spear_df %>% slice_max(abs(spearman_rho), n=10), aes(reorder(variable, abs(spearman_rho)), spearman_rho)) +
  geom_col(fill="steelblue") + coord_flip() + theme_minimal() + labs(title="Top Spearman", x=NULL, y="rho")

cliff_vals <- sapply(names(num_vars), \(v) cliff.delta(df[[v]], df$response_bin)$estimate)
plot2 <- ggplot(data.frame(variable=names(cliff_vals), delta=cliff_vals) %>% slice_max(abs(delta), n=10),
                aes(reorder(variable, abs(delta)), delta)) +
  geom_col(fill="darkgreen") + coord_flip() + theme_minimal() + labs(title="Top Cliff's Delta", x=NULL, y="Delta")

iv_df <- df %>% mutate(response_bin = as.numeric(response_bin), across(where(is.character), as.factor)) %>% select_if(~ length(unique(class(.))) == 1)
info <- create_infotables(iv_df, y="response_bin", bins=10)
plot3 <- ggplot(info$Summary %>% slice_max(IV, n=10), aes(reorder(Variable, IV), IV)) +
  geom_col(fill="darkred") + coord_flip() + theme_minimal() + labs(title="Top IV", x=NULL, y="IV")

safe_pdf("G:/Clients/Strategic America/Pella/PPM/Data/top_predictors_visuals.pdf", function() {
  gridExtra::grid.arrange(plot1, plot2, plot3, ncol=1)
})

# ---- Caret + ROSE XGBoost -------------------------------------------------

# Close only graphics devices—not all file connections
try({while (!is.null(dev.list())) dev.off()}, silent = TRUE)

model_df <- df %>% select(all_of(top_vars), response_bin) %>% mutate(across(-response_bin, as.numeric),
                                                                     response_bin=factor(ifelse(response_bin==1,"Yes","No")))
set.seed(123)
train_idx <- createDataPartition(model_df$response_bin, p=.7, list=FALSE)
train_df <- na.omit(model_df[train_idx,])
test_df  <- na.omit(model_df[-train_idx,])

ctrl <- trainControl(method="repeatedcv", number=5, repeats=3, sampling="rose", classProbs=TRUE, summaryFunction=twoClassSummary)
param_grid <- expand.grid(nrounds=c(100,200), max_depth=c(3,5), eta=c(.05,.1), gamma=0, colsample_bytree=c(.6,.8), min_child_weight=c(1,5), subsample=c(.6,.8))

set.seed(123)
xgb_fit <- train(response_bin~., data=train_df, method="xgbTree", metric="ROC", trControl=ctrl, tuneGrid=param_grid, verbose=FALSE)

best_auc <- max(xgb_fit$results$ROC)
prob_xgb <- predict(xgb_fit, test_df, type="prob")[,"Yes"]
roc_xgb <- roc(ifelse(test_df$response_bin=="Yes",1,0), prob_xgb)
auc_xgb <- auc(roc_xgb)

safe_pdf("G:/Clients/Strategic America/Pella/PPM/Data/caret_xgb_roc.pdf", function() {
  plot(roc_xgb, col="purple", lwd=2, main=paste0("Caret XGB ROC (AUC ", round(auc_xgb,3), ")"))
})

# ---- LightGBM -------------------------------------------------------------
train_mat <- as.matrix(select(train_df,-response_bin)); train_lbl <- ifelse(train_df$response_bin=="Yes",1,0)
valid_mat <- as.matrix(select(test_df,-response_bin));  valid_lbl <- ifelse(test_df$response_bin=="Yes",1,0)

lgb_train <- lgb.Dataset(train_mat, label=train_lbl)
params <- list(objective="binary", metric="auc", learning_rate=.05, num_leaves=31, feature_fraction=.8, bagging_fraction=.8, bagging_freq=5)
set.seed(123)
lgb_mod <- lgb.train(params, lgb_train, 500, valids=list(valid=lgb.Dataset.create.valid(lgb_train, valid_mat, label=valid_lbl)), early_stopping_rounds=50, verbose=-1)

pred_lgb <- predict(lgb_mod, valid_mat)
roc_lgb <- roc(valid_lbl, pred_lgb)
auc_lgb <- auc(roc_lgb)

safe_pdf("G:/Clients/Strategic America/Pella/PPM/Data/lightgbm_roc.pdf", function() {
  plot(roc_lgb, col="orange", lwd=2, main=paste0("LightGBM ROC (AUC ", round(auc_lgb,3), ")"))
})

# ---- Ensemble (Average Probabilities) ------------------------------------
ensemble_prob <- (prob_xgb + pred_lgb) / 2
roc_ens <- roc(valid_lbl, ensemble_prob)
auc_ens <- auc(roc_ens)

safe_pdf("G:/Clients/Strategic America/Pella/PPM/Data/ensemble_roc.pdf", function() {
  plot(roc_ens, col="darkblue", lwd=2, main=paste0("Ensemble ROC (AUC ", round(auc_ens,3), ")"))
})

# ---- Append performance summary ------------------------------------------
perf_file <- paste0("G:/Clients/Strategic America/Pella/PPM/Data/model_perf_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt")"
summary_lines <- c(
  paste0("Caret XGBoost Tuned: CV AUC = ", round(best_auc, 3), ", Holdout AUC = ", round(auc_xgb, 3)),
  paste0("LightGBM Holdout AUC = ", round(auc_lgb, 3)),
  paste0("Ensemble (Avg) Holdout AUC = ", round(auc_ens, 3))
)

# Ensure directory exists
try(dir.create(dirname(perf_file), recursive = TRUE, showWarnings = FALSE), silent = TRUE)

# Close any lingering connections before writing
try({while (!is.null(dev.list())) dev.off()}, silent = TRUE)

# Write summary safely using explicit connection
tryCatch({
  con <- file(perf_file, open = "a")
  if (!isOpen(con)) stop("File connection could not be opened.")
  writeLines(summary_lines, con)
  close(con)
}, error = function(e) {
  message("Cannot write summary: ", e$message)
})

# ---- Save predictions ----------------------------------------------------- ----------------------------------------------------- -----------------------------------------------------
pred_df <- data.frame(actual=test_df$response_bin, prob_xgb=prob_xgb, prob_lgb=pred_lgb)
tryCatch(write.csv(pred_df, "G:/Clients/Strategic America/Pella/PPM/Data/model_predictions.csv", row.names=FALSE), error=function(e) message("Cannot write predictions: ", e$message))
