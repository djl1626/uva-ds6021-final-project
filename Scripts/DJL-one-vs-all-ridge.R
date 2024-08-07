library(tidyverse)
library(glmnet)
library(gridExtra)

# Set a seed to replicate results
set.seed(6021)

# Set working directory to correct file path
setwd("~/Documents/MSDS/DS6021/uva-ds6021-final-project/")

# Read in the data-set. Drop the first column
stars <- read.csv('Data/Stars_cleanest_r.csv')
stars <- stars %>% dplyr::select(Temperature, L, R, A_M, Color, Spectral_Class, Type)
stars <- stars %>% mutate(log_L = log(L), log_R=log(R), log_Temp=log(Temperature))
dummies <- dummyVars('~Type', data=stars)
stars_dummies <- data.frame(stars, predict(dummies, newdata=stars))

# Split the data into train and test
train_sample <- sample(1:nrow(stars_dummies), .8*nrow(stars_dummies), replace=FALSE)
train_data <- stars_dummies[train_sample,]
test_data <- stars_dummies[-train_sample,]

# Brown Dwarf Model
bd_X <- model.matrix(TypeBrown.Dwarf~0+log_L+log_R+log_Temp, data=train_data)
bd_y <- train_data$TypeBrown.Dwarf

bd_ridge_model <- glmnet(x=bd_X, y=bd_y, data=stars, alpha=0, family='binomial')

bd_cv_model <- cv.glmnet(x=bd_X, y=bd_y, data=stars, alpha=0, family='binomial', nfolds=10)

bd_train_probs <- predict(bd_ridge_model, newx=bd_X, s=bd_cv_model$lambda.min, type='response')

# Red Dwarf Model
rd_X <- model.matrix(TypeRed.Dwarf~0+log_L+log_R+log_Temp, data=train_data)
rd_y <- train_data$TypeRed.Dwarf

rd_ridge_model <- glmnet(x=rd_X, y=rd_y, data=stars, alpha=0, family='binomial')

rd_cv_model <- cv.glmnet(x=rd_X, y=rd_y, data=stars, alpha=0, family='binomial', nfolds=10)

rd_train_probs <- predict(rd_ridge_model, newx=rd_X, s=rd_cv_model$lambda.min, type='response')

# White Dwarf Model
wd_X <- model.matrix(TypeWhite.Dwarf~0+log_L+log_R+log_Temp, data=train_data)
wd_y <- train_data$TypeWhite.Dwarf

wd_ridge_model <- glmnet(x=wd_X, y=wd_y, data=stars, alpha=0, family='binomial')

wd_cv_model <- cv.glmnet(x=wd_X, y=wd_y, data=stars, alpha=0, family='binomial', nfolds=10)

wd_train_probs <- predict(wd_ridge_model, newx=wd_X, s=wd_cv_model$lambda.min, type='response')

# Main Sequence Model
ms_X <- model.matrix(TypeMain.Sequence~0+log_L+log_R+log_Temp, data=train_data)
ms_y <- train_data$TypeMain.Sequence

ms_ridge_model <- glmnet(x=ms_X, y=ms_y, data=stars, alpha=0, family='binomial')

ms_cv_model <- cv.glmnet(x=ms_X, y=ms_y, data=stars, alpha=0, family='binomial', nfolds=10)

ms_train_probs <- predict(ms_ridge_model, newx=ms_X, s=ms_cv_model$lambda.min, type='response')

# Hyper Giants Model
hg_X <- model.matrix(TypeHyper.Giants~0+log_L+log_R+log_Temp, data=train_data)
hg_y <- train_data$TypeHyper.Giants

hg_ridge_model <- glmnet(x=hg_X, y=hg_y, data=stars, alpha=0, family='binomial')

hg_cv_model <- cv.glmnet(x=hg_X, y=hg_y, data=stars, alpha=0, family='binomial', nfolds=10)

hg_train_probs <- predict(hg_ridge_model, newx=hg_X, s=hg_cv_model$lambda.min, type='response')

# Super Giants Model
sg_X <- model.matrix(TypeSuper.Giants~0+log_L+log_R+log_Temp, data=train_data)
sg_y <- train_data$TypeSuper.Giants

sg_ridge_model <- glmnet(x=sg_X, y=sg_y, data=stars, alpha=0, family='binomial')

sg_cv_model <- cv.glmnet(x=sg_X, y=sg_y, data=stars, alpha=0, family='binomial', nfolds=10)

sg_train_probs <- predict(sg_ridge_model, newx=sg_X, s=sg_cv_model$lambda.min, type='response')

train_probs <- data.frame(bd_train_probs, rd_train_probs, wd_train_probs, ms_train_probs, hg_train_probs, sg_train_probs)
colnames(train_probs) <- c('Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Hyper Giants', 'Super Giants')

train_rowmax <- apply(train_probs, 1, max)
train_log_odds <- rowmax / log(1-train_rowmax)
train_rowmax_index <- apply(train_probs, 1, which.max)

# ggplot(mapping=aes(x=train_data$log_Temp, y=log_odds)) + geom_point()

train_preds <- colnames(train_probs)[rowmax_index]

paste('Training Accuracy:', sum((as.integer(train_preds == train_data$Type)) / nrow(train_data)))

M_train <- as.matrix(train_data[,11:16])
colnames(M_train) <- c("Brown Dwarf", "Hyper Giants", "Main Sequence", "Red Dwarf", "Super Giants", "White Dwarf")
pROC::multiclass.roc(train_preds, M_train)

# Test the model
bd_test_X <- model.matrix(TypeBrown.Dwarf~0+log_L+log_R+log_Temp, data=test_data)
bd_test_y <- test_data$TypeBrown.Dwarf
bd_test_probs <- predict(bd_ridge_model, newx=bd_test_X, s=bd_cv_model$lambda.min, type='response')

rd_test_X <- model.matrix(TypeRed.Dwarf~0+log_L+log_R+log_Temp, data=test_data)
rd_test_y <- test_data$TypeRed.Dwarf
rd_test_probs <- predict(rd_ridge_model, newx=rd_test_X, s=rd_cv_model$lambda.min, type='response')

wd_test_X <- model.matrix(TypeWhite.Dwarf~0+log_L+log_R+log_Temp, data=test_data)
wd_test_y <- test_data$TypeWhite.Dwarf
wd_test_probs <- predict(wd_ridge_model, newx=wd_test_X, s=wd_cv_model$lambda.min, type='response')

ms_test_X <- model.matrix(TypeMain.Sequence~0+log_L+log_R+log_Temp, data=test_data)
ms_test_y <- test_data$TypeMain.Sequence
ms_test_probs <- predict(ms_ridge_model, newx=ms_test_X, s=ms_cv_model$lambda.min, type='response')

hg_test_X <- model.matrix(TypeHyper.Giants~0+log_L+log_R+log_Temp, data=test_data)
hg_test_y <- test_data$TypeHyper.Giants
hg_test_probs <- predict(hg_ridge_model, newx=hg_test_X, s=hg_cv_model$lambda.min, type='response')

sg_test_X <- model.matrix(TypeSuper.Giants~0+log_L+log_R+log_Temp, data=test_data)
sg_test_y <- test_data$TypeSuper.Giants
sg_test_probs <- predict(sg_ridge_model, newx=sg_test_X, s=sg_cv_model$lambda.min, type='response')

test_probs <- data.frame(bd_test_probs, rd_test_probs, wd_test_probs, ms_test_probs, hg_test_probs, sg_test_probs)
colnames(test_probs) <- c('Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Hyper Giants', 'Super Giants')

test_rowmax <- apply(test_probs, 1, max)
test_log_odds <- test_rowmax / log(1-test_rowmax)
test_rowmax_index <- apply(test_probs, 1, which.max)

ggplot(mapping=aes(x=test_data$log_Temp, y=log_odds)) + geom_point()

test_preds <- colnames(test_probs)[test_rowmax_index]

paste('testing Accuracy:', sum((as.integer(test_preds == test_data$Type)) / nrow(test_data)))

M_test <- as.matrix(test_data[,11:16])
colnames(M_test) <- c("Brown Dwarf", "Hyper Giants", "Main Sequence", "Red Dwarf", "Super Giants", "White Dwarf")
pROC::multiclass.roc(test_preds, M_test)

log_L_plot <- ggplot(mapping=aes(x=train_data$log_L, y=train_log_odds)) + geom_point() + xlab('log(L)')
log_R_plot <- ggplot(mapping=aes(x=train_data$log_R, y=train_log_odds)) + geom_point() + xlab('log(R)')
log_Temp_plot <- ggplot(mapping=aes(x=train_data$log_Temp, y=train_log_odds)) + geom_point() + xlab('log(Temperature)')
plot_grid <- grid.arrange(log_L_plot, log_R_plot, log_Temp_plot)
ggsave('logistic_assumptions.png', plot=plot_grid)
