library(tidyverse)
library(caret)
library(broom)
library(car)

absolute_data <- read.csv("/Users/trent/Documents/MSDS/DS6021/Project/uva-ds6021-final-project/Data/Stars_clean_a.csv")
relative_data <- read.csv("/Users/trent/Documents/MSDS/DS6021/Project/uva-ds6021-final-project/Data/Stars_clean_r.csv")

stars <- relative_data

long <- gather(stars, key ='predictor', value = 'value',
               Temperature, R, A_M)
long2 <- gather(stars2, key ='predictor', value = 'value',
               log_Temp, log_R, A_M)

ggplot(long, aes(x = value, y = L, color = predictor)) +geom_point() + 
  facet_wrap(~predictor, scale = 'free_x')

ggplot(long2, aes(x = value, y = log_L, color = predictor)) +geom_point() + 
  facet_wrap(~predictor, scale = 'free_x')

model1 <- lm(L~Temperature + R + A_M, data = stars)
summary(model1)

model2 <- lm(log_L~log_Temp + log_R + A_M, data = stars2)
summary(model2)

model3 <- lm(log_L ~ log_Temp + log_R, data = stars2)
summary(model3)

# Assumptions
# Residual Plot
stars_pred <- mutate(stars, predictions = fitted(model1),
                     resid = residuals(model1))

ggplot(stars_pred, aes(x = predictions, y = resid)) + geom_point() +
  geom_hline(yintercept = 0, color = 'red')

stars_pred2 <- mutate(stars2, predictions = fitted(model2),
                      resid = residuals(model2))

ggplot(stars_pred2, aes(x = predictions, y = resid)) + geom_point() +
  geom_hline(yintercept = 0, color = 'red')

stars_pred3 <- mutate(stars2, predictions = fitted(model3),
                      resid = residuals(model3))

ggplot(stars_pred3, aes(x = predictions, y = resid)) + geom_point() +
  geom_hline(yintercept = 0, color = 'red')

# QQ Plot
ggplot(stars_pred, aes(sample = resid)) + stat_qq() +
  stat_qq_line(color = 'red')

ggplot(stars_pred2, aes(sample = resid)) + stat_qq() +
  stat_qq_line(color = 'red')

ggplot(stars_pred3, aes(sample = resid)) + stat_qq() +
  stat_qq_line(color = 'red')

# Correlation Plot
dat <- stars[, -c(5, 6, 7)]
cor_mat <- round(cor(dat), 2)

ggcorrplot::ggcorrplot(cor_mat, lab = T, method = 'circle', type = 'lower')

dat2 <- stars2[, -c(1, 2, 3, 5, 6, 7)]
cor_mat2 <- round(cor(dat2), 2)

ggcorrplot::ggcorrplot(cor_mat2, lab = T, method = 'circle', type = 'lower')

dat3 <- stars2[, -c(1, 2, 3, 4, 5, 6, 7)]
cor_mat3 <- round(cor(dat3), 2)

ggcorrplot::ggcorrplot(cor_mat3, lab = T, method = 'circle', type = 'lower')

#Collinearity
vif(model1)
vif(model2)
vif(model3)

# Assumptions with stars_dummies
stars2 <- absolute_data
stars2 <- stars2 %>% select(Temperature, L, R, A_M, Color, Spectral_Class, Type)
stars2 <- stars2 %>% mutate(log_L = log(L), log_R=log(R), log_Temp=log(Temperature))
dummies <- dummyVars('~.', data=stars2)
stars_dummies <- data.frame(predict(dummies, newdata=stars2))

stars_dummies_clean <- stars_dummies[, -c(1, 2, 3)]
model2 <- lm(log_L~., data = stars_dummies_clean)
summary(model2)

# Residual Plot
stars_pred_dum <- mutate(stars, predictions = fitted(model2),
                         resid = residuals(model2))
ggplot(stars_pred_dum, aes(x = predictions, y = resid)) + geom_point() +
  geom_hline(yintercept = 0, color = 'red')
# QQ Plot
ggplot(stars_pred_dum, aes(sample = resid)) + stat_qq() +
  stat_qq_line(color = 'red')


# Ridge
x <- model.matrix(log_L~0+., data = stars_dummies_clean)
y <- stars_dummies_clean$log_L

rmodel <- glmnet(x = x, y = y, alpha = 0, family = 'gaussian')
plot(rmodel, label = T, xvar = 'lambda')

rmodelglmnet <- cv.glmnet(x = x, y = y, alpha = 0, nfolds = 10)
rmodelglmnet$lambda.1se
rmodelglmnet$lambda.min

coef(rmodelglmnet, s = rmodelglmnet$lambda.min)
coef(rmodelglmnet, s = rmodelglmnet$lambda.1se)

plot(rmodelglmnet, label = T, xvar = 'lambda')

#min lambda
model_lambda_min <- lm(log_L~., data = stars_dummies_clean)
predictions <- predict(model_lambda_min, stars_dummies_clean)
resid(model_lambda_min)
summary(model_lambda_min)

model_lambda_min2 <- lm(log_L~.-A_M-ColorBlue-ColorBlue.White-
                          ColorRed-ColorWhite-ColorYellow.Orange-
                          ColorYellow.White-Spectral_ClassG-
                          Spectral_ClassK-Spectral_ClassM-
                          Spectral_ClassO-TypeRed.Dwarf-
                          TypeWhite.Dwarf, data = stars_dummies_clean)
predictions <- predict(model_lambda_min2, stars_dummies_clean)
resid(model_lambda_min2)
summary(model_lambda_min2)

model_lambda_min3 <- lm(log_L~.-A_M-ColorBlue-ColorBlue.White-
                          ColorRed-ColorWhite-ColorYellow.Orange-
                          ColorYellow.White-Spectral_ClassG-
                          Spectral_ClassK-Spectral_ClassM-
                          Spectral_ClassO-TypeRed.Dwarf-
                          TypeWhite.Dwarf-ColorOrange-log_R, data = stars_dummies_clean)
summary(model_lambda_min3)
ggplot(mapping=aes(x=predict(model_lambda_min3, stars_dummies), y = resid(model_lambda_min3))) + geom_point()

model_lambda_se <- lm