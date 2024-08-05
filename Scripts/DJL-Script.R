library(tidyverse)
library(glmnet)
library(caret)

# Set a seed to replicate results
set.seed(6021)

# Set working directory to correct file path
setwd("~/Documents/MSDS/DS6021/uva-ds6021-final-project/")

# Read in the data-set. Drop the first column
stars <- read.csv('Data/Stars_clean_a.csv')
stars <- stars %>% select(Temperature, L, R, A_M, Color, Spectral_Class, Type)
stars <- stars %>% mutate(log_L = log(L), log_R=log(R), log_Temp=log(Temperature))
dummies <- dummyVars('~.', data=stars)
stars_dummies <- data.frame(predict(dummies, newdata=stars))
View(stars_dummies)

# get the design matrix and Y vector
X <- model.matrix(log_L~0+.-L-R-Temperature, data=stars)
Y <- stars$log_L

# Run LASSO regression for feature selection
lasso <- glmnet(x=X, y=Y, alpha=1, family='gaussian')
plot(lasso, label=T, xvar='lambda')

# Use k-fold CV to get the ideal lambda value
kcv_lambda <- cv.glmnet(x=X, y=Y, alpha=1, nfolds=10)
print(paste('lambda.min:', kcv_lambda$lambda.min))
print(paste('lambda.1se:', kcv_lambda$lambda.1se))

# print the coefficients that exist at the lambda.min
coef(kcv_lambda, s=kcv_lambda$lambda.min)

# print the coefficients that exist in the lambda.1se
coef(kcv_lambda, s=kcv_lambda$lambda.1se)

# print the coefficients for a more extreme lambda
lambda_extreme = 0.1353353
coef(kcv_lambda, s=lambda_extreme)

# create plot of lambdas from k-fold cv
plot(kcv_lambda, label=T, xvar='lambda')

# create a model with the predictors present with lambda.min

# Temperature         2.014603e+27
# R                   1.962654e+19
# A_M                 .           
# ColorBlue          -1.524688e+31
# ColorBlue-White     .           
# ColorOrange         1.080312e+32
# ColorOrange-Red     .           
# ColorRed           -1.781286e+31
# ColorWhite          1.398875e+31
# ColorYellow         .           
# ColorYellow-Orange  1.312005e+31
# ColorYellow-White   1.439331e+31
# Spectral_ClassB    -6.962318e+30
# Spectral_ClassF    -9.961080e+30
# Spectral_ClassG    -1.066636e+31
# Spectral_ClassK     1.165961e+31
# Spectral_ClassM    -7.694310e+30
# Spectral_ClassO     1.935197e+31
# TypeHyper Giants    6.619715e+31
# TypeMain Sequence  -3.977143e+31
# TypeRed Dwarf       .           
# TypeSuper Giants    6.819637e+31
# TypeWhite Dwarf    -3.998087e+31

model_lambda_min_predictors = lm(L~.--A_M-ColorBlue.White-ColorOrange.Red-ColorYellow-TypeRed.Dwarf, data=stars_dummies)
predictions <- predict(model_lambda_min_predictors, stars_dummies)
resid(model_lambda_min_predictors)
summary(model_lambda_min_predictors)

# A_M                -0.52898010
# ColorBlue           .         
# ColorBlue-White     0.06413317
# ColorOrange         1.33013889
# ColorOrange-Red    -1.26373562
# ColorRed            .         
# ColorWhite          .         
# ColorYellow        -2.27606050
# ColorYellow-Orange  .         
# ColorYellow-White  -0.35958904
# Spectral_ClassB     .         
# Spectral_ClassF    -1.38683631
# Spectral_ClassG     .         
# Spectral_ClassK    -0.17994256
# Spectral_ClassM     .         
# Spectral_ClassO     0.19290708
# TypeHyper Giants    .         
# TypeMain Sequence   2.22240897
# TypeRed Dwarf       1.29077358
# TypeSuper Giants    3.58772248
# TypeWhite Dwarf     .         
# log_R               0.67454301
# log_Temp            0.69531489 

model_lambda_one_se <- lm(log_L~.-TypeWhite.Dwarf-TypeHyper.Giants-Spectral_ClassM-Spectral_ClassG-Spectral_ClassB-ColorYellow.Orange-ColorWhite-ColorRed-ColorBlue-L-R-Temperature, data=stars_dummies)
ggplot(mapping=aes(x=predict(model_lambda_one_se, stars_dummies), y=resid(model_lambda_one_se))) +
  geom_point()
summary(model_lambda_one_se)

# A_M                -0.5499706
# ColorBlue           .        
# ColorBlue-White     0.0263492
# ColorOrange         0.2644727
# ColorOrange-Red     .        
# ColorRed            .        
# ColorWhite          .        
# ColorYellow        -1.3835943
# ColorYellow-Orange  .        
# ColorYellow-White   .        
# Spectral_ClassB     .        
# Spectral_ClassF    -1.2819090
# Spectral_ClassG     .        
# Spectral_ClassK     .        
# Spectral_ClassM     .        
# Spectral_ClassO     0.2476681
# TypeHyper Giants    .        
# TypeMain Sequence   1.5417829
# TypeRed Dwarf       0.8252365
# TypeSuper Giants    3.1285907
# TypeWhite Dwarf    -0.2979259
# log_R               0.5979357
# log_Temp            0.6305876

# started with all non-zero coefficients from above after increasing the lambda value. Removed A_M because of collinearity issues.
# Then removed all other predictors with a large p-value as per the principal of parsimony until everything had a significant p-value
# and all VIF values were appropriate.

model_lambda_extreme <- lm(log_L~ColorYellow+Spectral_ClassF+TypeMain.Sequence+TypeSuper.Giants+TypeWhite.Dwarf+log_R+log_Temp, data=stars_dummies)
ggplot(mapping=aes(x=predict(model_lambda_extreme, stars_dummies), y=resid(model_lambda_extreme))) +
  geom_point()

summary(model_lambda_extreme)
car::vif(model_lambda_extreme)
