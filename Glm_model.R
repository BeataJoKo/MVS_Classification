library(ggplot2)
library(tidyr)
library(MVN)
library(caTools)
#install.packages('MVN')
#install.packages('caret')
library(lattice)
library(caret)

data <- read.csv('/Users/asgermollernielsen/Downloads/winequality-red.csv', header = TRUE)
summary(data)
data$new <- ifelse(data$quality <= 5, 0, 1)
df <- subset(data, select = -quality)
df$new <-factor(data$new, levels = c(0, 1), labels = c("bad", "good"))

normality_qq_plot <- function(variable) {
  qq <- qqnorm(variable)
  qqline(variable)
  cor_qq <- cor(qq$x, qq$y)
  return(cor_qq)
}

shapiro.test(df$pH)

# Perform normality test and generate QQ plots for each variable
correlations <- sapply(data[,1:11], normality_qq_plot)

# Print correlations
print(correlations)


#normality for fixed.acidity
qqc_fixed.acidity <- qqnorm(data$fixed.acidity)
qqline(data$fixed.acidity)
corqq_fixed.acidity <- cor(qqc_fixed.acidity$x, qqc_fixed.acidity$y)


# for volatile.acidity
qqc_volatile.acidity <- qqnorm(data$volatile.acidity)
qqline(data$volatile.acidity)
corqq_volatile.acidity <- cor(qqc_volatile.acidity$x, qqc_volatile.acidity$y)

# for citric.acid
qqc_citric.acid <- qqnorm(data$citric.acid)
qqline(data$citric.acid)
corqq_citric.acid <- cor(qqc_citric.acid$x, qqc_citric.acid$y)

# Assumption 1: Normality
mvn_test <- mvn(data[,1:11], mvnTest = "royston")

# Print the result
print("Multivariate Normality Test:")
print(mvn_test$multivariateNormality)

# Perform univariate normality tests for each variable
print("Univariate Normality Tests:")
print(mvn_test$univariateNormality)




set.seed(1)

#use 70% of dataset as training set and 30% as test set
sample <- sample.split(df, SplitRatio = 0.7)
train  <- subset(data, sample == TRUE)
test   <- subset(data, sample == FALSE)

model <- glm(new ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates +alcohol, train, family = binomial(link='logit'))
summary(model)
predictions_train <- predict(model, train)
binary_predictions_train <- ifelse(predictions_train >= 0.5, 1, 0)
train$pre <- binary_predictions_train
xtab_train <- table(train$pre,train$new)
cm_train <- caret::confusionMatrix((xtab_train))
cm_train
precision_train <- 357/(357+215)
recall_train <- 357/(357+72)
print(precision_train)
print(recall_train)

predictions <- predict(model, test)
binary_predictions <- ifelse(predictions >= 0.5, 1, 0)
test$pre <- binary_predictions

xtab <- table(test$pre, test$new)

cm <- caret::confusionMatrix(xtab)
cm

precision <- 177/(177+83)
recall <- 177/(177+43)
print(precision)
print(recall)



# k-fold 
ctrl <- trainControl(method = "cv", number = 10)
model_kfold <- train(new ~ ., data = df, method = "glm", trControl = ctrl)

summary(model_kfold)
confusionMatrix(model_kfold)
