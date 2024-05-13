library(tidyverse)
library(ggpubr)
library(MASS) 
library(tidyverse) 
library(biotools)
library(caret) 
library(nnet)
library(rstatix)
library(pROC)
dfc <- read.csv("C:/Users/Mathi/Documents/Studie/2 semester/ST514, DS805 Multivariat Statistisk Analyse/Projekt/winequality-red.csv",header = TRUE)

summary(dfc)

# Create a new window to display the plots
par(mfrow = c(3, 4)) # Adjust the layout based on the number of columns in dfc

# Loop through each column and create QQ plots
for (i in 1:ncol(dfc)) {
  qqnorm(dfc[,i], main = paste("QQ - plot for", colnames(dfc)[i]))
  qqline(dfc[,i])
}


# Loop through each column and create Histogram plots
for (i in 1:ncol(dfc)) {
  hist(dfc[, i], main = paste("Histogram for", colnames(dfc)[i]))
}
par(mfrow = c(1, 1))
#It does not seem as though, there are any variables, that are normally distributed
#However let us try with a shapiro test.
shapiro_p_table <- data.frame(matrix(nrow = 1, ncol = ncol(dfc)))

colnames(shapiro_p_table) <- colnames(dfc)

for (i in 1:ncol(dfc)) {
  result <- shapiro.test(dfc[, i])
  shapiro_p_table[1, i] <- result$p.value
}

print(shapiro_p_table)

#We see that none are normally distributed





# We decide that quality can be put into three groups Bad, Fine and Great.
dfc$grade <- case_when(dfc$quality <= 7 ~ "Not great",
                       dfc$quality > 7 ~ "Great")




#We would like to check for homogeneity of covariance matrices, since that 
#and the type of distribution have much to say about which classification methods
#That we are going to use.

box_m(dfc[,1:11],dfc[,"grade"])

#We see that the p-value is belov 0.05, in which case we must reject the null-hypthosis
#There are at least two variables that do not have homogeneity of covariance matrices.


#However 
box_m(dfc[, c(1, 2, 3, 4, 6,7,8,9,10)], dfc[,"grade"]) #Does show a clear homogeneity of covariance matrices for these variables
# with a p-value > 0.05

# and
box_m(dfc[, c(1, 2, 3, 4, 6,7,8,9,10,11)], dfc[,"grade"]) #also shows a homogeneity of covariance matrices
# if we have p > 0.01


#Since that is the case, we choose to remove the variable chlrodies from our future model.


dfc_for_model <- dfc[, c(1, 2, 3, 4, 6,7,8,9,10)]
dfc_for_model$grade <- dfc$grade
#Since we do not have normal distribution but do have homogeneity we shall use a LDA and logistic regression
#Linear discriminant analysis

#Make a train and test dataset
training.individuals <- createDataPartition(dfc_for_model$grade, p = 0.8, list = FALSE)

train.data <- dfc_for_model[training.individuals, ] 
test.data <- dfc_for_model[-training.individuals, ] 
test.data$grade <- as.factor(test.data$grade)
train.data$grade <- as.factor(train.data$grade)


#LDA model
LDA_model <- lda(grade ~. ,data =train.data )
LDA_model

predictions <- LDA_model %>% predict(test.data) 


mean(predictions$class==test.data$grade) 



#Logistic Regression

LR_model <- glm(grade ~., data = train.data, family = binomial)


predictions <- predict(LR_model, newdata = test.data, type = "response")

predicted_classes <- ifelse(predictions > 0.5, "Not great", "Great")

mean(predicted_classes == test.data$grade)


#ROC -curve
test.data$predicted = predicted_classes

roc_curve <- roc(ifelse(test.data$grade == "Not great", 1, 0), ifelse(predicted_classes == "Not great", 1, 0))

# Plot ROC curve
plot(roc_curve, main = "ROC Curve", col = "blue")



#Our models tells us very little - since there are so few cases where the wine quality is "Great"
#It just predicts that - our choice of cattegories might have been too narrow
#for us to gain any meaningful insights.


#We will now try with three categories and see what happens

dfc$grade <- case_when(dfc$quality <= 4 ~ "bad",
                       dfc$quality > 4 & dfc$quality <= 6 ~ "okay",
                       dfc$quality > 6 ~ "Great")
dfc <- dfc[,-12]
box_m(dfc[,1:11],dfc[,"grade"])
#We can see that there are atleast two variables, that still do not have the same covariance matrices


#We also try to see if there are some subsets of size >3, where they have the same covariance, but we
#were not able to find such a thing.

#therefor we have to conclude, that with our three class' there are no homogeneity
#And the data is not normaly distributed, so we will use a logistic regression to predict.


#We will do as we did before
training.individuals <- createDataPartition(dfc$grade, p = 0.8, list = FALSE)

train.data <- dfc[training.individuals, ] 
test.data <- dfc[-training.individuals, ] 
test.data$grade <- as.factor(test.data$grade)
train.data$grade <- as.factor(train.data$grade)



model <- multinom(grade ~ ., data = train.data)

predictions <- predict(model, newdata = test.data, type = "class")

test.data$predicted <-predictions 



correct_predictions <- sum(predictions == test.data$grade)



# Calculate total number of predictions
total_predictions <- length(predictions)

# Calculate accuracy
accuracy <- correct_predictions / total_predictions

# Print accuracy
cat("Accuracy:", accuracy)

# 
#𝐸(APER)
aer(test.data$grade,test.data$predicted)
