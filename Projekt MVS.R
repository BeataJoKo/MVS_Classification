library(tidyverse)
library(ggpubr)

library(biotools)
library(rstatix)

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



#We would like to see if there are homogeneity of covariance matrices

#We do this with the Box's-M test
#%% 

# 
# res <- boxM(dfc[, 1:7], dfc[, "chlorides"])
# res
# 
# summary(res)
# # Print the result of Box's M test
# print(boxm_result)


# We decide that quality can be put into three groups Bad, Fine and Great.
dfc$grade <- case_when(dfc$quality <= 4 ~ "Bad",
                       dfc$quality > 4 & dfc$quality <= 7 ~ "Fine",
                       dfc$quality > 7 ~ "Great")


dfc$grade <- factor(dfc$grade)


