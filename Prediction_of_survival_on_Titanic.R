rm(list=ls(all=TRUE))

setwd("C:/Users/Prerit/Desktop/Datatest/Titanic")

library(tidyverse)

# Read data

train <- read.table("train.csv", header = TRUE, sep=",")
test<- read.table("test.csv", header = TRUE, sep=",")

str(train)

summary(train)

# writing to file

i<-1
write_to_file <- function(pred_vec, type)
{
submit1 <- data.frame(test2$PassengerId, pred_vec)
colnames(submit1)[1]<-"PassengerId"
submit1$Survived <- submit1$pred_vec
str(submit1)

table(submit1$Survived)

submit1$pred_vec <- NULL

str(submit1)
filename <- paste("submission_",type,i,".csv",sep="")

write.csv(submit1, filename, row.names = FALSE)
i<-i+1
}

# Pre-processing

sum(is.na(train))

missing <- colSums(is.na(train))
sort(missing, decreasing=TRUE)

library(mice)

md.pattern(train)

library(VIM)

mice_plot <- aggr(train, col=c('navyblue','yellow'),
                    numbers=TRUE, sortVars=TRUE,
                    labels=names(train), cex.axis=.7,
                    gap=3, ylab=c("Missing data","Pattern"))


# Less than 2% missing values

library(DMwR)
train2<- centralImputation(train) #Central Imputation

# train2 <- knnImputation(train,scale=T,k=10) #KNN Imputation
# sum(is.na(train2))

sum(is.na(train2))

# train2 <- na.omit(train)
# str(train2)
# sum(is.na(train2))



#Looking at data

str(train2)

train3 <- train2

# train3$Pclass <- as.factor(train3$Pclass)

# str(train3)

train3$Survived <- as.factor(train3$Survived)
str(train3)

hist(train3$Age)
hist(train3$SibSp)
hist(train3$Parch)
hist(train3$Fare)

# cor(train3$Pclass, train3$Fare)

train4 <- train3[,c("Pclass","Sex","Age","SibSp", "Survived", "Parch", "Fare", "Embarked")]

set.seed(123)
library(caTools)
spl <- sample.split(train4, 0.7)
train5 <- train4[spl==TRUE,]
valid <- train4[spl==FALSE,]

# Logistic Regression

logit <- glm(Survived~., data=train5, family=binomial)
summary(logit)

# model2 <- step(logit)
# summary(model2)

# Validation

# Removing null values

predictions <- predict(logit,valid,type="response") 
summary(predictions)
head(predictions)

library(ROCR)
# to make an ROC curve one needs actual values and predicted values, both are given below.
# These functions will do the groupings on their own (p > 0.2, etc.) like we were doing above
ROCRpred = prediction(predictions,valid$Survived)
# Performance function
ROCRperf = performance(ROCRpred, "tpr", "fpr")

# Plot ROC curve
plot(ROCRperf)

# Add colors
plot(ROCRperf, colorize=TRUE)

# Add threshold labels 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

# model evalaution
library(caret)
predClass <- ifelse(predictions>=0.8,1,0)
confusionMatrix(valid$Survived, predClass)

predClass <- ifelse(predictions>=0.7,1,0)
confusionMatrix(valid$Survived, predClass)

predClass <- ifelse(predictions>=0.6,1,0)
confusionMatrix(valid$Survived, predClass)

# Predictions on test data

# Removing null values

sum(is.na(test))

library(DMwR)
test2<- centralImputation(test) #Cenral Imputation

# test2 <- knnImputation(test,scale=T,k=5) #KNN Imputation
# sum(is.na(test2))

# sum(is.na(test2))

# test2$Pclass <- as.factor(test2$Pclass)

test3 <- test2[,c("Pclass","Sex","Age","SibSp", "Parch", "Fare", "Embarked")]

predictions <- predict(logit,test3,type="response") 
summary(predictions)
head(predictions)

# Baseline Submission

write_to_file(predictions, "logistic")

# CART

# using data train 5


library(rpart)
model_cart <- rpart(Survived~., data=train5, method = "class", control = rpart.control(cp=0.015, minbucket = 5))
summary(model_cart)

library(rpart.plot)
prp(model_cart,varlen=10)
plotcp(model_cart)

library(caret)
pred_train <- predict(model_cart, train5, type='class')
confusionMatrix(pred_train, train5$Survived)

pred_valid <- predict(model_cart, newdata=valid, type = 'class')
confusionMatrix(pred_valid, valid$Survived)

pred_cart <- predict(model_cart, newdata=test3, type = 'class')

# writing to file

write_to_file(pred_cart, "CART")

# Random Forest

library(randomForest)
model_random <- randomForest(Survived~., data = train5, importance = TRUE, ntree = 55)
summary(model_random)
plot(model_random)

importance(model_random)

pred_rf <- predict(model_random, train5, type = 'class')
confusionMatrix(pred_rf, train5$Survived)

pred_valid_rf <- predict(model_random, valid)
confusionMatrix(pred_valid_rf, valid$Survived)

levels(test3$Embarked) <- levels(train5$Embarked)

pred_rft <- predict(model_random, test3)

# Writing to file

write_to_file(pred_rft,"RandomF")

# Support Vector Machines

library(e1071)

model_svm <- svm(Survived~., data = train5, kernel = "radial",  gamma = 0.15)
summary(model_svm)

pred_train_svm <- predict(model_svm, newdata = train5)
confusionMatrix(pred_train_svm, train5$Survived)

pred_valid_svm <- predict(model_svm, newdata = valid)
confusionMatrix(pred_valid_svm, valid$Survived)

pred_test_svm <- predict(model_svm, newdata = test3)

# writing to file

write_to_file(pred_test_svm, "SVM")

# Evolutionary Tree

library(evtree)
set.seed(123)
model_evtree <- evtree(Survived~ ., data = train5)
plot(model_evtree)

pred_train_ev <- predict(model_evtree, newdata = train5)
confusionMatrix(pred_train_ev, train5$Survived)

pred_valid_ev <- predict(model_evtree, newdata = valid)
confusionMatrix(pred_valid_ev, valid$Survived)

pred_test_ev <- predict(model_evtree, newdata = test3)

# writing to file

write_to_file(pred_test_ev, "EVTree")

# Adaboost

library(adabag)

model_ada <- boosting(Survived ~ .,data=train5, 
                                 boos=TRUE, 
                                 mfinal=100)
pred_train_ada <- predict(model_ada, newdata = train5)
confusionMatrix(pred_train_ada$class, train5$Survived)

pred_valid_ada <- predict(model_ada, newdata = valid)
confusionMatrix(pred_valid_ada$class, valid$Survived)

pred_test_ada <- predict(model_ada, newdata = test3)

# Writing to file

write_to_file(pred_test_ada, "Ada")

# Gradient Boosting Machines

library(gbm)
set.seed(1234)


# Disclaimer : GBM needs output variable in numeric format

train6 <- train5
train6$Survived <- as.numeric(train5$Survived)-1
str(train6)

model_GBM=gbm(Survived ~ ., # formula
        data= train6, # dataset
        distribution="bernoulli", # see the help for other choices
        n.trees=1000, # number of trees)
        shrinkage=0.01, # shrinkage or learning rate,
        # 0.001 to 0.1 usually work
        interaction.depth=5, # 1: additive model, 2: two-way interactions, etc.
        bag.fraction = 0.5, # subsampling fraction, 0.5 is probably best
        n.minobsinnode = 10, # minimum total weight needed in each node
        verbose=FALSE) # don't print out progress

summary(model_GBM$fit)
hist(model_GBM$fit)
print(model_GBM)

best.iter <- gbm.perf(model_GBM,method="OOB")
print(best.iter)

pred_train_gbm <- predict(model_GBM,newdata=train5, type = "response",n.trees = 1000)

summary(pred_train_gbm)
hist(pred_train_gbm)


library(ROCR)

ROCRpred = prediction(pred_train_gbm,train5$Survived)
# Performance function
ROCRperf = performance(ROCRpred, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

pred_train_gbm2 <- ifelse(pred_train_gbm>0.5,1,0)
confusionMatrix(pred_train_gbm2, train5$Survived)

pred_valid_gbm <- predict(model_GBM,newdata=valid, type = "response",n.trees = 1000)
pred_valid_gbm2 <- ifelse(pred_valid_gbm>0.5,1,0)
confusionMatrix(pred_valid_gbm2, valid$Survived)

pred_test_gbm <- predict(model_GBM,newdata=test, type = "response",n.trees = 1000)
pred_test_gbm2 <- ifelse(pred_test_gbm>0.5,1,0)

# Writing to file

write_to_file(pred_test_gbm2, "GBM")

# Extreme Gradient Boosting(XGBoost)

# Disclaimer : xgboost doesn't work with categorical variables

train7 <- apply(train5, 2, as.numeric)

library(xgboost)
model_xgboost <- xgboost(data = as.matrix(train6), label = train6$Survived, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")



# Ensemble - 1 -> Logistic, CART and Random Forest

ensemble1 <- data.frame(as.vector(as.factor(pred_log)), as.vector(pred_cart), as.vector(as.factor(pred_rft)))
str(ensemble1)

ensemble1$pred_log <- as.numeric(ensemble1$as.vector.as.factor.pred_log..)
ensemble1$pred_cart <- as.numeric(ensemble1$as.vector.pred_cart.)
ensemble1$pred_rft <- as.numeric(ensemble1$as.vector.as.factor.pred_rft..)
str(ensemble1)

ensemble1[,c(1:3)]<- NULL
str(ensemble1)

ensemble1$pred_log <- ifelse(ensemble1$pred_log == 1, 0, 1)
ensemble1$pred_cart <- ifelse(ensemble1$pred_cart == 1, 0, 1)
ensemble1$pred_rft <- ifelse(ensemble1$pred_rft == 1, 0, 1)

str(ensemble1)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

for(j in 1:418)
{
  ensemble1$final_val[j] <- Mode(ensemble1[j,])
}
str(ensemble1)
ensemble1$pred1_final <- as.numeric(ensemble1$final_val)
ensemble1$final_val <- NULL
str(ensemble1)

# writing output

write_to_file(ensemble1$pred1_final, "Ensemble_log_CART_RF")

# Ensemble - 2 <- Logistic, CART, RF, EvTree, SVM, AdaBoost, GBM

str(pred_log)
str(pred_cart)
str(pred_rft)
str(pred_test_svm)
str(pred_test_ev)
str(pred_test_ada$class)
str(pred_test_gbm2)

ensemble2 <- data.frame(pred_log, as.numeric(pred_cart)-1, 
              as.numeric(pred_rft)-1, as.numeric(pred_test_svm)-1,
              as.numeric(pred_test_ev)-1, as.numeric(pred_test_ada$class),
              pred_test_gbm2)
str(ensemble2)
head(ensemble2)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

for(j in 1:418)
{
  ensemble2$final_val[j] <- Mode(ensemble2[j,])
}
str(ensemble2)
ensemble2$pred_final <- as.numeric(ensemble2$final_val)
ensemble2$final_val <- NULL
str(ensemble2)

write_to_file(ensemble2$pred_final, "Ensemble_2")

# Neural Nets

library(neuralnet)

# Pre-processing

library(dummies)
str(train4)

dummysex <- dummy(train4$Sex)
dummyemb <- dummy(train4$Embarked)
dummypclass <- dummy(as.factor(train4$Pclass))

colnames(dummypclass) <- c("pclass1", "pclass2", "pclass3")

train4_1 <- cbind(train4, dummysex, dummyemb, dummypclass)
str(train4_1)

train4_1$Sex <- NULL
train4_1$Embarked <- NULL
train4_1$Pclass <- NULL

str(train4_1)

# Standardizing numeric values

train4_2 <- train4_1

library(vegan) 
train4_2$Age <- as.numeric(decostand(train4_2$Age, "range"))
train4_2$SibSp <- as.numeric(decostand(train4_2$SibSp, "range"))
train4_2$Parch <- as.numeric(decostand(train4_2$Parch, "range"))
train4_2$Fare <- as.numeric(decostand(train4_2$Fare, "range"))
train4_2$Survived <- as.numeric(train4_2$Survived)-1

str(train4_2)

# Splitting into training and testing

set.seed(123)
library(caTools)
spl <- sample.split(train4_2, 0.7)
train5 <- train4_2[spl==TRUE,]
valid <- train4_2[spl==FALSE,]

# Fitting a neural net

library(neuralnet)
n <- names(train4_2)
f <- as.formula(paste("Survived ~", paste(n[!n %in% "Survived"], collapse = " + ")))
nn <- neuralnet(f,data=train5,hidden=6,linear.output=F)

plot(nn)

pred_nn <- compute(nn, train5[-c(3)])
pred_nn2 <- as.numeric(pred_nn$net.result)
summary(pred_nn2)

library(ROCR)

ROCRpred = prediction(pred_nn2,train5$Survived)
# Performance function
ROCRperf = performance(ROCRpred, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

pred_train_ann <- ifelse(pred_nn2>0.5,1,0)
library(caret)
confusionMatrix(pred_train_ann, train5$Survived)

# Computing on validation set

pred_nn_valid <- compute(nn, valid[-c(3)])
pred_nn2_valid <- as.numeric(pred_nn_valid$net.result)
summary(pred_nn2_valid)

library(ROCR)

ROCRpred = prediction(pred_nn2_valid,valid$Survived)
# Performance function
ROCRperf = performance(ROCRpred, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

pred_valid_ann <- ifelse(pred_nn2_valid>0.5,1,0)
library(caret)
confusionMatrix(pred_valid_ann, valid$Survived)

# Computing on test set

# Test pre-processing

str(test3)

dummysex <- dummy(test3$Sex)
levels(test3$Embarked) <- levels(train4$Embarked)
dummyemb <- dummy(test3$Embarked)
EmbarkedS <- rep(0,418)
dummyemb <- cbind(dummyemb, EmbarkedS)
dummypclass <- dummy(as.factor(test3$Pclass))

colnames(dummypclass) <- c("pclass1", "pclass2", "pclass3")

test3_1 <- cbind(test3, dummysex, dummyemb, dummypclass)
str(test3_1)

test3_1$Sex <- NULL
test3_1$Embarked <- NULL
test3_1$Pclass <- NULL

str(test3_1)

# Standardizing numeric values

test3_2 <- test3_1

library(vegan) 
test3_2$Age <- as.numeric(decostand(test3_2$Age, "range"))
test3_2$SibSp <- as.numeric(decostand(test3_2$SibSp, "range"))
test3_2$Parch <- as.numeric(decostand(test3_2$Parch, "range"))
test3_2$Fare <- as.numeric(decostand(test3_2$Fare, "range"))

str(test3_2)

pred_nn_test <- compute(nn, test3_2)
pred_nn2_test <- as.numeric(pred_nn_test$net.result)
summary(pred_nn2_test)
hist(pred_nn2_test)

pred_nn2_test <- ifelse(pred_nn2_test>0.5,1,0)
table(pred_nn2_test)

# Writing out output

write_to_file(pred_nn2_test, "ANN")

## Algorithms applied:

# Logistic Regression
# Decision Tree(CART)
# Random Forest
# Support Vector Machines(SVM)
# Evolutionary Tree
# Adaptive Boosting(AdaBoost)
# Gradient Boosting Machines(GBM)
# XGBoost
# Artificial Neural Networks
# Ensemble