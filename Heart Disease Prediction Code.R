
#Install Required Packages
pacman::p_load(caTools,dplyr,caret,rpart,ggplot2,ROSE,DMwR,randomForest,e1071)

data<-read.csv("framingham.csv")#input data
data<-data[,-3]
sapply(data, function(x)sum(is.na(x)))#check for missing values
data<-na.omit(data)
str(data)#structure of data
data$TenYearCHD<-as.factor(data$TenYearCHD)#convert target variable to factor

#EDA
data$age_group<-ifelse(data$age>=32 & data$age <42,"32-42",ifelse(data$age>=42 & data$age<49,"42-49",ifelse(data$age >= 49 & data$age <56,"49-56","above 56")))#bin age variable into 4 groups
ggplot(data = data ) + geom_bar(mapping = aes(x = as.factor(male),fill = TenYearCHD),position = "fill")+ggtitle("Gender and Age by TenYearCHD")+facet_wrap(~age_group,nrow = 2)+xlab("Gender")

#Males above 56 years of age are more likely to develop coronary heart disease compared to females above 56.      
ggplot(data = data ) + geom_bar(mapping = aes(x = as.factor(prevalentHyp),fill = TenYearCHD),position = "fill")+ggtitle("Prevalent hypertension and Gender by TenYearCHD")+facet_wrap(~as.factor(male))+xlab("Gender")

ggplot(data = data ) + geom_bar(mapping = aes(x = as.factor(BPMeds),fill = TenYearCHD),position = "fill")+ggtitle("BPMeds and Gender by TenYearCHD")+facet_wrap(~as.factor(male))

ggplot(data = data ) + geom_bar(mapping = aes(x = as.factor(diabetes),fill = TenYearCHD),position = "fill")+ggtitle("Diabetes and Gender by TenYearCHD")+facet_wrap(~as.factor(male))

#Males with prevalent hypertension,on Bp meds and diabetic are more likely to develop CHD.        
ggplot(data = data) + 
  geom_point(mapping = aes(x = sysBP, y = diaBP, color = TenYearCHD))+ggtitle("SysBP and diaBP scatterplot by TenYearCHD")
#Higher sysBP and diaBP,higher possibility of developing CHD.       

#splitting data into test and train 
set.seed(123)
split <- sample.split(data,SplitRatio = 0.8)
train.set <- subset(data,split=="TRUE")
test.set <- subset(data,split=="FALSE")
table(train.set$TenYearCHD)/nrow(train.set)*100#percentage of 0s and 1s
train.set<-train.set[,-16]
test.set<-test.set[,-16]

#Oversampling and Undersampling data 
table(train.set$TenYearCHD)
data_over <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "over",N =5000 )$data #oversampling training data
table(data_over$TenYearCHD) #proportion of 
data_under <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "under", N = 890, seed = 1)$data #undersampling training data
table(data_under$TenYearCHD) #proportion of
data_both <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "both", p=0.5,N=2945,seed = 1)$data #both under and over sampled training data      
table(data_both$TenYearCHD)

#SVM
svm_tune_train <- tune(svm, TenYearCHD~.,data=train.set,ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)),tunecontrol = tune.control( sampling = "cross", cross = 5))#finding the best parameters for svm model using cross validation
svm_tune_train$best.parameters
svm_tune_data_under <- tune(svm, TenYearCHD~.,data=data_under,ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)),tunecontrol = tune.control( sampling = "cross", cross = 5))
svm_tune_data_under$best.parameters
svm_tune_data_over <- tune(svm, TenYearCHD~.,data=data_over,ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)),tunecontrol = tune.control( sampling = "cross", cross = 5))
svm_tune_data_over$best.parameters
svm_tune_data_both <- tune(svm, TenYearCHD~.,data=data_both,ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)),tunecontrol = tune.control( sampling = "cross", cross = 5))
svm_tune_data_both$best.parameters

svm_train<-function(x,y,z)#function to train data with SVM and validate model on training data.
{
  svm_model<-svm(TenYearCHD ~ ., data=x,scaled =TRUE, cost=y, gamma=z)
  pred <- predict(svm_model,x[,-15])
  roc<-roc.curve(x$TenYearCHD,pred)
  con<-confusionMatrix(table(pred, x$TenYearCHD))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

svm_test<-function(x,y,z)#function to train data with SVM and validate model on testing data
{
  svm_model<-svm(TenYearCHD ~ ., data=x,scaled =TRUE, cost=y, gamma=z)
  pred <- predict(svm_model,test.set[,-15])
  roc<-roc.curve(test.set$TenYearCHD,pred)
  con<-confusionMatrix(table(pred, test.set$TenYearCHD))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

svm_train(train.set,0.1,0.5)
svm_train(data_under,1,0.5)
svm_train(data_over,10,2)
svm_train(data_both,10,2)

svm_test(train.set,0.1,0.5)
svm_test(data_under,1,0.5)
svm_test(data_over,10,2)
svm_test(data_both,10,2)

#The support vector machine classification model is build using Radial Basis Function (RBF) kernel. Cross validation and grid search methods are applied in analysing the best parameter values for SVM model. 
#The best SVM model is the model build on under sampled dataset with parameters C =1 and gamma = 0.5 respectively. The model has test accuracy of 59%. Sensitivity of the model is 57% and specificity of the model is 79%.AUC is 64%.       

#Logistic regression
train.set$age<-ifelse(train.set$age>=32 & train.set$age <42,"32-42",ifelse(train.set$age>=42 & train.set$age<49,"42-49",ifelse(train.set$age >= 49 & train.set$age <56,"49-56","above 56")))#bin age variable into 4 groups
test.set$age<-ifelse(test.set$age>=32 & test.set$age <42,"32-42",ifelse(test.set$age>=42 & test.set$age<49,"42-49",ifelse(test.set$age >= 49 & test.set$age <56,"49-56","above 56")))
data_under$age<-ifelse(data_under$age>=32 & data_under$age <42,"32-42",ifelse(data_under$age>=42 & data_under$age<49,"42-49",ifelse(data_under$age >= 49 & data_under$age <56,"49-56","above 56")))
data_over$age<-ifelse(data_over$age>=32 & data_over$age <42,"32-42",ifelse(data_over$age>=42 & data_over$age<49,"42-49",ifelse(data_over$age >= 49 & data_over$age <56,"49-56","above 56")))
data_both$age<-ifelse(data_both$age>=32 & data_both$age <42,"32-42",ifelse(data_both$age>=42 & data_both$age<49,"42-49",ifelse(data_both$age >= 49 & data_both$age <56,"49-56","above 56")))
train.set[,c(1,2,3,5,6,7,8)] <- as.data.frame(sapply(train.set[,c(1,2,3,5,6,7,8)] , factor))
test.set[,c(1,2,3,5,6,7,8)] <- as.data.frame(sapply(test.set[,c(1,2,3,5,6,7,8)] , factor))
data_under[,c(1,2,3,5,6,7,8)] <- as.data.frame(sapply(data_under[,c(1,2,3,5,6,7,8)] , factor))
data_over[,c(1,2,3,5,6,7,8)] <- as.data.frame(sapply(data_over[,c(1,2,3,5,6,7,8)] , factor))
data_both[,c(1,2,3,5,6,7,8)] <- as.data.frame(sapply(data_both[,c(1,2,3,5,6,7,8)] , factor))

log_reg_train<-function(x)#function to train data with logistic regression and validate model on training data
{
  log_reg<-glm(TenYearCHD ~ male + age + sysBP + glucose + cigsPerDay + 
                 heartRate + prevalentHyp, family = "binomial", data = x)
  logit.step.pred <- predict(log_reg, x[,-15], type = "response")
  logit.step.pred[logit.step.pred > 0.5]="1"
  logit.step.pred[logit.step.pred <= 0.5]="0"
  logit.step.pred<-as.factor(logit.step.pred)
  roc<-roc.curve(x$TenYearCHD,logit.step.pred)
  con<-confusionMatrix(table(x$TenYearCHD,logit.step.pred))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
  
}

log_reg_test<-function(x)#function to train data with logistic regression and validate model on testing data
{
  log_reg<-glm(TenYearCHD ~ male + age + sysBP + glucose + cigsPerDay + 
                 heartRate + prevalentHyp, family = "binomial", data = x)
  logit.step.pred <- predict(log_reg, test.set[,-15], type = "response")
  logit.step.pred[logit.step.pred > 0.5]="1"
  logit.step.pred[logit.step.pred <= 0.5]="0"
  logit.step.pred<-as.factor(logit.step.pred)
  roc<-roc.curve(test.set$TenYearCHD,logit.step.pred)
  con<-confusionMatrix(table(test.set$TenYearCHD,logit.step.pred))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

log_reg_train(train.set)
log_reg_train(data_under)
log_reg_train(data_over)
log_reg_train(data_both)

log_reg_test(train.set)
log_reg_test(data_under)
log_reg_test(data_over)
log_reg_test(data_both)

#Feature engineering for naive bayes and random forest model  
data_fact<-read.csv("framingham.csv")
data_fact<-data_fact[,-c(3,4)]
data_fact<-na.omit(data_fact)

quantile(data$age, probs = seq(0, 1, 0.25), na.rm = TRUE)
data_fact$age<-ifelse(data_fact$age>=32 & data_fact$age <42,"32-42",ifelse(data_fact$age>=42 & data_fact$age<49,"42-49",ifelse(data_fact$age >= 49 & data_fact$age <56,"49-56","above 56")))

data_fact$cigsPerDay<-ifelse(data_fact$cigsPerDay==0,"None",ifelse(data_fact$cigsPerDay>20,"High",ifelse(data_fact$cigsPerDay >0 & data_fact$cigsPerDay <= 10,"Low","Medium")))

data_fact$totChol<-ifelse(data_fact$totChol>240,"High",ifelse(data_fact$diaBP<200,"low","Boderline High"))
data_fact$sysBP<-ifelse(data_fact$sysBP>180,"High",ifelse(data_fact$sysBP<140,"low","Boderline High"))
data_fact$diaBP<-ifelse(data_fact$diaBP>120,"High",ifelse(data_fact$diaBP<80,"low","Boderline High"))

data_fact$BMI<-ifelse(data_fact$BMI>30,"Obese",ifelse(data_fact$BMI<18.5,"Underweight",ifelse(data_fact$BMI >= 18.5 & data_fact$BMI <= 24.9,"Healthy","Overweight")))

data_fact$heartRate<-ifelse(data_fact$heartRate>100,"High",ifelse(data_fact$heartRate<60,"low","Normal"))
data_fact$glucose<-ifelse(data_fact$glucose>125,"Diabetic",ifelse(data_fact$glucose<100,"Normal","Pre-Diabetic"))
col_names <- names(data_fact)
data_fact[,col_names] <- lapply(data_fact[,col_names] , factor)

set.seed(123)
split <- sample.split(data_fact,SplitRatio = 0.8)
train.set <- subset(data_fact,split=="TRUE")
test.set <- subset(data_fact,split=="FALSE")

#undersampling and oversampling
table(train.set$TenYearCHD)
data_over <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "over",N =5000 )$data #oversampling training data
table(data_over$TenYearCHD) #proportion of TenYearCHD

data_under <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "under", N = 890, seed = 1)$data #undersampling training data
table(data_under$TenYearCHD) #proportion of TenYearCHD

data_both <- ovun.sample(TenYearCHD ~ ., data = train.set, method = "both", p=0.5,N=2945,seed = 1)$data #both under and over sampled training data      
table(data_both$TenYearCHD)

##Naive Bayes    
NV_train<-function(x)#function to train data with Naive Bayes and validate model on training data
{
  NV_model<-naiveBayes(TenYearCHD ~ .,data=x)
  pred.class <- predict(NV_model,x[,-14])
  roc<-roc.curve(x$TenYearCHD,pred.class)
  con<-confusionMatrix(table(x$TenYearCHD,pred.class))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

NV_test<-function(x)#function to train data with Naive Bayes and validate model on testing data
{
  NV_model<-naiveBayes(TenYearCHD ~ .,data=x)
  pred.class <- predict(NV_model,test.set[,-14])
  roc<-roc.curve(test.set$TenYearCHD,pred.class)
  con<-confusionMatrix(table(test.set$TenYearCHD,pred.class))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

NV_train(train.set)
NV_train(data_over)
NV_train(data_both)
NV_train(data_under)

NV_test(train.set)
NV_test(data_over)
NV_test(data_both)
NV_test(data_under)

#Random Forest Model  
rf_train<-function(x)#function to train data with Random Forest and validate model on training data.
{
  rf = randomForest(TenYearCHD~.,ntree = 1000,data = x)
  rf.pred.tree<-predict(rf, newdata = x[,-14],type="class")
  roc<-roc.curve(x$TenYearCHD,rf.pred.tree)
  con<-confusionMatrix(table(rf.pred.tree,x$TenYearCHD))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

rf_test<-function(x)#function to train data with Random Forest and validate model on testing data.
{
  rf = randomForest(TenYearCHD~.,ntree = 1000,data = x)
  rf.pred.tree<-predict(rf, newdata = test.set[,-14],type="class")
  roc<-roc.curve(test.set$TenYearCHD,rf.pred.tree)
  con<-confusionMatrix(table(rf.pred.tree,test.set$TenYearCHD))
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}

rf_train(train.set)
rf_train(data_over)
rf_train(data_both)
rf_train(data_under)

rf_test(train.set)
rf_test(data_over)
rf_test(data_both)
rf_test(data_under)

rf = randomForest(TenYearCHD~.,ntree = 1000,data = data_both)   
varImpPlot(rf)#Variable importance plot   

