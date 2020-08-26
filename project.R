##**Install required packages**.         
{r loadPackages, message=FALSE, warning=FALSE, results='hide'}
#if(!require("pacman")) install.packages("pacman")
pacman::p_load(caTools,dplyr,caret,rpart,ggplot2,ROSE,DMwR,randomForest,e1071)


data<-read.csv("framingham.csv")
sapply(data,function(x)sum(is.na(x)))
data<-na.omit(data)
str(data)
data[,c(1,4,6,7,8,9,16)] <- lapply(data[,c(1,4,6,7,8,9,16)],factor)

data_num<-data[,c(2,3,5,10:15)]
plot(data_num)
cor(data_num) #sysBPand diaBP are highly correlated

#summary statistics of categorical variables.
table(data$TenYearCHD)/nrow(data)*100
#We can see that the dataset is unbalanced,the positive Ten Year CHD cases account for only 15% of total cases.     

#Using the following methods to treat the imbalance in the dataset:      
#1.Undersampling : Reduce the observations in the majority class to make the dataset balanced.    
#2.Oversampling : Replicate observations in minority class to balance the dataset.      
#3.Synthetic data generation : Used to overcome imbalance by creating artificial data.We will use SMOTE algorithm to create artificial data. 

table(data$male)/nrow(data)*100
#cross classification
ggplot(data = data ) + geom_bar(mapping = aes(x = male,fill = TenYearCHD))+ggtitle("Gender by TenYearCHD")
ggplot(data = data ) + geom_bar(mapping = aes(x = currentSmoker,fill = TenYearCHD))+ggtitle("CurrentSmoker by TenYearCHD")
ggplot(data = data ) + geom_bar(mapping = aes(x = prevalentHyp,fill = TenYearCHD))+ggtitle("Prevalent hypertension by TenYearCHD")
ggplot(data = data ) + geom_bar(mapping = aes(x = BPMeds ,fill = TenYearCHD))+ggtitle("BPMeds by TenYearCHD")
ggplot(data = data ) + geom_bar(mapping = aes(x = diabetes ,fill = TenYearCHD))+ggtitle("diabetes by TenYearCHD")

ggplot(data=data,mapping= aes(y=totChol))+geom_boxplot()
ggplot(data=data,mapping= aes(y=sysBP))+geom_boxplot()
data<-data%>%filter(totChol<500,sysBP<250)

split <- sample.split(data,SplitRatio = 0.8)
train.set <- subset(data,split=="TRUE")
test.set <- subset(data,split=="FALSE")

log_reg<-function(x)#function to train data with logistic regression and validate model on test data.
{
  logit.reg <- glm(TenYearCHD ~ ., data = train.set, family = "binomial") 
  logit.reg.pred <- predict(logit.reg, test.set[,-16], type = "response")
  logit.reg.pred[logit.reg.pred > 0.5]="1"
  logit.reg.pred[logit.reg.pred <= 0.5]="0"
  logit.reg.pred<-as.factor(logit.reg.pred)
  con<-confusionMatrix(table(test.set$TenYearCHD,logit.reg.pred))
  roc<-roc.curve(test.set$TenYearCHD, logit.reg.pred)
  listret<-list(roc,con)#Return confusion matrix and ROC curve
  return(listret)
}
#Our primary goal is to identify risk.
#As missclassifying risk as not a risk will
#have a higher negative impact than misclassifying not a risk as risk.
#Hence we want higher specificity.

#best subset selection
smallest <- TenYearCHD ~ 1
biggest <- TenYearCHD ~ male+age+education+currentSmoker+cigsPerDay+BPMeds+prevalentStroke+prevalentHyp+diabetes+totChol+sysBP+diaBP+BMI+heartRate+glucose 
m <- glm(TenYearCHD ~ male, data=train.set,family = "binomial")
stats::step(m, scope=list(lower=smallest, upper=biggest)) 
log_step<-glm(TenYearCHD ~ male + age + sysBP + glucose + cigsPerDay + 
                totChol + prevalentStroke, family = "binomial", data = train.set)
logit.reg.pred <- predict(log_step, test.set[,-16], type = "response")
logit.reg.pred[logit.reg.pred > 0.5]="1"
logit.reg.pred[logit.reg.pred <= 0.5]="0"
logit.reg.pred<-as.factor(logit.reg.pred)
con<-confusionMatrix(table(test.set$TenYearCHD,logit.reg.pred))
##`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
#SVM

### Linear SVM using cross-validated data (using trControl option) 
data_both$TenYearCHD<-as.factor(data_both$TenYearCHD)
svm_radial <- train(TenYearCHD ~., data = data_under, 
              method = "svmLinear",
              trControl=trainControl(method = "cv", 
                                     number = 10),
              preProcess = c("center", "scale"),
              tuneLength = 10)

svm_radial

# Generate predictions
pred <- predict(svm_model, data_both[,-14])
pred

# Performance evaluation - confusion matrix
confusionMatrix(table(pred,data_both$TenYearCHD))


### Hyperparameter Optimization (Grid Search) 
grid <- expand.grid(C=seq(0,2.5,0.1))
svm_grid <- train(TenYearCHD ~., data = data_both, 
                   method = "svmLinear",
                   trControl=trainControl(method ="cv", 
                                          number = 10),
                   preProcess = c("center", "scale"),
                   tuneGrid = grid)

svm_grid

# Plot the grid
plot(svm_grid)

# Generate predictions
pred2 <- predict(svm_grid,test.set[,-14])
pred2

# Performance evaluation - confusion matrix
confusionMatrix(table(pred2, test.set$TenYearCHD))

##```````````````````````````````````````````````````````````````````````````````````````````````

gammalist <- c(0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05)
tune<- tune.svm(TenYearCHD ~., data=train.set,gamma =seq(.01, 0.1, by = .01), cost = seq(1,1000, by = 5))
summary(tune.out)
summary(tune.out$best.model)
svm_tune <- tune(svm, TenYearCHD~.,data=train.set,scaled =TRUE,ranges=list(cost=10^(-1:2), gamma=gammalist))
svm_model<-svm(TenYearCHD ~ ., data=train.set,scaled =TRUE, cost=100, gamma=0.03)

tune.out <- tune(svm, TenYearCHD~., data=train.set,decision.values =TRUE,scaled =TRUE, ranges=list(cost=2^(-1:5),gamma =seq(.01, 0.1, by = .01),tunecontrol = tune.control( sampling = "cross", cross = 5)))
##``````````````````````````````````````````````````````````````````````````````````````````````````

log_step<-glm(TenYearCHD ~ male + age + sysBP + glucose + cigsPerDay + 
                heartRate + prevalentHyp, family = "binomial", data = data_under)
##summary(log_step)
logit.step.pred <- predict(log_step, data_under[,-14], type = "response")
logit.step.pred[logit.step.pred > 0.5]="1"
logit.step.pred[logit.step.pred <= 0.5]="0"
logit.step.pred<-as.factor(logit.step.pred)
roc.curve(test.set$TenYearCHD,logit.step.pred)
confusionMatrix(table(data_under$TenYearCHD,logit.step.pred))

##```````````````````````````````````````````````````````````````````````````````````````````````````
data[,c(1,4,5,6,7)] <- as.data.frame(sapply(data[,c(1,4,5,6,7,14)] , factor))
data_over$TenYearCHD<-as.factor(data_over$TenYearCHD)
test.set[,c(1,4,5,6,7)] <- lapply(test.set[,c(1,4,5,6,7)] , factor)
test.set$TenYearCHD<-as.factor(test.set$TenYearCHD)


rf_under = randomForest(TenYearCHD~.,  
                        ntree = 500,
                        data = train.set)
rf.pred.tree<-predict(rf_under, newdata = test.set[,-14],type="class")
roc.curve(train.set$TenYearCHD,rf.pred.tree)
confusionMatrix(table(rf.pred.tree,test.set$TenYearCHD))
#`````````````````````````````````````````````````````````````````````````````````````````````````````
lapply(data_over[,c(2,3,8,9,10,11,12,13)],function(x)hist(x))

##Naive bayes
data1<-data
data1$age <- cut(data1$age, 
                   +                      breaks=quantile(data$age, probs = seq(0, 1, 0.25), na.rm = TRUE), 
                   +                      labels=c("low","middle","high","very high"))

data1$cigsPerDay<-ifelse(data1$cigsPerDay==0,"None",ifelse(data1$cigsPerDay>20,"High",ifelse(data1$cigsPerDay >0 & data1$cigsPerDay <= 10,"Low","Medium")))

data1$totChol<-ifelse(data1$totChol>240,"High",ifelse(data1$diaBP<200,"low","Boderline High"))
data1$sysBP<-ifelse(data1$sysBP>180,"High",ifelse(data1$sysBP<140,"low","Boderline High"))
data1$diaBP<-ifelse(data1$diaBP>120,"High",ifelse(data1$diaBP<80,"low","Boderline High"))

data1$BMI<-ifelse(data1$BMI>30,"Obese",ifelse(data1$BMI<18.5,"Underweight",ifelse(data1$BMI >= 18.5 & data1$BMI <= 24.9,"Healthy","Overweight")))

data1$heartRate<-ifelse(data1$heartRate>100,"High",ifelse(data1$heartRate<60,"low","Normal"))
data1$glucose<-ifelse(data1$glucose>125,"Diabetic",ifelse(data1$glucose<100,"Normal","Pre-Diabetic"))

data1<-as.data.frame(data1)
NV_model<-naiveBayes(TenYearCHD ~ .,data=train.set)
pred.class <- predict(NV_model,test.set[,-14])
confusionMatrix(table(test.set$TenYearCHD,pred.class))
#````````````````````````````````````````````````````````````````````````````````````````````````````````
data[,c(1,4,5,6,7)] <- as.data.frame(sapply(data[,c(1,4,5,6,7)] , factor))
