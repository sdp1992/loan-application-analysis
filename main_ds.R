#install packages
install.packages("ggplot2","gbm","e1071")

#load dependencies
libraries<-c("ggplot2","gbm","e1071","MASS")
lapply(libraries,require,character.only = TRUE)

#reading data from csv
train<- read.csv("F:\\AnalyticsVidya\\loanPrediction\\data\\train.csv",header = TRUE)
test<-read.csv("F:\\AnalyticsVidya\\loanPrediction\\data\\test.csv",header=TRUE)


#--------------------------------------------------------Train and Test Data-------------------------------------------------------#

#Creating 'Loan_Status' feature in test dataset
test$Loan_Status<-"N"
test$Loan_Status<-as.factor(test$Loan_Status)
levels(test$Loan_Status)<-c("N","Y")


#Combining Train and Test dataset
full_data<-rbind(train,test)

#train and Test Index
train_idx<- 1:614
test_idx<- 615:981

#EDA on full data frame
full_data$Gender[full_data$Gender==""]<-NA
full_data$Self_Employed[full_data$Self_Employed==""]<-NA
full_data$Married[full_data$Married==""]<-NA
full_data$Dependents[full_data$Dependents==""]<-NA


full_data$Loan_Amount_Term<-as.factor(full_data$Loan_Amount_Term)
full_data$Credit_History<-as.factor(full_data$Credit_History)
full_data<-droplevels(full_data)
full_data<-full_data[,-1]


# #Checking variable distribution
# boxplot(full_data$LoanAmount[train_idx])
# rm_loanamount_idx<-which(full_data$LoanAmount[train_idx]>420)
# 
# boxplot(full_data$ApplicantIncome[train_idx])
# rm_ApplicantIncome_idx<-which(full_data$ApplicantIncome[train_idx]>25000)
# 
# boxplot(full_data$CoapplicantIncome[train_idx])
# rm_CoApplicantIncome_idx<-which(full_data$CoapplicantIncome[train_idx]>10000)
# 
# rm_married_idx<-which(is.na(full_data$Married[train_idx]))
# 
# #removing outliers and missing values
# rm_unique_idx<-unique(c(rm_ApplicantIncome_idx,rm_CoApplicantIncome_idx,rm_loanamount_idx,rm_married_idx))
# length(rm_unique_idx)
# full_data<-full_data[-rm_unique_idx,]

train_idx<-1:589
test_idx<-590:956


#manipulating missing gender field
gender.idx<-which(is.na(full_data$Gender))
gender.model<-randomForest(Gender~.,data=full_data[-gender.idx,c(1,2,4,6,7,11)],ntree=500)
gender.pred<-predict(gender.model,full_data[gender.idx,c(1,2,4,6,7,11)])
full_data$Gender[gender.idx]<-gender.pred

dependents.idx<-which(is.na(full_data$Dependents))
dependents.model<-randomForest(Dependents~.,data=full_data[-dependents.idx,c(1,2,3,4,6,7,11)],ntree=500)
dependents.pred<-predict(dependents.model,full_data[dependents.idx,c(1,2,3,4,6,7,11)])
full_data$Dependents[dependents.idx]<-dependents.pred

selfemployed.idx<-which(is.na(full_data$Self_Employed))
selfemployed.model<-randomForest(Self_Employed~.,data=full_data[-selfemployed.idx,c(1,2,3,4,5,6,7,11)],ntree=500)
selfemployed.pred<-predict(selfemployed.model,full_data[selfemployed.idx,c(1,2,3,4,5,6,7,11)])
full_data$Self_Employed[selfemployed.idx]<-selfemployed.pred

credithistory.idx<-which(is.na(full_data$Credit_History))
credithistory.model<-randomForest(Credit_History~.,data=full_data[-credithistory.idx,c(1,2,3,4,5,6,7,10,11)],ntree=500)
credithistory.pred<-predict(credithistory.model,full_data[credithistory.idx,c(1,2,3,4,5,6,7,10,11)])
full_data$Credit_History[credithistory.idx]<-credithistory.pred

loanterm.idx<-which(is.na(full_data$Loan_Amount_Term))
loanterm.model<-randomForest(Loan_Amount_Term~.,data=full_data[-loanterm.idx,c(1,2,3,4,5,6,7,9,10,11)],ntree=500)
loanterm.pred<-predict(loanterm.model,full_data[loanterm.idx,c(1,2,3,4,5,6,7,9,10,11)])
full_data$Loan_Amount_Term[loanterm.idx]<-loanterm.pred

loanamount.idx<-which(is.na(full_data$LoanAmount))
loanamount.model<-lm(LoanAmount~ApplicantIncome+CoapplicantIncome+Education+Married,data=full_data[-loanamount.idx,-12])
loanamount.pred<-predict(loanamount.model,full_data[loanamount.idx,-12])
full_data$LoanAmount[loanamount.idx]<-loanamount.pred
quantile(full_data$LoanAmount)


#-------------   -------------------------------------------Prediction-------------------------------------------------------#

#RandomForest
rf.loan<-randomForest(Loan_Status~.,
                      data=full_data[train_idx,],importance=TRUE,ntree=500)
rf.loan
importance(rf.loan)
varImpPlot(rf.loan)


output<-cbind(test[1],Loan_Status=predict(rf.loan,full_data[test_idx,]))
write.csv(output,"Sample_Submission.csv",row.names = FALSE)


#bagging
bag.loan =randomForest(Loan_Status~.,data=train_data_df_predictor ,
                         mtry=7, importance =TRUE)
output<-cbind(test_data[1],Loan_Status=predict(bag.loan,test_data_df))
write.csv(output,"Sample_Submission.csv",row.names = FALSE)

#Boosting
full_data_temp<-full_data
full_data_temp$Loan_Status<-ifelse(full_data_temp$Loan_Status=="Y",1,0)
full_data_temp$Loan_Status<-as.factor(full_data_temp$Loan_Status)

boost.loan =gbm(Loan_Status~.,data=full_data_temp[train_idx,],distribution= "bernoulli",n.trees =5000, interaction.depth =2,shrinkage =0.002)

output<-cbind(test[1],Loan_Status=predict(boost.loan,full_data_temp[test_idx,],n.trees = 2000))

#SVM
svm.loan<-tune(svm,Loan_Status~.,
                      data=full_data[train_idx,],kernal="linear",ranges = list(cost=c(1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2,2.5,3)))

pred.loan_Status<-predict(svm.loan$best.model,full_data[test_idx,])
output<-cbind(test[1],Loan_Status=pred.loan_Status)
write.csv(output,"Sample_Submission.csv",row.names = FALSE)

