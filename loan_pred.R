
#____________________________________ INSTALLING DEPENDENCIES AND IMPORTING RAW DATA __________________________#

#load dependencies
libraries<-c("ggplot2","randomForest","gbm","caret","AzureML")
sapply(libraries,require,character.only = TRUE)

#reading data from csv
train<- read.csv("F:\\AnalyticsVidya\\loanPrediction\\data\\train.csv",header = TRUE)
test<-read.csv("F:\\AnalyticsVidya\\loanPrediction\\data\\test.csv",header=TRUE)

#_______________________________________________________ EDA ____________________________________________________#

#Creating 'Loan_Status' feature in test dataset so that we can bind train and test dataset by row
test$Loan_Status<-"N"
test$Loan_Status<-as.factor(test$Loan_Status)
levels(test$Loan_Status)<-c("N","Y")


#Combining Train and Test dataset
full_data<-rbind(train,test)
summary(full_data)

#train and Test Index
train_idx<- 1:614
test_idx<- 615:981

#EDA on full data frame

#As we can see from summary of full data "" is present is some feature as a level.So we have to manipulate these fields.
full_data$Gender[full_data$Gender==""]<-NA
full_data$Self_Employed[full_data$Self_Employed==""]<-NA
full_data$Married[full_data$Married==""]<-NA
full_data$Dependents[full_data$Dependents==""]<-NA

#some features which should be factor are present in the dataset as integer.We have to convert those features in facotrs
full_data$Credit_History<-as.factor(full_data$Credit_History)

#loan amount term is a imbalanced class.So we can perform bucketization
table(full_data$Loan_Amount_Term)
full_data$Loan_Amount_Term<-ceiling(full_data$Loan_Amount_Term/180)
#full_data$Loan_Amount_Term<-ifelse(full_data$Loan_Amount_Term<360,1,2)
full_data$Loan_Amount_Term<-as.factor(full_data$Loan_Amount_Term)

full_data<-droplevels(full_data)

#Loan id doesn't help in prediction
full_data<-full_data[,-1]


#Checking variable distribution.There are some large valued outliers in Loan amount.
boxplot(full_data$LoanAmount)
rm_loanamount_idx<-which(full_data$LoanAmount[train_idx]>400)
ggplot(data=full_data[train_idx,],aes(x=LoanAmount,fill=Loan_Status))+ geom_density(position = "fill")
#from above plot we can infer that loan amount only is not thedeciding factor for eligibility for the loan

#Checking variable distribution.There are some large valued outliers in Applicant Income.
boxplot(full_data$ApplicantIncome)
rm_ApplicantIncome_idx<-which(full_data$ApplicantIncome[train_idx]>20000)
#ggplot(data=full_data[train_idx,],aes(x=ApplicantIncome,fill=Loan_Status))+ geom_density(position = "fill")

#Checking variable distribution.There are some large valued outliers in Co Applicant Income
boxplot(full_data$CoapplicantIncome)
rm_CoApplicantIncome_idx<-which(full_data$CoapplicantIncome[train_idx]>10000)
#ggplot(data=full_data[train_idx,],aes(x=CoapplicantIncome,fill=Loan_Status))+ geom_density(position = "fill")

#removing unknown marraige status observations
rm_married_idx<-which(is.na(full_data$Married[train_idx]))

#Creting a vector of incices which has to be removed
rm_index<-unique(c(rm_loanamount_idx,rm_ApplicantIncome_idx,rm_CoApplicantIncome_idx,rm_married_idx))

#removing outliers and missing values
full_data<-full_data[-rm_index,]

train_idx<-1:586
test_idx<-587:953

#Manipulating missing gender field
gender.idx<-which(is.na(full_data$Gender))
gender.model<-randomForest(Gender~.,data=full_data[-gender.idx,c(1,2,4,6,7,11)],ntree=500)
gender.pred<-predict(gender.model,full_data[gender.idx,c(1,2,4,6,7,11)])
full_data$Gender[gender.idx]<-gender.pred

#Manipulating missing Dependents field
dependents.idx<-which(is.na(full_data$Dependents))
dependents.model<-randomForest(Dependents~.,data=full_data[-dependents.idx,c(1,2,3,4,6,7,11)],ntree=500)
dependents.pred<-predict(dependents.model,full_data[dependents.idx,c(1,2,3,4,6,7,11)])
full_data$Dependents[dependents.idx]<-dependents.pred

#Manipulating missing Self Employed field
selfemployed.idx<-which(is.na(full_data$Self_Employed))
selfemployed.model<-randomForest(Self_Employed~.,data=full_data[-selfemployed.idx,c(1,2,3,4,5,6,7,11)],ntree=500)
selfemployed.pred<-predict(selfemployed.model,full_data[selfemployed.idx,c(1,2,3,4,5,6,7,11)])
full_data$Self_Employed[selfemployed.idx]<-selfemployed.pred

#Manipulating missing Credit History field
credithistory.idx<-which(is.na(full_data$Credit_History))
credithistory.model<-randomForest(Credit_History~.,data=full_data[-credithistory.idx,c(1,2,3,4,5,6,7,10,11)],ntree=500)
credithistory.pred<-predict(credithistory.model,full_data[credithistory.idx,c(1,2,3,4,5,6,7,10,11)])
full_data$Credit_History[credithistory.idx]<-credithistory.pred

#Manipulating missing loan term field
loanterm.idx<-which(is.na(full_data$Loan_Amount_Term))
loanterm.model<-randomForest(Loan_Amount_Term~.,data=full_data[-loanterm.idx,c(1,2,3,4,5,6,7,9,10,11)],ntree=500)
loanterm.pred<-predict(loanterm.model,full_data[loanterm.idx,c(1,2,3,4,5,6,7,9,10,11)])
full_data$Loan_Amount_Term[loanterm.idx]<-loanterm.pred

#Manipulating missing loan amount field
loanamount.idx<-which(is.na(full_data$LoanAmount))
loanamount.model<-lm(LoanAmount~poly(ApplicantIncome,2)+poly(CoapplicantIncome,3)
                     +Loan_Amount_Term,data=full_data[-loanamount.idx,-12])
loanamount.pred<-predict(loanamount.model,full_data[loanamount.idx,-12])
full_data$LoanAmount[loanamount.idx]<-loanamount.pred


#_______________________________________________________TRAIN MODEL________________________________________________________________#

#Creating a model using gradient boosting
gbmGrid <-  expand.grid(interaction.depth = c(1, 2,3),
                        n.trees = c(50,100,250,400,500,1000), 
                        shrinkage = c(.01,0.025,.05, .1),
                        n.minobsinnode = 10
                        ) 
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats =10)
set.seed(825)
gbm.fit = train(Loan_Status~., data = full_data[train_idx,], 
                 method = "gbm", 
                 tuneGrid = gbmGrid,
                 trControl = fitControl,
                 verbose = TRUE)

trellis.par.set(caretTheme())
plot(gbm.fit)


#_______________________________________________ BATCH PREDICTION_______________________________________________________________#

output<-cbind(test[1],Loan_Status=predict(gbm.fit,full_data[test_idx,]))
write.csv(output,"Sample_Submission.csv",row.names = FALSE)

#_____________________________________________ FOR WEB DEPLOY ONLY _____________________________________________________#


app_pred <- function(new_data){
  require(gbm)
  predict(gbm.fit,new_data,type="response")
}


#_____________________________________________ AUTHENTICATION AND WEB SERVICE ___________________________________________________#

wsAuth <- #__ YOUR WORKSPACE ID ___#
wsID <- #____ YOUR AUTHENTICATION KEY ___#
wsObj <- workspace(wsID,wsAuth)
final_input_schema <- temp_data[,-12]
myWebService<-publishWebService(ws = wsObj,fun = app_pred ,name = 'loan_application_validator',
                                inputSchema = final_input_schema,outputSchema = temp_data[,12],
                                .retry = 3)

