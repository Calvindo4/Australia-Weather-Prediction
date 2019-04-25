suppressPackageStartupMessages(library(knitr))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(gmodels))
suppressPackageStartupMessages(library(lattice))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(ROCR))
suppressPackageStartupMessages(library(corrplot))

set.seed(1023)
weather_data <- read.csv("D:\\The Vault\\College Materials\\Semester 5\\Operations Research\\Project\\weatherAUS.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
kable(head(weather_data))

colnames(weather_data)

str(weather_data)

(n <- nrow(weather_data))
c(as.character(weather_data$Date[1]), as.character(weather_data$Date[n]))

all.equal(weather_data$RISK_MM > 1, weather_data$RainTomorrow == "Yes")
all.equal(weather_data$Rainfall > 1, weather_data$RainToday == "Yes")

weather_data2 <- subset(weather_data, select = -c(Date, Location, RISK_MM, Rainfall, RainToday))
colnames(weather_data2)

(cols_withNa <- apply(weather_data2, 2, function(x) sum(is.na(x))))

weather_data3 <- weather_data2[complete.cases(weather_data2),]

factor_vars <- names(which(sapply(weather_data3, class) == "factor"))
factor_vars <- setdiff(factor_vars, "RainTomorrow")
chisq_test_res <- lapply(factor_vars, function(x) { 
  chisq.test(weather_data3[,x], weather_data3[, "RainTomorrow"], simulate.p.value = TRUE)
})
names(chisq_test_res) <- factor_vars
chisq_test_res

barchart_res <- lapply(factor_vars, function(x) { 
  title <- colnames(weather_data3[,x, drop=FALSE])
  wgd <- CrossTable(weather_data3[,x], weather_data3$RainTomorrow, prop.chisq=F)
  barchart(wgd$prop.row, stack=F, auto.key=list(rectangles = TRUE, space = "top", title = title))
})
names(barchart_res) <- factor_vars
barchart_res$WindGustDir

barchart_res$WindDir9am
barchart_res$WindDir3pm

weather_data4 <- subset(weather_data2, select = -c(WindDir9am, WindDir3pm))
weather_data5 <- weather_data4[complete.cases(weather_data4),]
colnames(weather_data5)

factor_vars <- names(which(sapply(weather_data5, class) == "factor"))
numeric_vars <- setdiff(colnames(weather_data5), factor_vars)
numeric_vars <- setdiff(numeric_vars, "RainTomorrow")
numeric_vars
numeric_vars_mat <- as.matrix(weather_data5[, numeric_vars, drop=FALSE])
numeric_vars_cor <- cor(numeric_vars_mat)
corrplot(numeric_vars_cor)

pairs(weather_data5[,numeric_vars], col=weather_data5$RainTomorrow)


nrow(weather_data5)
sum(weather_data5["RainTomorrow"] == "Yes")
sum(weather_data5["RainTomorrow"] == "No")

train_rec <- createDataPartition(weather_data5$RainTomorrow, p = 0.7, list = FALSE)
training <- weather_data5[train_rec,]
testing <- weather_data5[-train_rec,]

sum(training["RainTomorrow"] == "Yes")/sum(training["RainTomorrow"] == "No")
sum(testing["RainTomorrow"] == "Yes")/sum(testing["RainTomorrow"] == "No")

trControl <- trainControl(method = "repeatedcv",  repeats = 5, number = 10, verboseIter = FALSE)

predictors_9am_c1 <- c("Cloud9am",  "Humidity9am", "Pressure9am", "Temp9am")
formula_9am_c1 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c1, collapse="+"), sep="~"))
mod9am_c1_fit <- train(formula_9am_c1,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod9am_c1_fit$results$Accuracy

(summary_rep <- summary(mod9am_c1_fit$finalModel))
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)
drop1(mod9am_c1_fit$finalModel, test="Chisq")

predictors_9am_c2 <- c("Cloud9am",  "Humidity9am", "Pressure9am", "MinTemp")
formula_9am_c2 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c2, collapse="+"), sep="~"))
mod9am_c2_fit <- train(formula_9am_c2,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod9am_c2_fit$results$Accuracy
(summary_rep <- summary(mod9am_c2_fit$finalModel))
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)
mod9am_pred <- predict(mod9am_c1_fit, testing)
confusionMatrix(mod9am_pred, testing[,"RainTomorrow"])

predictors_3pm_c1 <- c("Cloud3pm", "Humidity3pm", "Pressure3pm", "Temp3pm")
formula_3pm_c1 <- as.formula(paste("RainTomorrow", paste(predictors_3pm_c1, collapse="+"), sep="~"))
mod3pm_c1_fit <- train(formula_3pm_c1,  data = training, method = "glm", family = "binomial",
                       trControl = trControl, metric = 'Accuracy')
mod3pm_c1_fit$results$Accuracy
(summary_rep <- summary(mod3pm_c1_fit$finalModel))
drop1(mod3pm_c1_fit$finalModel, test="Chisq")
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)
mod3pm_pred <- predict(mod3pm_c1_fit, testing)
confusionMatrix(mod3pm_pred, testing[,"RainTomorrow"])

predictors_evening_c1 <- c("Pressure3pm", "Temp3pm", "Sunshine")
formula_evening_c1 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c1, collapse="+"), sep="~"))
mod_ev_c1_fit <- train(formula_evening_c1,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c1_fit$results$Accuracy
(summary_rep <- summary(mod_ev_c1_fit$finalModel))
drop1(mod_ev_c1_fit$finalModel, test="Chisq")
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

predictors_evening_c2 <- c(predictors_3pm_c1, "WindGustDir", "WindGustSpeed")
formula_evening_c2 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c2, collapse="+"), sep="~"))
mod_ev_c2_fit <- train(formula_evening_c2,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c2_fit$results$Accuracy
(summary_rep <- summary(mod_ev_c2_fit$finalModel))
drop1(mod_ev_c2_fit$finalModel, test="Chisq")
predictors_evening_c2 <- c(predictors_3pm_c1, "WindGustDir")
formula_evening_c2 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c2, collapse="+"), sep="~"))
mod_ev_c2_fit <- train(formula_evening_c2,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c2_fit$results$Accuracy
(summary_rep <- summary(mod_ev_c2_fit$finalModel))
drop1(mod_ev_c2_fit$finalModel, test="Chisq")
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

predictors_evening_c3 <- c("Pressure3pm", "Sunshine")
formula_evening_c3 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c3, collapse="+"), sep="~"))
mod_ev_c3_fit <- train(formula_evening_c3,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c3_fit$results$Accuracy
(summary_rep <- summary(mod_ev_c3_fit$finalModel))
drop1(mod_ev_c3_fit$finalModel, test="Chisq")
1 - pchisq(summary_rep$deviance, summary_rep$df.residual)
anova(mod_ev_c2_fit$finalModel, mod_ev_c3_fit$finalModel, test="Chisq")
mse <- function(mod_ev_c3_fit)
mean(mod_ev_c3_fit$residuals^2)
modevening_pred <- predict(mod_ev_c2_fit, testing)
confusionMatrix(modevening_pred, testing[,"RainTomorrow"])

modevening_pred <- predict(mod_ev_c3_fit, testing)
confusionMatrix(modevening_pred, testing[,"RainTomorrow"])

mod9am_c1_fit: RainTomorrow ~ Cloud9am + Humidity9am + Pressure9am + Temp9am
mod3pm_c1_fit: RainTomorrow ~ Cloud3pm + Humidity3pm + Pressure3pm + Temp3pm
mod_ev_c2_fit: RainTomorrow ~ Cloud3pm + Humidity3pm + Pressure3pm + Temp3pm + WindGustDir
mod_ev_c3_fit: RainTomorrow ~ Pressure3pm + Sunshine

weather_data6 <- subset(weather_data, select = -c(Date, Location, RISK_MM, RainToday, WindDir9am, WindDir3pm))
weather_data6$RainfallTomorrow <- c(weather_data6$Rainfall[2:nrow(weather_data6)], NA)
weather_data6$Humidity3pmTomorrow <- c(weather_data6$Humidity3pm[2:nrow(weather_data6)], NA)
weather_data6$WindGustSpeedTomorrow <- c(weather_data6$WindGustSpeed[2:nrow(weather_data6)], NA)
weather_data6$SunshineTomorrow <- c(weather_data6$Sunshine[2:nrow(weather_data6)], NA)
weather_data6$MinTempTomorrow <- c(weather_data6$MinTemp[2:nrow(weather_data6)], NA)
weather_data6$MaxTempTomorrow <- c(weather_data6$MaxTemp[2:nrow(weather_data6)], NA)

weather_data7 = weather_data6[complete.cases(weather_data6),]
head(weather_data7)

hr_idx = which(weather_data7$RainfallTomorrow > 15)
(train_hr <- hr_idx[hr_idx %in% train_rec])
(test_hr <- hr_idx[!(hr_idx %in% train_rec)])

rain_test <- weather_data7[test_hr,]
rain_test

opt_cutoff <- 0.42
pred_test <- predict(mod_ev_c2_fit, rain_test, type="prob")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
data.frame(prediction = prediction, RainfallTomorrow = rain_test$RainfallTomorrow)


opt_cutoff <- 0.56
pred_test <- predict(mod_ev_c3_fit, rain_test, type="prob")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
data.frame(prediction = prediction, RainfallTomorrow = rain_test$RainfallTomorrow)

chance_of_rain <- function(model, data_record){
  chance_frac <- predict(mod_ev_c3_fit, data_record, type="prob")[,"Yes"]
  paste(round(chance_frac*100),"%",sep="")
}

chance_of_rain(mod_ev_c3_fit, testing[1:10,])

weather_data8 = weather_data7[weather_data7$RainfallTomorrow > 1,]
rf_fit <- lm(RainfallTomorrow ~ MaxTemp + Sunshine + WindGustSpeed -1, data = weather_data8)
summary(rf_fit)

lm_pred <- predict(rf_fit, weather_data8)
plot(x = seq_along(weather_data8$RainfallTomorrow), y = weather_data8$RainfallTomorrow, type='p', xlab = "observations", ylab = "RainfallTomorrow")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = seq_along(weather_data8$RainfallTomorrow), y = lm_pred, col='red')

sun_fit <- lm(SunshineTomorrow ~ Sunshine*Humidity3pm + Cloud3pm + Evaporation + I(Evaporation^2) + WindGustSpeed - 1, data = weather_data7)
summary(sun_fit)

lm_pred <- predict(sun_fit, weather_data7)
plot(x = seq_along(weather_data7$SunshineTomorrow), y = weather_data7$SunshineTomorrow, type='p', xlab = "observations", ylab = "SunshineTomorrow")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = seq_along(weather_data7$SunshineTomorrow), y = lm_pred, col='red')

cloud9am_fit <- lm(Cloud9am ~ Sunshine, data = weather_data7)
summary(cloud9am_fit)

lm_pred <- round(predict(cloud9am_fit, weather_data7))
lm_pred[lm_pred == 9] = 8
plot(x = weather_data7$Sunshine, y = weather_data7$Cloud9am, type='p', xlab = "Sunshine", ylab = "Cloud9am")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = weather_data7$Sunshine, y = lm_pred, col='red')

minTemp_fit <- lm(MinTempTomorrow ~ MinTemp + Humidity3pm, data = weather_data7)
summary(minTemp_fit)

lm_pred <- predict(minTemp_fit, weather_data7)
plot(x = weather_data7$Sunshine, y = weather_data7$MinTemp, type ='p', xlab = "Sunshine", ylab = "MinTemp")
legend("topright", c("actual","fitted"), fill = c("black","red"))
points(x = weather_data7$Sunshine, y = lm_pred, col='red')

maxTemp_fit <- lm(MaxTempTomorrow ~ MaxTemp + Evaporation, data = weather_data7)
summary(maxTemp_fit)

lm_pred <- predict(maxTemp_fit, weather_data7)
plot(x = weather_data7$Sunshine, y = weather_data7$MaxTemp, type = 'p', xlab = "Sunshine", ylab = "MaxTemp")
legend("topright", c("actual", "fitted"), fill = c("black", "red"))
points(x = weather_data7$Sunshine, y = lm_pred, col = 'red')

computeCloudConditions = function(cloud_9am, cloud_3pm) {
  cloud_avg = min(round((cloud_9am + cloud_3pm)/2), 8)
  cc_str = NULL
  if (cloud_avg == 8) {
    cc_str = "Cloudy"
  } else if (cloud_avg >= 6) {
    cc_str = "Mostly Cloudy"
  } else if (cloud_avg >= 3) {
    cc_str = "Partly Cloudy"
  } else if (cloud_avg >= 1) {
    cc_str = "Mostly Sunny"
  } else if (cloud_avg < 1) {
    cc_str = "Sunny"
  }
  cc_str
}

