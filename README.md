# Australia-Weather-Prediction
Predicting weather in Australia using R with a given dataset

---------------
# Prerequisites
- Latest version of R

--------------
# Acknowledgements
https://datascienceplus.com/weather-forecast-with-regression-models-part-1/
https://www.statmethods.net/stats/regression.html

--------------
# Code Explaination

--------------
Starting from the first rows of lines, include the packages that are required for the code

suppressPackageStartupMessages(library(knitr))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(gmodels))
suppressPackageStartupMessages(library(lattice))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(ROCR))
suppressPackageStartupMessages(library(corrplot))

--------------
Then, include the CSV file that contains every information that are used in the code

set.seed(1023)
weather_data <- read.csv("C:\\Users\\ASUS\\Downloads\\Project RO sejauh ini\\weatherAUS.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)
kable(head(weather_data))

--------------
Metrics at specific Date and Location are given together with the RainTomorrow variable indicating if rain occurred on next day.

colnames(weather_data)

--------------
The description of the variables is the following.

Date: The date of observation (a date object).

Location: The common name of the location of the weather station

MinTemp: The minimum temperature in degrees centigrade

MaxTemp: The maximum temperature in degrees centigrade

Rainfall: The amount of rainfall recorded for the day in millimeters.

Evaporation: Class A pan evaporation (in millimeters) during 24 h

Sunshine: The number of hours of bright sunshine in the day

WindGustDir: The direction of the strongest wind gust in the 24 h to midnight

WindGustSpeed: The speed (in kilometers per hour) of the strongest wind gust in the 24 h to midnight

WindDir9am: The direction of the wind gust at 9 a.m.

WindDir3pm: The direction of the wind gust at 3 p.m.

WindSpeed9am: Wind speed (in kilometers per hour) averaged over 10 min before 9 a.m.

WindSpeed3pm: Wind speed (in kilometers per hour) averaged over 10 min before 3 p.m.

Humidity9am: Relative humidity (in percent) at 9 am

Humidity3pm: Relative humidity (in percent) at 3 pm

Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9 a.m.

Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3 p.m.

Cloud9am: Fraction of sky obscured by cloud at 9 a.m. This is measured in ”oktas,” which are a unit of eighths. It records how many eighths of the sky are obscured by cloud. A 0 measure indicates completely clear sky, while an 8 indicates that it is completely overcast

Cloud3pm: Fraction of sky obscured by cloud at 3 p.m; see Cloud9am for a description of the values

Temp9am: Temperature (degrees C) at 9 a.m.

Temp3pm: Temperature (degrees C) at 3 p.m.

RainToday: Integer 1 if precipitation (in millimeters) in the 24 h to 9 a.m. exceeds 1 mm, otherwise 0

RISK_MM: The continuous target variable; the amount of rain recorded during the next day

RainTomorrow: The binary target variable whether it rains or not during the next day


Looking then at the data structure, we discover the dataset includes a mix of numerical and categorical variables.

str(weather_data)

--------------
We have available 142193 records:

(n <- nrow(weather_data))

--------------
which spans the following timeline:

c(as.character(weather_data$Date[1]), as.character(weather_data$Date[n]))

--------------
We further notice that RISK_MM relation with the RainTomorrow variable is the following.

all.equal(weather_data$RISK_MM > 1, weather_data$RainTomorrow == "Yes")

--------------
The Rainfall variable and the RainToday are equivalent according to the following relationship (as anticipated by the Rainfall description).

all.equal(weather_data$Rainfall > 1, weather_data$RainToday == "Yes")

--------------
To make it more challenging, we decide to take RISK_MM, RainFall and RainToday out, and determine a new dataset as herein depicted.

weather_data2 <- subset(weather_data, select = -c(Date, Location, RISK_MM, Rainfall, RainToday))
colnames(weather_data2)

(cols_withNa <- apply(weather_data2, 2, function(x) sum(is.na(x))))

--------------
Look at the NA’s counts associated to WindDir9am. If WindDir9am were a not significative predictor for RainTomorrow, we could take such data column out and increased the complete cases count. When dealing with the categorical variable analysis we determine if that is possible. For now, we consider records without NA’s values.

weather_data3 <- weather_data2[complete.cases(weather_data2),]

--------------
Categorical Variable Analysis

factor_vars <- names(which(sapply(weather_data3, class) == "factor"))
factor_vars <- setdiff(factor_vars, "RainTomorrow")
chisq_test_res <- lapply(factor_vars, function(x) { 
  chisq.test(weather_data3[,x], weather_data3[, "RainTomorrow"], simulate.p.value = TRUE)
})
names(chisq_test_res) <- factor_vars
chisq_test_res

--------------
Above shown Chi-squared p-value results tell us that RainTomorrow values depend upon WindGustDir (we reject the null hypothesis that RainTomorrow does not depend upon the WindGustDir). We do not reject the null-hypothesis for WindDir9am and WindDir3pm as p.value > 0.05, hence RainTomorrow does not depend upon those two predictors. We therefore expect to take advantage of WindGustDir as predictor and not to consider WindDir9am and WindDir3pm for such purpose.

It is also possible to obtain a visual understanding of the significativeness of such categorical variables by taking advantage of barcharts with the cross table row proportions as input.

barchart_res <- lapply(factor_vars, function(x) { 
  title <- colnames(weather_data3[,x, drop=FALSE])
  wgd <- CrossTable(weather_data3[,x], weather_data3$RainTomorrow, prop.chisq=F)
  barchart(wgd$prop.row, stack=F, auto.key=list(rectangles = TRUE, space = "top", title = title))
})
names(barchart_res) <- factor_vars
barchart_res$WindGustDir

barchart_res$WindDir9am

barchart_res$WindDir3pm

--------------
Being WindDir9am not a candidate predictor and having got more than 30 NA’s values, we decide to take it out. As a consequence, we have increased the number of NA-free records from 328 to 352. Same for WindDir3pm.

weather_data4 <- subset(weather_data2, select = -c(WindDir9am, WindDir3pm))
weather_data5 <- weather_data4[complete.cases(weather_data4),]
colnames(weather_data5)

--------------
So, we end up with a dataset made up of 16 potential predictors, one of those is a categorical variable (WindGustDir) and 15 are quantitative.

Quantitative Variable Analysis
In this section, we carry on the exploratory analysis of quantitative variables. We start first by a visualization of the correlation relationship among variables.

factor_vars <- names(which(sapply(weather_data5, class) == "factor"))
numeric_vars <- setdiff(colnames(weather_data5), factor_vars)
numeric_vars <- setdiff(numeric_vars, "RainTomorrow")
numeric_vars
numeric_vars_mat <- as.matrix(weather_data5[, numeric_vars, drop=FALSE])
numeric_vars_cor <- cor(numeric_vars_mat)
corrplot(numeric_vars_cor)

--------------
By taking a look at above shown correlation plot, we can state that:

Temp9am strongly positive correlated with MinTemp

Temp9am strongly positive correlated with MaxTemp

Temp9am strongly positive correlated with Temp3pm

Temp3pm strongly positive correlated with MaxTemp

Pressure9am strongly positive correlated with Pressure3pm

Humidity3pm strongly negative correlated with Sunshine

Cloud9am strongly negative correlated with Sunshine

Cloud3pm strongly negative correlated with Sunshine

The pairs plot put in evidence if any relationship among variables is in place, such as a linear relationship.

pairs(weather_data5[,numeric_vars], col=weather_data5$RainTomorrow)

--------------
Visual analysis, put in evidence a linear relationship among the following variable pairs:

MinTemp and MaxTemp

MinTemp and Evaporation

MaxTemp and Evaporation

Temp9am and MinTemp

Temp3pm and MaxTemp

Pressure9am and Pressure3pm

Humidity9am and Humidity3pm

WindSpeed9am and WindGustSpeed

WindSpeed3pm and WindGustSpeed

Boxplots and density plots give a visual understanding of the significativeness of a predictor by looking how much are overlapping the predictor values set depending on the variable to be predicted (RainTomorrow).

gp <- invisible(lapply(numeric_vars, function(x) { 
  ggplot(data=weather_data5, aes(x= RainTomorrow, y=eval(parse(text=x)), col = RainTomorrow)) + geom_boxplot() + xlab("RainTomorrow") + ylab(x) + ggtitle("") + theme(legend.position="none")}))
grob_plots <- invisible(lapply(chunk(1, length(gp), 4), function(x) {
  marrangeGrob(grobs=lapply(gp[x], ggplotGrob), nrow=2, ncol=2)}))
grob_plots

gp <- invisible(lapply(numeric_vars, function(x) { 
  ggplot(data=weather_data5, aes(x=eval(parse(text=x)), col = RainTomorrow)) + geom_density() + xlab(x) + ggtitle(paste(x, "density", sep= " "))}))
grob_plots <- invisible(lapply(chunk(1, length(gp), 4), function(x) {
  marrangeGrob(grobs=lapply(gp[x], ggplotGrob), nrow=2, ncol=2)}))
grob_plots

--------------
From all those plots, we can state that Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm and Sunshine are predictors to be considered.


We suppose the MinTemp already available at 9am as we expect the overnight temperature resulting with that information. We suppose the MaxTemp already available at 3pm, as determined on central day hours. Further, we suppose Sunshine, Evaporation, WindGustDir and WindGustSpeed final information only by late evening. Other variables are explicitely bound to a specific daytime.

nrow(weather_data5)

sum(weather_data5["RainTomorrow"] == "Yes")

sum(weather_data5["RainTomorrow"] == "No")

--------------
We are going to split the original dataset in a training dataset (70% of original data) and a testing dataset (30% remaining).

train_rec <- createDataPartition(weather_data5$RainTomorrow, p = 0.7, list = FALSE)
training <- weather_data5[train_rec,]
testing <- weather_data5[-train_rec,]

--------------
We check the balance of RainTomorrow Yes/No fractions in the training and testing datasets.

sum(training["RainTomorrow"] == "Yes")/sum(training["RainTomorrow"] == "No")

sum(testing["RainTomorrow"] == "Yes")/sum(testing["RainTomorrow"] == "No")

--------------
The dataset is slightly unbalanced with respect the label to be predicted. We will check later if some remedy is needed or achieved accuracy is satisfactory as well.

9AM Forecast Model
For all models, we are going to take advantage of a train control directive made available by the caret package which prescribes repeated k-fold cross-validation. The k-fold cross validation method involves splitting the training dataset into k-subsets. For each subset, one is held out while the model is trained on all other subsets. This process is completed until accuracy is determined for each instance in the training dataset, and an overall accuracy estimate is provided. The process of splitting the data into k-folds can be repeated a number of times and this is called repeated k-fold cross validation. The final model accuracy is taken as the mean from the number of repeats.

trControl <- trainControl(method = "repeatedcv",  repeats = 5, number = 10, verboseIter = FALSE)

--------------
The trainControl is passed as a parameter to the train() caret function. As a further input to the train() function, the tuneLength parameter indicates the number of different values to try for each algorithm parameter. However in case of the “glm” method, it does not drive any specific behavior and hence it will be omitted.

By taking into account the results from exploratory analysis, we herein compare two models for 9AM prediction. The first one is so determined and evaluated.

predictors_9am_c1 <- c("Cloud9am",  "Humidity9am", "Pressure9am", "Temp9am")
formula_9am_c1 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c1, collapse="+"), sep="~"))
mod9am_c1_fit <- train(formula_9am_c1,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod9am_c1_fit$results$Accuracy

--------------
The resulting accuracy shown above is the one determined by the repeated k-fold cross validation as above explained. The obtained value is not that bad considering the use of just 9AM available data.

(summary_rep <- summary(mod9am_c1_fit$finalModel))

--------------
From above summary, all predictors result as significative for the logistic regression model. We can test the null hypothesis that the logistic model represents an adequate fit for the data by computing the following p-value.

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
We further investigate if any predictor can be dropped by taking advantage of the drop1() function.

drop1(mod9am_c1_fit$finalModel, test="Chisq")

--------------
We can evaluate a second model where the MinTemp is replaced by the Temp9am. We do not keep both as they are correlated (see part #1 exploratory analysis).

predictors_9am_c2 <- c("Cloud9am",  "Humidity9am", "Pressure9am", "MinTemp")
formula_9am_c2 <- as.formula(paste("RainTomorrow", paste(predictors_9am_c2, collapse="+"), sep="~"))
mod9am_c2_fit <- train(formula_9am_c2,  data=training, method="glm", 
                       family="binomial", trControl = trControl, metric = 'Accuracy')
mod9am_c2_fit$results$Accuracy

(summary_rep <- summary(mod9am_c2_fit$finalModel))

--------------
The p-value associated with the null hypothesis that this model is a good fit for the data is:

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
This second model shows similar accuracy values, however MinTemp p-value shows that such predictor is not significative. Further, the explained variance is slightly less than the first model one. As a conclusion, we select the first model for 9AM predictions purpose and we go on by calculating accuracy with the test set.

mod9am_pred <- predict(mod9am_c1_fit, testing)
confusionMatrix(mod9am_pred, testing[,"RainTomorrow"])

--------------
The confusion matrix shows encouraging results, not a bad test accuracy for a 9AM weather forecast. We then go on building the 3PM prediction model.

3PM Forecast Model
We are going to use what we expect to be reliable predictors at 3PM. We have already seen that they are not strongly correlated each other and they are not linearly dependent.

predictors_3pm_c1 <- c("Cloud3pm", "Humidity3pm", "Pressure3pm", "Temp3pm")
formula_3pm_c1 <- as.formula(paste("RainTomorrow", paste(predictors_3pm_c1, collapse="+"), sep="~"))
mod3pm_c1_fit <- train(formula_3pm_c1,  data = training, method = "glm", family = "binomial",
                       trControl = trControl, metric = 'Accuracy')
mod3pm_c1_fit$results$Accuracy

--------------
This model shows an acceptable accuracy as measured on the training set.

(summary_rep <- summary(mod3pm_c1_fit$finalModel))

--------------
All predictors are reported as significative for the model. Let us further verify if any predictor can be dropped.

drop1(mod3pm_c1_fit$finalModel, test="Chisq")

--------------
The p-value associated with the null hypothesis that this model is a good fit for the data is:

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
We go on with the computation of the test set accuracy.

mod3pm_pred <- predict(mod3pm_c1_fit, testing)
confusionMatrix(mod3pm_pred, testing[,"RainTomorrow"])

--------------
The test set prediction accuracy is quite satisfactory.

Evening Forecast Model
As first evening forecast model, we introduce the Sunshine variable and at the same time we take Cloud3pm and Humidity3pm out as strongly correlated to Sunshine.

predictors_evening_c1 <- c("Pressure3pm", "Temp3pm", "Sunshine")
formula_evening_c1 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c1, collapse="+"), sep="~"))
mod_ev_c1_fit <- train(formula_evening_c1,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c1_fit$results$Accuracy

--------------
The training set based accuracy is acceptable.

(summary_rep <- summary(mod_ev_c1_fit$finalModel))

--------------
The Temp3pm predictor is reported as not significative and can be dropped as confirmed below.

drop1(mod_ev_c1_fit$finalModel, test="Chisq")

--------------
The p-value associated with the null hypothesis that this model is a good fit for the data is:

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
As a second tentative model, we take advantage of the 3PM model predictors together with WindGustDir and WindGustSpeed.

predictors_evening_c2 <- c(predictors_3pm_c1, "WindGustDir", "WindGustSpeed")
formula_evening_c2 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c2, collapse="+"), sep="~"))
mod_ev_c2_fit <- train(formula_evening_c2,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c2_fit$results$Accuracy

--------------
It results with an improved training set accuracy.

(summary_rep <- summary(mod_ev_c2_fit$finalModel))

--------------
The WindGustDir and WindGustSpeed predictors are reported as not significative. Also the AIC value is sensibly increased.

drop1(mod_ev_c2_fit$finalModel, test="Chisq")

--------------
WindGustDir has some borderline p-value for some specific directions. WindGustSpeed is not significative and we should drop it from the model. Hence, we redefine such model after having taken WindGustSpeed off while keeping WindGustDir.

predictors_evening_c2 <- c(predictors_3pm_c1, "WindGustDir")
formula_evening_c2 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c2, collapse="+"), sep="~"))
mod_ev_c2_fit <- train(formula_evening_c2,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c2_fit$results$Accuracy

(summary_rep <- summary(mod_ev_c2_fit$finalModel))

drop1(mod_ev_c2_fit$finalModel, test="Chisq")

--------------
WindGustDirESE is reported as significant. Hence to include WindGustDir is right and then we accept that model as a candidate one. The p-value associated with the null hypothesis that this model is a good fit for the data is:

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
To investigate a final third choice, we gather a small set of predictors, Pressure3pm and Sunshine.

predictors_evening_c3 <- c("Pressure3pm", "Sunshine")
formula_evening_c3 <- as.formula(paste("RainTomorrow", paste(predictors_evening_c3, collapse="+"), sep="~"))
mod_ev_c3_fit <- train(formula_evening_c3,  data=training, method="glm", family="binomial", trControl = trControl, metric = 'Accuracy')
mod_ev_c3_fit$results$Accuracy

--------------
The training set based accuracy has an acceptable value.

(summary_rep <- summary(mod_ev_c3_fit$finalModel))

--------------
All predictors are reported as significative.

drop1(mod_ev_c3_fit$finalModel, test="Chisq")

--------------
The p-value associated with the null hypothesis that this model is a good fit for the data is:

1 - pchisq(summary_rep$deviance, summary_rep$df.residual)

--------------
We compare the last two models by running an ANOVA analysis on those to check if the lower residual deviance of the first model is significative or not.

anova(mod_ev_c2_fit$finalModel, mod_ev_c3_fit$finalModel, test="Chisq")

--------------
Based on p-value, there is no significative difference between them. We then choose both models. The first model because it provides with a better accuracy than the second. The second model for its simplicity. Let us evaluate the test accuracy for both of them.

modevening_pred <- predict(mod_ev_c2_fit, testing)
confusionMatrix(modevening_pred, testing[,"RainTomorrow"])

--------------
Good test accuracy with a high sensitivity and positive predicted values. We then test the second evening forecast candidate model.

modevening_pred <- predict(mod_ev_c3_fit, testing)
confusionMatrix(modevening_pred, testing[,"RainTomorrow"])

--------------
Slightly higher accuracy and lower sensitivity than the previous model. Positive predicitive value is improved with respect the same.

Resuming up our final choices:

mod9am_c1_fit: RainTomorrow ~ Cloud9am + Humidity9am + Pressure9am + Temp9am
mod3pm_c1_fit: RainTomorrow ~ Cloud3pm + Humidity3pm + Pressure3pm + Temp3pm
mod_ev_c2_fit: RainTomorrow ~ Cloud3pm + Humidity3pm + Pressure3pm + Temp3pm + WindGustDir
mod_ev_c3_fit: RainTomorrow ~ Pressure3pm + Sunshine

--------------
We then start the moderate rainfall scenario analysis. At the same time, we prepare the dataset for the rest of the analysis.

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

--------------
It is important to understand what is the segmentation of moderate rainfall records in terms of training and testing set, as herein shown.

(train_hr <- hr_idx[hr_idx %in% train_rec])
(test_hr <- hr_idx[!(hr_idx %in% train_rec)])

--------------
We see that some of the “at least moderate” Rainfall records belong to the testing dataset, hence we can use it in order to have a measure based on unseen data by our model. We test the evening models with a test-set comprising such moderate rainfall records.

rain_test <- weather_data7[test_hr,]
rain_test

--------------
Let us see how the first weather forecast evening model performs.

opt_cutoff <- 0.42
pred_test <- predict(mod_ev_c2_fit, rain_test, type="prob")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
data.frame(prediction = prediction, RainfallTomorrow = rain_test$RainfallTomorrow)

--------------
Then, the second evening weather forecast model.

opt_cutoff <- 0.56
pred_test <- predict(mod_ev_c3_fit, rain_test, type="prob")
prediction <- ifelse(pred_test$Yes >= opt_cutoff, "Yes", "No")
data.frame(prediction = prediction, RainfallTomorrow = rain_test$RainfallTomorrow)

--------------
For both evening forecast models, one of the testing set predictions is wrong. If we like to improve it, we have to step back to the tuning procedure and determine a decision threshold value more suitable to capture such scenarios. From the tables that we show in the previous part, we can try to lower the cutoff value to increase specificity, however likely implying to reduce accuracy.

Chance of Rain
In the previous part of this series, when discussing with the tuning of the decision threshold, we dealt with probabilities associated to the predicted RainTomorrow variable. The chances of having RainTomorrow == “Yes” are equivalent to the chance of rain. Hence the following utility function can be depicted at the purpose

chance_of_rain <- function(model, data_record){
  chance_frac <- predict(mod_ev_c3_fit, data_record, type="prob")[,"Yes"]
  paste(round(chance_frac*100),"%",sep="")
}

--------------
We try it out passing ten records of the testing dataset.

chance_of_rain(mod_ev_c3_fit, testing[1:10,])

--------------
To build all the following models, we are going to use all the data available in order to capture the variability of an entire year. For brevity, we do not make comparison among models for same predicted variable and we do not show regression models diagnostic plots.

Tomorrow’s Rainfall Prediction
If the logistic regression model predicts RainTomorrow = “Yes”, we would like to take advantage of a linear regression model capable to predict the Rainfall value for tomorrow. In other words, we are just interested in records whose Rainfall outcome is greater than 1 mm.

weather_data8 = weather_data7[weather_data7$RainfallTomorrow > 1,]
rf_fit <- lm(RainfallTomorrow ~ MaxTemp + Sunshine + WindGustSpeed -1, data = weather_data8)
summary(rf_fit)

--------------
All predictors are reported as significant.

lm_pred <- predict(rf_fit, weather_data8)
plot(x = seq_along(weather_data8$RainfallTomorrow), y = weather_data8$RainfallTomorrow, type='p', xlab = "observations", ylab = "RainfallTomorrow")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = seq_along(weather_data8$RainfallTomorrow), y = lm_pred, col='red')

--------------
Tomorrow’s Humidity 3pm Prediction

sun_fit <- lm(SunshineTomorrow ~ Sunshine*Humidity3pm + Cloud3pm + Evaporation + I(Evaporation^2) + WindGustSpeed - 1, data = weather_data7)
summary(sun_fit)

--------------
All predictors are reported as significant.

lm_pred <- predict(sun_fit, weather_data7)
plot(x = seq_along(weather_data7$SunshineTomorrow), y = weather_data7$SunshineTomorrow, type='p', xlab = "observations", ylab = "SunshineTomorrow")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = seq_along(weather_data7$SunshineTomorrow), y = lm_pred, col='red')

--------------
We have furthermore to take into account tomorrow’s Cloud9am and Cloud3pm. For those quantitative variables, corresponding predictions are needed, and, at the purpose, the following linear regression models based on the Sunshine predictor can be depicted.

cloud9am_fit <- lm(Cloud9am ~ Sunshine, data = weather_data7)
summary(cloud9am_fit)

--------------
All predictors are reported as significant.

lm_pred <- round(predict(cloud9am_fit, weather_data7))
lm_pred[lm_pred == 9] = 8
plot(x = weather_data7$Sunshine, y = weather_data7$Cloud9am, type='p', xlab = "Sunshine", ylab = "Cloud9am")
legend("topright", c("actual", "predicted"), fill = c("black", "red"))
points(x = weather_data7$Sunshine, y = lm_pred, col='red')

--------------
Tomorrow’s MinTemp Prediction

minTemp_fit <- lm(MinTempTomorrow ~ MinTemp + Humidity3pm, data = weather_data7)
summary(minTemp_fit)

--------------
All predictors are reported as significant.

lm_pred <- predict(minTemp_fit, weather_data7)
plot(x = weather_data7$Sunshine, y = weather_data7$MinTemp, type ='p', xlab = "Sunshine", ylab = "MinTemp")
legend("topright", c("actual","fitted"), fill = c("black","red"))
points(x = weather_data7$Sunshine, y = lm_pred, col='red')

--------------
Tomorrow’s MaxTemp Prediction

maxTemp_fit <- lm(MaxTempTomorrow ~ MaxTemp + Evaporation, data = weather_data7)
summary(maxTemp_fit)

--------------
All predictors are reported as significant.

lm_pred <- predict(maxTemp_fit, weather_data7)
plot(x = weather_data7$Sunshine, y = weather_data7$MaxTemp, type = 'p', xlab = "Sunshine", ylab = "MaxTemp")
legend("topright", c("actual", "fitted"), fill = c("black", "red"))
points(x = weather_data7$Sunshine, y = lm_pred, col = 'red')

--------------
CloudConditions computation
Based on second reference given at the end, we have the following mapping between a descriptive cloud conditions string and the cloud coverage in ”oktas,” which are a unit of eights.
We can figure out the following utility function to help.

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

--------------
Weather Forecast Report
Starting from a basic example of weather dataset, we were able to build several regression models. The first one, based on logistic regression, is capable of predict the RainTomorrow factor variable. The linear regression models are to predict the Rainfall, Humidity3pm, WindGustSpeed, MinTemp, MaxTemp, CloudConditions weather metrics.

Chance of rain is computed only if RainTomorrow prediction is “Yes”. The Humidity3pm prediction is taken as humidity prediction for the whole day, in general.

weather_report <- function(today_record, rain_tomorrow_model, cutoff) {
  rainTomorrow_prob <- predict(rain_tomorrow_model, today_record, type="prob")
  rainTomorrow_pred = ifelse(rainTomorrow_prob$Yes >= cutoff, "Yes", "No")
  
  rainfall_pred <- NA
  chance_of_rain <- NA
  if (rainTomorrow_pred == "Yes") {
    rainfall_pred <- round(predict(rf_fit, today_record), 1)
    chance_of_rain <- round(rainTomorrow_prob$Yes*100)
  }
  
  wgs_pred <- round(predict(wgs_fit, today_record), 1)
  
  h3pm_pred <- round(predict(h3pm_fit, today_record), 1)
  
  sun_pred <- predict(sun_fit, today_record)
  
  cloud9am_pred <- min(round(predict(cloud9am_fit, data.frame(Sunshine=sun_pred))), 8)
  cloud3pm_pred <- min(round(predict(cloud3pm_fit, data.frame(Sunshine=sun_pred))), 8)
  CloudConditions_pred <- computeCloudConditions(cloud9am_pred, cloud3pm_pred)
  
  minTemp_pred <- round(predict(minTemp_fit, today_record), 1)
  
  maxTemp_pred <- round(predict(maxTemp_fit, today_record), 1)
  
  if (is.na(rainfall_pred)) {
    rainfall_pred_str <- "< 1 mm"
  } else {
    rainfall_pred_str <- paste(rainfall_pred, "mm", sep = " ")
  }
  
  if (is.na(chance_of_rain)) {
    chance_of_rain_str <- ""
  } else {
    chance_of_rain_str <- paste(chance_of_rain, "%", sep="")
  }
  
  wgs_pred_str <- paste(wgs_pred, "Km/h", sep= " ")
  h3pm_pred_str <- paste(h3pm_pred, "%", sep = "")
  minTemp_pred_str <- paste(minTemp_pred, "°C", sep= "")
  maxTemp_pred_str <- paste(maxTemp_pred, "°C", sep= "")
  
  report <- data.frame(Rainfall = rainfall_pred_str,
                       ChanceOfRain = chance_of_rain_str,
                       WindGustSpeed = wgs_pred_str, 
                       Humidity = h3pm_pred_str,
                       CloudConditions = CloudConditions_pred,
                       MinTemp = minTemp_pred_str,
                       MaxTemp = maxTemp_pred_str)
 
}
report

--------------
Sure there are confidence and prediction intervals associated to our predictions. However, since we intend to report our forecasts to Canberra’s citizens, our message should be put in simple words to reach everybody and not just statisticians.

Finally we can try our tomorrow weather forecast report out.

(tomorrow_report <- weather_report(weather_data[32,], mod_ev_c3_fit, 0.56))

(tomorrow_report <- weather_report(weather_data[10,], mod_ev_c3_fit, 0.56))

--------------
Please note that some records of the original weather dataset may show NA’s values and our regression models did not cover a predictor can be as such. To definitely evaluate the accuracy of our weather forecast report, we would need to check its unseen data predictions with the occurred tomorrow’s weather. Further, to improve the adjusted R-squared metric of our linear regression models is a potential area of investigation.
