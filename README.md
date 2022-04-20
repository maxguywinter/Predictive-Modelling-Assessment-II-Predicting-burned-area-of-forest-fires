# Predictive-Modelling-Assessment-II-Predicting-burned-area-of-forest-fires
Constructing logistic regression model to predict the probability of a forest fire bigger than 1 ha. (Grade 87%).  

# Introduction 

This report aims to build a model to predict the probability of a forest fire bigger than 1 ha, in the northeast region of Portugal, by using meteorological and other data from the dataset Forest Fires (Morais, 2007). 

Section 1.0 of this report illustrates the statistical and graphical summaries of the Forest Fires dataset. Moreover, section1.0 cleans and transforms the data suitable for the binary logistic regression model. Section 2.0 constructs an initial model and then utilises Akaike information criterion (AIC), Bayesian information criterion (BIC), Brier score, pseudo-R squared, Wald test and the likelihood ratio test to create the best prediction model. Section 3.0 validates the best fitting model by analysing the model’s statistical significance, residuals, overdispersion and outliers. Ultimately, section 4.0 uses the best fitting model for predicting the probability of a forest fire bigger than 1 ha and subsequently validates the model prediction using test and train sets, confusion matrix, histogram, classifications prediction accuracy and the Receiver operating characteristic curve (ROC).

# Section 1.0 Data Summary 

Section 1.1 Numerical Summary and Data Transforming 

Figure 1.0 Data Summary 

<img width="517" alt="image" src="https://user-images.githubusercontent.com/97530878/153616545-94db7d75-8305-4ff2-aac8-e7e3f371fac2.png">

The fire forest dataset (fire_data) include 517 observations and 13 variables. Figure 1 illustrates the summary function that summarises each of the 13 variables on the forest fire dataset. The explanatory variables are X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain. In comparison, the response variable is area that shows the burned area of the forest (in ha). However, the response variable area is not binary. Therefore, the area was transformed into area_dummy to become binary (0 = under 1 ha and 1= over 1 ha) to be used for logistic regression. 

Moreover, X and Y variables were removed as they are factors and are not relevant to the model as they do not impact the burned area. Lastly, the variables month and day were transformed to factors as they were both categorical data. Consequently, a train (77.37%) and test (22.63%) set were constructed, removing the irrelevant variables X, Y and area. The train set is used to develop the predictive model and the test set for predictions and testing the model.
 
Section 1.2 Graphical Summary 

Figure 2.0 Missing Plot

<img width="444" alt="image" src="https://user-images.githubusercontent.com/97530878/153616642-d00aa493-2c97-4c10-ade7-321a7d52a9ec.png">

Figure 2 highlights if there is any missing data in the dataset. According to Peng (2013), missing data can decrease statistical power, increase standard errors, and weaken generalisability. Therefore, a missing plot was utilised to check for missing data. The x-axis shows the variables, and the y-axis shows the observations. If there were missing data, the block would be white. However, all the blocks are red, highlighting no missing data to be cleaned.

Section 1.3 Correlation Analysis 

Figure 4.0 Correlation Matrix 

<img width="396" alt="image" src="https://user-images.githubusercontent.com/97530878/153616707-f8bca304-232a-45b8-9b2b-5643e7958704.png">


Figure 5.0 Scatterplot Matrix 

<img width="497" alt="image" src="https://user-images.githubusercontent.com/97530878/153616740-7d689c8e-5d4f-4d49-82ad-e13425176b03.png">

Figures 4 and 5 above check for collinearity and multicollinearity of the explanatory variables. Both figures display that the variables are not highly correlated or have a strong linear relationship with one another. None of the absolute values of the correlation is greater than 0.9. Therefore, multicollinearity is not an issue in this model. 

# Section 2.0 Model Selection 

Section 2.1 Initial Model Construction 

Figure 6.0 Model 1 Summary 

<img width="210" alt="image" src="https://user-images.githubusercontent.com/97530878/153616787-b4d684f3-fce7-4538-a346-f59145c514c7.png">

Figure 6.1 Model 1 ANOVA

<img width="229" alt="image" src="https://user-images.githubusercontent.com/97530878/153616822-cf36d7fc-953f-4dde-b582-c2b0cd5ea035.png">

Model 1 was constructed with all explanatory variables from the train sample. The Wald test Pr(> | Z | ) from figure 6.0 highlights that only the wind variable had some evidence of contribution to forest fires larger than 1ha. Therefore, all other variables appear to be insignificant covariates in model 1. Additionally, figure 6.1 exhibits that the likelihood ratio test Pr(>Chi) that only month and wind have some evidence of an effect on the response variable. Consequently, having unnecessary variables in the model will increase the variability of the predictions. Therefore, the factors month and day were removed for model 2 as the majority had the highest p-values and were insignificant.  

Figure 6.2 Model 1 Half-Normal Plot 

<img width="369" alt="image" src="https://user-images.githubusercontent.com/97530878/153616852-8c82f38b-895d-42cc-afbc-d3ea79f018ec.png">

Figure 6.3 Model 1 Overdispersion and Outlier Test

<img width="208" alt="image" src="https://user-images.githubusercontent.com/97530878/153616896-552b0708-59a4-4ba2-bcaa-c368a06a7232.png">

Furthermore, figure 6.3 displays that model 1’s ratio of the residual deviance to the residual degrees of freedom was 1.4 and not exceeding 1.5. Therefore model 1 is not overdispersed. Nevertheless, the ratio is very high, showing that model 1 does not fit that data well. Hence, Mangiafico (2015) proclaims that the ‘explanatory variables may not well describe the dependent variable, or the model may not be specified correctly for these data’. 

Moreover, the outlier test tests whether the largest absolute residual is larger than expected. Therefore, the identified point might be an outlier if the Bonferroni p-value is significant (p<0.05). Model 1’s Bonferroni p-value highlights that observation 510 is the largest residual but has a p-value of NA. Thus, the outlier test is inconclusive to determine any outliers. However, the half-normal plot (figure 6.2) above illustrates many points outside the simulation envelope, suggesting a systematic problem with the initial model.  

Figure 6.4 Model 1 DHARMa Residual Plots 

<img width="385" alt="image" src="https://user-images.githubusercontent.com/97530878/153616949-cb435365-5a3e-4b0a-a762-ec85937d0ac7.png">

However, figure 6.4 illustrates that model 1 has no outliers or suffers from under or overdispersion. According to Hartig (2016), all cumulative distribution values will appear with equal probability for a correctly specified model. Therefore, the distribution of the residuals should be flat, regardless of whether the model is Poisson, binomial or another.  

As stated above, the variables month and day were the first variables removed in section 2.2 variable selection. Variable selection is crucial for developing a predictive model, as failing to include essential variables could lead to biased predictions. Additionally, including irrelevant variables will increase the variability of the predictions.  

Section 2.2 Variables Selection 

Figure 7.0 Model 2 Summary 

<img width="174" alt="image" src="https://user-images.githubusercontent.com/97530878/153617007-6aaae789-3dfb-4023-a25e-735a72de828a.png">

Figure 7.0 shows that removing the variables month and day improved the model as the AIC score decreased from 576.22 to 559.1, showing that model 2 has smaller deviance and thus a better fit than model 1. However, the Wald test p-values still show that majority of the variables are insignificant. The variable temp had the largest p-value and was removed from model 3.

Figure 7.1 Model 3 Summary 

<img width="172" alt="image" src="https://user-images.githubusercontent.com/97530878/153617043-5fe5a713-c43e-4a6c-9f4e-67b8b7932e84.png">

Figure 7.1 model 3 was a better fitting model than the previous model as the AIC score decreased from 559.1 to 557.1. Nevertheless, the Wald test p-values still show that majority of the variables are insignificant. Therefore, illustrated below by figure 7.2, the variable with the largest p-value will be removed from each new model until the model only contains significant variables.

Figure 7.2 Model 4,6 and 7 Summaries 

<img width="218" alt="image" src="https://user-images.githubusercontent.com/97530878/153617090-12fdad4e-490d-4445-b7c9-92095c3ad394.png">

Figure 7.2 shows that each new model AIC score improved from the previous model, thus showing a better fit. Model 7 had two out of its three variables having fairly strong evidence of an effect on the response variable. Hence, the only non-significant variable rain was removed, and the model is re-fitted for model 8 below.  

Figure 7.3 Model 8 Summary 

<img width="235" alt="image" src="https://user-images.githubusercontent.com/97530878/153617151-a600489d-68d5-46e2-8d2a-08999f9e3bd9.png">

Figure 7.4 Model 8 ANOVA 

<img width="241" alt="image" src="https://user-images.githubusercontent.com/97530878/153617189-152be395-39e8-4b68-82d0-4b3dfb85c8e2.png">

According to Figure 7.3, the two variables (DC and wind) are moderately significant. Figure 7.4 exhibits that its likelihood ratio test p-value is significant for both variables though less significant for DC. Moreover, model 8 has the lowest AIC score of 549.18 out of all the previous models above. However, a stepwise model selection using AIC and BIC are utilised to create an AIC and BIC model in the following section. These two new models are then compared to model 8 to decide the best prediction model.  

Section 2.3 Best Fitting Model 

Figure 8.0 AIC Model Summary 

<img width="252" alt="image" src="https://user-images.githubusercontent.com/97530878/153617242-f216a867-4eb8-48cf-9b6c-73caac2be440.png">

Figure 8.1 AIC Model ANOVA 

<img width="254" alt="image" src="https://user-images.githubusercontent.com/97530878/153617271-b10f40bc-db44-4b8f-85ff-281db42db36a.png">

The stepwise model selection using AIC from the full model (model 1) produced the model aicmod. As expected, figures 8.0 and 8.1 show that the aicmod was identical to model 8, with the exact p-values and AIC score. Therefore, model 8 for the following will be referred to as the AIC model aicmod.

Figure 8.2 BIC Model Summary 

<img width="229" alt="image" src="https://user-images.githubusercontent.com/97530878/153617306-da1f5e0a-4325-4390-97d9-25e6c07175c5.png">

The BIC model above (figure 8.2) differs from the AIC model by taking twice the number of parameters in the model. The BIC model takes log(n) times the number of parameters, where n is the number of data points in the train set. 

The BIC model (bicmod) was the same as the null model as the model did not keep any variables. Moreover, the BIC model had a larger and worse AIC score of 553.62 compared to the AIC model. However, the AIC will always produce a larger model than the BIC model and is more helpful in developing prediction models. In contrast, the BIC is better for models for understanding. Consequently, AIC, ANOVA, pseudo-R squared, and the Brier score is utilised below to compare the AIC and BIC models to select the best fitting model for prediction.

Figure 8.3 AIC Test

<img width="172" alt="image" src="https://user-images.githubusercontent.com/97530878/153617359-9cd028a4-13a7-4de8-a99e-1ebaac2c37d5.png">

The AIC test above (figure 8.3) shows that the AIC model (aicmod) was the best fitting model and significantly improved from the initial model (model 1). Nevertheless, the BIC model’s (bicmod) AIC was slightly higher than the AIC model.

Figure 8.4 ANOVA Test 

<img width="299" alt="image" src="https://user-images.githubusercontent.com/97530878/153617398-d474dc82-06b6-4e13-9902-020fe29b2d80.png">

The ANOVA test (figure 8.4) between the AIC and BIC models indicates that the AIC model is the better fitting model. The results show that the more complex model (AIC model) produces more accurate fitted values than the simplest model (BIC model) as the p-value was 0.0147. The AIC model is favoured as the p-value was sufficiently low as it was not greater than 0.05.

Figure 8.5 Pseudo-R Squared  

<img width="314" alt="image" src="https://user-images.githubusercontent.com/97530878/153617426-9d35da2e-1efe-44c4-8d09-6372a96ec3c9.png">

The pseudo-R squared is used to evaluate a model’s accuracy and predictive power. A pseudo-R squared score range from 0 to just under 1, with scores closer to 0 indicating that the model has no predictive power. The pseudo-R squared results exhibit that model 1 is the most accurate model and has the most predictive power. Figure 8.5 indicates that the BIC model pseudo-R squared was 0, thus highlighting that the model does not explain the variability in the outcome and thus has no predictive power. Therefore, the AIC model is better for prediction than the BIC model.

Figure 8.6 Brier Score  

<img width="287" alt="image" src="https://user-images.githubusercontent.com/97530878/153617470-03353e78-c58e-40dd-a61c-1c3343e80fab.png">

Figure 8.7 Model Comparison   

<img width="318" alt="image" src="https://user-images.githubusercontent.com/97530878/153617501-046b4043-bd78-4613-9d51-f03d68ef092c.png">

However, the Brier score (figure 8.6) and the function compareGLM (figure 8.7) to assess the fit statistics above suggest that the BIC model predicts better than the AIC. The BIC had the lowest Brier score and lowest p-value. Nevertheless, the Brier score for BIC is marginally different from the AIC Brier score.

Overall, the AIC model (aicmod) had the best AIC, pseudo-R squared, likelihood ratio and Wald test, indicating that the AIC is the best prediction model. The BIC model (bicmod) only had a better Brier score, but having a pseudo-R squared of 0 exposing the BIC model does not have any predictive power. Consequently, the AIC model (aicmod) will be used for predictions and validated in the following section.

# Section 3.0 Model Validation 

Section 3.1 Analysis of AIC Model

As concluded in the previous section, the AIC model is the best fitting model. The AIC model (aicmod) had the best AIC, pseudo-R squared, likelihood ratio and Wald test, indicating that the AIC is the best prediction model. However, the AIC model’s pseudo-R squared was only 0.015 indicating the model has very little predictive power. Moreover, model 1 had the best pseudo-R squared of 0.05. Nevertheless, R-Squared will always increase when there are more variables, even if those variables are only weakly associated with the outcome.

Moreover, the Wald and the likelihood ratio tests show that the AIC model variables have low p-values being statically significant. Not including significant variables could lead to biased predictions. Whereas including irrelevant variables will increase the variability of the predictions. Consequently, having a small pseudo-R squared and p-value means that the AIC model does not explain much variation of the data, but it is significant.

Section 3.2 Analysis of Residuals

Figure 9.0 Odds Ratio 

<img width="209" alt="image" src="https://user-images.githubusercontent.com/97530878/153617553-ee75ef42-a489-45a9-8480-b0bc7a3e21b5.png">

Figure 9.1 AIC Model Plot 

<img width="459" alt="image" src="https://user-images.githubusercontent.com/97530878/153617571-8cd68958-9095-4962-aea6-7c0e0e839462.png">

Figure 9.0 shows that for each 1 unit increase in wind score, the odds of the forest fire greater than 1ha increases by 1.1 times after adjusting for DC. Likewise, with each one-unit increase in DC, the odds of the forest fire greater than 1ha increases by one time. Consequently, figure 9.1 displays a strong positive relationship between forest fire being over 1ha and DC/wind.

Figure 9.2 Model 1 Half-Normal Plot

<img width="545" alt="image" src="https://user-images.githubusercontent.com/97530878/153617631-c1434440-2e8f-4e77-9c20-0ee49e224d6a.png">

Figure 9.3 AIC Model Overdispersion and Outlier Test

<img width="277" alt="image" src="https://user-images.githubusercontent.com/97530878/153617709-4efa73b0-484a-4dd5-bb23-6fa2e93640e7.png">

Figure 9.3 displays that the AIC model’s ratio of the residual deviance to the residual degrees of freedom was 1.36 and not exceeding 1.5. Therefore, the AIC model does not suffer from overdispersion. Nevertheless, the ratio is very high, showing that the AIC model does not fit that data well. 

Additionally, the outlier test tests highlight that observation 466 is the largest residual but has a p-value of NA. Therefore, again the outlier test is inconclusive to determine any outliers. Nevertheless, the half-normal plot (figure 9.2) above illustrates that no points are outside the simulation envelope, suggesting there are no outliers and that the model is a good fit. Moreover, figure 9.4 illustrates that the AIC model has no outliers and does not suffer from heteroscedasticity or under/overdispersion.  

Figure 9.4 AIC Model DHARMA Residual Plots 

<img width="481" alt="image" src="https://user-images.githubusercontent.com/97530878/153617792-0297083d-b236-4591-a303-3363826bdf5a.png">

Figure 9.5 AIC Model Binned Residual Plot 

<img width="440" alt="image" src="https://user-images.githubusercontent.com/97530878/153617806-f8e2214f-b8ad-4b28-b2b2-b1e5ff657e22.png">

Binned residual plots are employed to assess the overall fit of regression models for binary outcomes. Kasza (2015) states that a model is correct if about 95% of the points are expected to lie within the confidence limits. Therefore, akin to a residual-versus-fitted plot used for linear regression, departures from random scatter indicate that the fitted model does not accurately describe the data. Figure 9.5 shows that no points lie outside the confidence limits and are randomly scattered, suggesting no systematic patterns in the plots. Therefore, the AIC model accurately describes the data and has a good overall fit.

Figure 9.6 AIC Model Collinearity  

<img width="369" alt="image" src="https://user-images.githubusercontent.com/97530878/153617839-3f850926-86c6-46a0-bee9-2c81b1131f50.png">

The VIF values for DC and wind was 1.04. A VIF value larger than 5 suggests a problem of multicollinearity/collinearity. Conversely, a VIF close to 1 indicates no multicollinearity/collinearity. Figure 9.6 above and the VIF results show that the AIC model has no collinear variables. 

# Section 4.0 Model Prediction 

Section 4.1 Predications 

Figure 10.0 AIC Model probabilities 

<img width="269" alt="image" src="https://user-images.githubusercontent.com/97530878/153617884-b418b5b1-8a68-48e2-9e8a-a91125d4a068.png">

Figure 10.1 shows the AIC model predictions with “yes” (over 0.5) predicting a forest fire area larger than 1 ha and “no” (under 0.5) predicting a forest fire area under 1 ha. 

Figure 10.1 AIC Model train set predictions 

<img width="270" alt="image" src="https://user-images.githubusercontent.com/97530878/153617951-4e8ea108-0fa4-4abc-a619-f8cf3d9023fb.png">

The confusion matrix illustrated in figure 10.1 shows the predictions of the AIC model using the train set. The AIC model correctly predicted 168/217 (77.4%) forest fire areas under 1 ha and correctly predicted 61/183 (33.3%) forest fire areas over 1 ha. Therefore, figure 10.1 confusion matrix indicates that the AIC model suffers from type 1 error by predicting more false positives than true positives. However, the AIC does not experience many type 2 errors by predicting more true negatives than false negatives.

Figure 10.2 AIC Model test set predictions 

<img width="320" alt="image" src="https://user-images.githubusercontent.com/97530878/153617988-adf22e44-5a13-4eeb-b23d-5fd246914106.png">

Whereas figure 10.2 shows the predictions of the AIC model using the test set. The AIC model correctly predicted 36/57 (63%) forest fire areas under 1 ha and correctly predicted 17/60 (28.3%) forest fire areas over 1 ha. Consequently, figure 10.2 confusion matrix indicates that the AIC model again suffers from type 1 error. Moreover, the AIC experiences more type 2 errors than the test set. 

Therefore, the predictions of the train and test sets are not similar, indicating that the AIC model has poor prediction accuracy. Additionally, the predictions shown in figure 10.2 are all-around 0.5, meaning the AIC model is not predicting well. The AIC model is doubting its predictions and cannot classify observations into classes (yes or no), as shown in figure 10.4.

Section 4.2 Prediction Validation

Figure 10.3 AIC Model Prediction Histogram 

<img width="445" alt="image" src="https://user-images.githubusercontent.com/97530878/153618039-7f8755d2-e181-4d54-8c82-b4eefd389c51.png">

The AIC model predictions from the confusion matrix had a low accuracy rate of just 45.3% and a high error rate of 54.7%. Therefore, further highlighting the low predictive power of the AIC model. 

Figure 10.3 exhibits the disruption of the predictions made by the AIC model. For a strong predictive model, the forest fire area over 1ha (1/yes) in red should be at the pr 1, and the forest fire area under 1ha (0/no) in grey should be at 0 pr. However, the AIC model shows a normal disruption and thus the overlapping predictions. Hence, there is a problem with the AIC model as the model has low predictive power. Nevertheless, this was expected as the pseudo-R squared score was 0.015 indicating the model does not explain much of the variation of the data and thus has very little predictive power. Moreover, the confusion matrix also highlights the poor predictive power, low accuracy and high error rates of the AIC model.

Lastly, figure 10.4 illustrates a ROC curve that evaluates the prediction accuracy of the AIC model. The ROC curve depicts the trade-off between the sensitivity (true positive rate) and specificity (false positive rate). The AUC evaluates the model performance and represents the area under the ROC curve. The AUC value ranges between 0 and 1, with the higher AUC showing a better performance of a model differentiating between the positive and negative classes.

Figure 10.4 AIC Model ROC Curve  

<img width="452" alt="image" src="https://user-images.githubusercontent.com/97530878/153618082-38bfc220-4ad8-4610-a98b-53c3c93c0475.png">

The AUC value for the AIC model is 0.53. However, an AUC value of 0.5 shows that a model cannot differentiate between the positive and negative classes. Thus, suggesting the model is no better than performing random guessing. Consequently, the AIC model (AUC = 0.53) has deplorable discrimination and the ability to classify observations into classes. Therefore, the AIC model does not fit the dataset well and thus has abysmal prediction accuracy.

# Section 5.0 Script 

library(corrplot) # for correlation matrix graph visualization.
library(ggplot2) # for graph visualizations. 
library(ggthemr)
library(ggthemes)
library(performance)
library(see)
library(car) # for VIF test.
library(caTools)
library(dplyr)
library(MASS) # for AIC test.
library(tidyverse)
library(Amelia)
library(mlbench)
library(flexmix)
library(caret)
library(hnp) # for half normal plot. 
library(rcompanion) # for model comparison. 
library(effects)
library(DHARMa) # for DHARMa plots. 
library(pROC) # for ROC and AUC curves. 

fire_data <- read.csv("forestfires.csv", header = TRUE) # imports the data set. The header=True command tells RStudio to use the first row of the data file as the names of each variable/column. 
attach(fire_data) # attaches the data to your environment so that you can directly refer to the variable by name.
names(fire_data) # shows the name of variables in the data set.
head(fire_data)

summary(fire_data) # provides more information from fitting the model such as Residuals.  
missmap(fire_data, col=c("blue", "red"), legend = FALSE) # Checks for missing data. 

datacorrelation <- subset(fire_data [,5:13]) # subset data to remove x,y, month and day as they are factors. 
nums <- unlist(lapply(datacorrelation, is.numeric)) # used to create correlation plot from numeric values in fire data.
firecorr <- cor(datacorrelation[, nums])
corrplot(firecorr, type = "upper", method = "number")# correlation matrix graph visualization. 
pairs(datacorrelation, col=ifelse(fire_data$area_dummy==0, "black", "red")) # graphical matrix visualization. 

summary(area)
str(fire_data) # shows the observations and variables of the data.

fire_data$area_dummy = ifelse(fire_data$area > 1,1,0 ) # Transforms the response variable area to binary with fire are over 1ha = 1 and under 1ha = 0. 
summary(fire_data)
names(fire_data)
table(fire_data$area_dummy)

fire_data$month <- factor(fire_data$month)
fire_data$day <- factor(fire_data$day) # factor returns the original object of a class with the requested column specified as a factor rather than numeric.

fire_data2 <- fire_data %>% dplyr::select(month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area_dummy) # Removes X, Y and area variables as they are factors and not relevant to the model. 

set.seed(42) # Makes simulations random numbers the same to ensure all results, figures  are reproducible.
x <- fire_data2[sample(1:nrow(fire_data2)),]
train <- x[1:400,]
test <- x[401:517,] # Sets up train and test populations for model validation and predication tests. 

model_1 <- glm(area_dummy ~ ., family = binomial, data = train) # Initial Model Construction (all variables and summary)
summary(model_1) # used for Wald test Pr(> | Z | ) to check significance of variables.

anova(model_1, test = "Chisq") # used for likelihood ratio test Pr(>Chi) to check significance of variables. 
hnp(model_1) # plots a half-normal plot for model 1.
par(mfrow=c(2,2)) # partitions your graphical display if you want to produce multiple graphs on a single plot.
summary(model_1)$deviance/summary(model_1)$df.residual # displays that model 1’s ratio of the residual deviance to the residual degrees of freedom.
outlierTest(model_1) # checks for outliers in model 1 using Bonferroni p-value.
simulateResiduals(model_1, plot = TRUE) # plots DHRAMa plots for model 1.
binned_residuals(model_1) # plots binned residual graph for model 1. 

model_2 <- glm(area_dummy ~ FFMC + DMC + DC + ISI + temp + RH + wind + rain, family = binomial, data = train) # removed month and day.
summary(model_2)

model_3 <- glm(area_dummy ~ FFMC + DMC + DC + ISI + RH + wind + rain, family = binomial, data = train) # removed temp.
summary(model_3)

model_4 <- glm(area_dummy ~ DMC + DC + ISI + RH + wind + rain, family = binomial, data = train) # removed FFMC.
summary(model_4)

model_5 <- glm(area_dummy ~ DC + ISI + RH + wind + rain, family = binomial, data = train) # removed DMC.
summary(model_5)

model_6 <- glm(area_dummy ~ DC + ISI + wind + rain, family = binomial, data = train) # removed RH.
summary(model_6)

model_7 <- glm(area_dummy ~ DC + wind + rain, family = binomial, data = train) # removed ISI.
summary(model_7)

model_8 <- glm(area_dummy ~ DC + wind, family = binomial, data = train) # removed rain.
summary(model_8)
anova(model_8, test = "Chisq")

aicmod = stepAIC(model_1, direction = "both", trace = 0) # produces bet AIC value model (same as model 8).
summary(aicmod)
anova(aicmod, test = "Chisq")
AIC(aicmod) # produces the AIC value of the model.

bicmod = stepAIC(model_1, direction = "both", trace = 0, k=log(nrow(train))) # produces bet BIC value model.
summary(bicmod)
AIC(bicmod)

nullmod = glm(area_dummy ~ 1, family = binomial, data = train) # for comparing models (same as BIC model).  
summary(nullmod)
AIC(nullmod)

AIC(model_1, aicmod, bicmod, nullmod) # AIC model had the best AIC value.  
anova(aicmod,bicmod, test = "Chisq") # p-value less than 0.05 thus aicmod is best model as most complex. 
 
d2 <- function(model, digits = 4) {
  round(1 - (model$deviance / model$null.deviance), digits = digits)
} # creates function to calculate r-squared. 

d2(aicmod) ; d2(bicmod) ; d2(nullmod) ; d2(model_1) ; d2(model_7) # acidmod is the best model as had highest r-squared value. 

logit <- function(p) log(p / (1-p)) # Calculate predictions and confidence limits on logit.
inv.logit <- function(z) 1 / (1 + exp(-z)) # scale, then convert back to probability scale.

predict.interval <- function(model, newdata, level=0.95, 
                             print=FALSE, dec=3) {
  
  z = qnorm(1-(1-level)/2) 
  pr = predict(model, newdata=newdata, se.fit=TRUE)
  prediction = inv.logit(pr$fit)
  lower = inv.logit(pr$fit - z * pr$se.fit)
  upper = inv.logit(pr$fit + z * pr$se.fit)
  tmp = data.frame(newdata, pred=round(prediction, dec), 
                   lower=round(lower,dec), upper=round(upper,dec))
  if (print) print(tmp)
  list(prediction=prediction, lower=lower, upper=upper)
} 

pr = predict.interval(model_1, newdata=test)
pr.model_1 = pr$prediction
pr = predict.interval(nullmod, newdata=test)
pr.nullmod = pr$prediction
pr = predict.interval(aicmod, newdata=test)
pr.aicmod = pr$prediction
pr = predict.interval(bicmod, newdata=test)
pr.bicmod = pr$prediction

brier.model_1 = mean( (test$area_dummy - pr.model_1)^2)
brier.null = mean( (test$area_dummy - pr.nullmod)^2)
brier.aic = mean( (test$area_dummy - pr.aicmod)^2)
brier.bic = mean( (test$area_dummy - pr.bicmod)^2)

cbind(brier.model_1, brier.null, brier.aic, brier.bic) # null model and bicmodel had the same and best brier score followed by aicmod. 
compareGLM(aicmod, bicmod, nullmod, model_1, model_7) # # Comparing and ranking models. BIC the best followed by aicmod. 

summary(aicmod)
anova(aicmod, test = "Chisq")
d2(aicmod)
cbind(brier.aic)
exp(cbind(coef(aicmod),confint(aicmod))) # calculates odds ration with confidence levels. 

par(mfrow=c(1,1))
hnp(aicmod)
par(mfrow=c(2,2))
plot(aicmod)
plot(allEffects(aicmod))
binned_residuals(aicmod)
car::vif(aicmod) # VIF test for collinear variables to remove (all fine as below all variables below 10). 
check_model(aicmod)
simulateResiduals(aicmod, plot = TRUE)# DHRAMa plots check for overdispersion and outliers.
summary(aicmod)$deviance/summary(aicmod)$df.residual # Ratio of the residual deviance to the residual degrees of freedom test.
outlierTest(aicmod) # Outlier Test.

head(predict(aicmod, type = "response")) # returns probabilities.
trn_pred <- ifelse(predict(aicmod, type = "response") > 0.5, "yes", "no") # converts probabilities to classifications. # predictions with “yes” (over 0.5) predicting a forest fire area larger than 1 ha and “no” (under 0.5) predicting a forest fire area under 1 ha. 
head(trn_pred) # predictions in "yes" and "no".

trn_tab <- table(predicted = trn_pred, actual = train$area_dummy) # Making predictions on the train set.
trn_tab # Confusion Matrix for train set.

tst_pred <- ifelse(predict(aicmod, newdata = test, type = "response") > 0.5, "yes", "no") # converts probabilities to classifications. 
tst_tab <- table(predicted = tst_pred, actual = test$area_dummy)# Making predictions on the test set.
tst_tab # Confusion Matrix for test set.
pr <- predict(aicmod, x, type="response") 
head(round(pr, 2)) # predictions from test set producing numeric values. 

par(mfrow=c(1,1))
hist(pr, breaks=20)
hist(pr[fire_data$area_dummy==TRUE], ylim =(0:200), col="red", breaks=20, add=TRUE)

mean(tst_pred == test$area_dummy)

calc_class_err <- function(actual, predicted) { mean(actual != predicted)
} # function to calculate error and accuracy rates.  

calc_class_err(actual = train$area_dummy, predicted = trn_pred) # error rate for train set. 
calc_class_err(actual = test$area_dummy, predicted = tst_pred) # Test error rate should be close to train error rate. 

test_prob <- predict(aicmod, newdata = test, type = "response") # ROC and AUC curves to test sensitivity and specificity. 
test_roc <- roc(test$area_dummy ~ test_prob, plot = TRUE, print.auc = TRUE)

# Works Cited
Hartig, F., 2016. theoretical ecology. [Online] 
Available at: https://theoreticalecology.wordpress.com/2016/08/28/dharma-an-r-package-for-residual-diagnostics-of-glmms/
[Accessed 22 January 2022].
Kasza, J., 2015. Stata tip 125: Binned residual plots for assessing the fit of regression models for binary outcomes. The Stata Journal, 15(2), pp. 599-604.
Mangiafico, S. S., 2015. An R Companion for the Handbook of Biological Statistics. [Online] 
Available at: https://rcompanion.org/rcompanion/e_07.html
[Accessed 28 January 2022].
Morais, P. C. a. A., 2007. A Data Mining Approach to Predict Forest Fires using Meteorological Data, s.l.: s.n.
Peng, Y. D. a. C.-Y. J., 2013. Principled missing data methods for researchers. Springerplus, 2(222).

