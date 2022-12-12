# Predicting-hotel-booking-cancellation
Predicting Hotel Booking Cancellations
Nesara Kashyap, Omkar Pradhan, Francesco Boccardo

Summary

No-shows or cancellations result in a sizable portion of hotel reservations being canceled. Common reasons for cancellation include change of plans, availability of better options, unforeseen circumstances etc. Many of the hotels charge a minimal amount for cancellations, hence last-minute cancellation has a huge impact on hotel revenues, because the canceled bookings generally end up remaining vacant. We use the ‘Hotel Booking Cancellation’ dataset [1] which has historical information of real time booking cancellations of both a city hotel and a resort hotel. It has 32 attributes like cost of booking, room type, number of adults, number of children, gender, previous booking canceled or not etc. The goal of this project is to investigate which features/variables of the dataset have higher effects on booking cancellations, and to use a machine learning model to try to predict which booking is more likely to be canceled. 

In order to predict whether booking will be canceled or not, we implemented classification models like Logistic Regression, Decision Trees, KNN, Random Forest and Adaboost. Our findings have led us to narrow down the most important features from 32 to 5 which would make hotel managers’ job of looking out for potential cancellations easier. Our models have all performed well with F1 scores above 90%. Although this means that any of the models implemented could be used to accurately predict cancellations, the best performing model was Adaboost with an F1 score of 99%. 
