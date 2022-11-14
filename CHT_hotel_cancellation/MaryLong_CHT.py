#!/usr/bin/env python
# coding: utf-8

# # **Project - Classification and Hypothesis Testing: Hotel Booking Cancellation Prediction** - Mary Long | November 2022
# 
# ## **Marks: 40**
# 
# ---------------
# ## **Problem Statement**
# 
# ### **Context**
# 
# **A significant number of hotel bookings are called off due to cancellations or no-shows.** Typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost. This may be beneficial to hotel guests, but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.
# 
# The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.
# 
# This pattern of cancellations of bookings impacts a hotel on various fronts:
# 1. **Loss of resources (revenue)** when the hotel cannot resell the room.
# 2. **Additional costs of distribution channels** by increasing commissions or paying for publicity to help sell these rooms.
# 3. **Lowering prices last minute**, so the hotel can resell a room, resulting in reducing the profit margin.
# 4. **Human resources to make arrangements** for the guests.
# 
# ### **Objective**
# 
# This increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal - they are facing problems with this high number of booking cancellations and have reached out to your firm for data-driven solutions. You, as a Data Scientist, have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.
# 
# 
# ### **Data Description**
# 
# The data contains the different attributes of customers' booking details. The detailed data dictionary is given below:
# 
# 
# **Data Dictionary**
# 
# * **Booking_ID:** Unique identifier of each booking
# * **no_of_adults:** Number of adults
# * **no_of_children:** Number of children
# * **no_of_weekend_nights:** Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# * **no_of_week_nights:** Number of weekday nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# * **type_of_meal_plan:** Type of meal plan booked by the customer:
#     * Not Selected – No meal plan selected
#     * Meal Plan 1 – Breakfast
#     * Meal Plan 2 – Half board (breakfast and one other meal)
#     * Meal Plan 3 – Full board (breakfast, lunch, and dinner)
# * **required_car_parking_space:** Does the customer require a car parking space? (0 - No, 1- Yes)
# * **room_type_reserved:** Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# * **lead_time:** Number of days between the date of booking and the arrival date
# * **arrival_year:** Year of arrival date
# * **arrival_month:** Month of arrival date
# * **arrival_date:** Date of the month
# * **market_segment_type:** Market segment designation.
# * **repeated_guest:** Is the customer a repeated guest? (0 - No, 1- Yes)
# * **no_of_previous_cancellations:** Number of previous bookings that were canceled by the customer prior to the current booking
# * **no_of_previous_bookings_not_canceled:** Number of previous bookings not canceled by the customer prior to the current booking
# * **avg_price_per_room:** Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# * **no_of_special_requests:** Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# * **booking_status:** Flag indicating if the booking was canceled or not.

# ## **Importing the libraries required**

# In[203]:


# Importing the basic libraries we will require for the project

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the Machine Learning models we require from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Importing the other functions we may require from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

# To get diferent metric scores
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,plot_confusion_matrix,precision_recall_curve,roc_curve,make_scorer

# Code to ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')


# ## **Loading the dataset**

# In[204]:


hotel = pd.read_csv("INNHotelsGroup.csv")


# In[205]:


# Copying data to another variable to avoid any changes to original data
data = hotel.copy()


# ## **Overview of the dataset**

# ### **View the first and last 5 rows of the dataset**
# 
# Let's **view the first few rows and last few rows** of the dataset in order to understand its structure a little better.
# 
# We will use the head() and tail() methods from Pandas to do this.

# In[206]:


data.head()


# In[207]:


data.tail()


# ### **Understand the shape of the dataset**

# In[208]:


data.shape


# * The dataset has 36275 rows and 19 columns. 

# ### **Check the data types of the columns for the dataset**

# In[209]:


data.info()


# * `Booking_ID`, `type_of_meal_plan`, `room_type_reserved`, `market_segment_type`, and `booking_status` are of object type while rest columns are numeric in nature.
# 
# * There are no null values in the dataset.

# ### **Dropping duplicate values**

# In[210]:


# checking for duplicate values
data.duplicated().sum()


# - There are **no duplicate values** in the data.

# ### **Dropping the unique values column**

# **Let's drop the Booking_ID column first before we proceed forward**, as a column with unique values will have almost no predictive power for the Machine Learning problem at hand.

# In[211]:


data = data.drop(["Booking_ID"], axis=1)


# In[212]:


data.head()


# ### **Question 1: Check the summary statistics of the dataset and write your observations (2 Marks)**
# 
# 

# **Let's check the statistical summary of the data.**

# In[213]:


data.describe().T

The average number of adults is less than 2, with a maximum of 4.
The average number of children is less than 1, with a maximum of 10.
So it sounds like most of the guests are 1-2 people traveling with no children but there are a couple of families and large groups.

People more commonly stay on weeknights than weekends - the maximum number of weeknights is 17 and weekends is 7.

It is a surpringly low number of people who bring a car and require a parking space; the maximum number of parking spaces required is 1. I would have expected a number much closer to a majority percentage. Is this hotel near an airport or other transit? It must be accessible if very few of the guests are bringing a car.

Everyone visited in 2017 or 2018, and very few were repeated guests.

The average price per room is around $103 per night +- $35 with a maximum of $540. Most of them fell within the $99-120 range.
The maximum number of special requests was 5 and most people who made requests only had 1 request (75% of those who made requests only made 1 request).

Only about 2.5% of guests were repeated guests.
~15% of previous bookings were maintained, i.e. not canceled. The maximum number of bookings which were kept and not canceled by one guest was 58.
The maximum number of cancellations by one guest was 13. Around a third of rooms were cancelled.
# ## **Exploratory Data Analysis**

# ### **Question 2: Univariate Analysis**

# Let's explore these variables in some more depth by observing their distributions.

# We will first define a **hist_box() function** that provides both a boxplot and a histogram in the same visual, with which we can perform univariate analysis on the columns of this dataset.

# In[214]:


# Defining the hist_box() function
def hist_box(data,col):
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12,6))
  # Adding a graph in each part
  sns.boxplot(data[col], ax=ax_box, showmeans=True)
  sns.distplot(data[col], ax=ax_hist)
  plt.show()


# #### **Question 2.1:  Plot the histogram and box plot for the variable `Lead Time` using the hist_box function provided and write your insights. (1 Mark)**

# In[215]:


hist_box(data, "lead_time") 


# Lead Time looks to be somewhat normally distributed and skewed to the left. 
# There are some outliers for the variable as we approach and go over 300 days lead time, but these are close to zero, but not non-zero. Most of the leads happened within about 120 days or less but between 0 and about 400 days for the dataset. There is a large and significant amount of leads with 0 days time.
# Thus, we conclude that lead time matters and most of the bookings occured within 0-100 days, so perhaps they can focus their advertising or other strategies for about 0-3 months out.

# #### **Question 2.2:  Plot the histogram and box plot for the variable `Average Price per Room` using the hist_box function provided and write your insights. (1 Mark)**

# In[216]:


hist_box(data, "avg_price_per_room")


# The average price per room appears to be normally distributed with the data centered around $100 per night and ranging from $0 to in the mid $500s. Nearly all of the rooms are under $200 per night. There are no rooms between $400 and $500 and very few above $500. Some rooms have a $0 and are complementary, and are worth looking into.

# **Interestingly some rooms have a price equal to 0. Let's check them.**

# In[217]:


data[data["avg_price_per_room"] == 0]


# - There are quite a few hotel rooms which have a price equal to 0.
# - In the market segment column, it looks like many values are complementary.

# In[218]:


data.loc[data["avg_price_per_room"] == 0, "market_segment_type"].value_counts()


# * It makes sense that most values with room prices equal to 0 are the rooms given as complimentary service from the hotel.
# * The rooms booked online must be a part of some promotional campaign done by the hotel.

# In[219]:


# Calculating the 25th quantile
Q1 = data["avg_price_per_room"].quantile(0.25)

# Calculating the 75th quantile
Q3 = data["avg_price_per_room"].quantile(0.75)

# Calculating IQR
IQR = Q3 - Q1

# Calculating value of upper whisker
Upper_Whisker = Q3 + 1.5 * IQR
Upper_Whisker


# In[220]:


# assigning the outliers the value of upper whisker
data.loc[data["avg_price_per_room"] >= 500, "avg_price_per_room"] = Upper_Whisker


# #### **Let's understand the distribution of the categorical variables**

# **Number of Children**

# In[221]:


sns.countplot(data['no_of_children'])
plt.show()


# In[222]:


data['no_of_children'].value_counts(normalize=True)


# * Customers were not travelling with children in 93% of cases.
# * There are some values in the data where the number of children is 9 or 10, which is highly unlikely. 
# * We will replace these values with the maximum value of 3 children.

# In[223]:


# replacing 9, and 10 children with 3
data["no_of_children"] = data["no_of_children"].replace([9, 10], 3)


# **Arrival Month**

# In[224]:


sns.countplot(data["arrival_month"])
plt.show()


# In[225]:


data['arrival_month'].value_counts(normalize=True)


# * October is the busiest month for hotel arrivals followed by September and August. **Over 35% of all bookings**, as we see in the above table, were for one of these three months.
# * Around 14.7% of the bookings were made for an October arrival.

# **Booking Status**

# In[226]:


sns.countplot(data["booking_status"])
plt.show()


# In[227]:


data['booking_status'].value_counts(normalize=True)


# * 32.8% of the bookings were canceled by the customers.

# **Let's encode Canceled bookings to 1 and Not_Canceled as 0 for further analysis**

# In[228]:


data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
)


# ### **Question 3: Bivariate Analysis**

# #### **Question 3.1: Find and visualize the correlation matrix using a heatmap and write your observations from the plot. (2 Marks)**
# 
# 

# In[229]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(data[cols_list].corr(), annot=True, vmin=-1,vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# The most strongly negatively correlated variables were arrival month and year. Most folks arrived in October, followed by September and then August, which makes sense. The worst travel month was January. Thus, we conclude that timing is very important and we can adjust our advertising and strategies for other times of the year or potentially charge more during the fall.
# 
# The price per room, number of adults and number of children are correlated. Typically, hotels charge more per person (typically something like $5 per adult. Sometimes kids are free.)
# The price per room changes with the year.
# 
# Repeated guests, lead time and average price per room are somewhat negatively correlated.
# Whether a guest is repeated or not appears to have little to nothing to do with the time of month or year of arrival. Time of arrival isn't much of a factor, although most booking are in the fall.
# 
# It appears that booking status is most strongly correlated with lead time, followed by arrival year and average price per room. They are positively correlated.
# Booking status is negatively correlated most strongly with number of special requests.

# **Hotel rates are dynamic and change according to demand and customer demographics. Let's see how prices vary across different market segments**

# In[230]:


plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data, x="market_segment_type", y="avg_price_per_room", palette="gist_rainbow"
)
plt.show()


# * Rooms booked online have high variations in prices.
# * The offline and corporate room prices are almost similar.
# * Complementary market segment gets the rooms at very low prices, which makes sense.

# We will define a **stacked barplot()** function to help analyse how the target variable varies across predictor categories.

# In[231]:


# Defining the stacked_barplot() function
def stacked_barplot(data,predictor,target,figsize=(10,6)):
  (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
  plt.legend(loc="lower right")
  plt.ylabel('Percentage Cancellations %')


# #### **Question 3.2: Plot the stacked barplot for the variable `Market Segment Type` against the target variable `Booking Status` using the stacked_barplot  function provided and write your insights. (1 Mark)**

# In[232]:


stacked_barplot(data, "market_segment_type", "booking_status")


# Even though about 1/3 of the rooms were cancelled, it appears that none of them (zero) came from any of the complmentary category.
# Online bookings had the most cancellations, followed by offline, aviation, and then corporate. Offline and aviation bookings are roughly the same. Corporate bookings are less than half of the cancellations than any other category (except complementary) bookings. Corporate bookings appear to be the most reliable.
# 
# So we should focus on addressing online, offline and aviation bookings to reduce cancellations.

# #### **Question 3.3: Plot the stacked barplot for the variable `Repeated Guest` against the target variable `Booking Status` using the stacked_barplot  function provided and write your insights. (1 Mark)**
# 
# Repeating guests are the guests who stay in the hotel often and are important to brand equity.

# In[233]:


stacked_barplot(data, "repeated_guest", "booking_status")


# Hardly any repeated guests cancelled. This means, that for those who actually kept their reservations, customer retention is high. We can interpret this to mean that the hotel is probably doing a fairly decent job and there are no immediately obvious red flags.
# 
# Most of our cancellations are from first time guests. We should focus on reducing these cancellations, or preventing wrong-fit customers from booking in the first place. Perhaps there can be better screening and/or recommendations, or a change in marketing around branding so we only attract the right-fit customers. We may also aim to receive more corporate clients as well, given the previous data, as corporate bookings are associated with lowest rates of cancellations at this hotel. Perhaps we can market to corporations or local chambers of commerce, small business associations or large businesses.
# 
# Given that the lead time, time of year/month are most correlated with cancellations, and most of the cancellations are happening with new (non-repeat) customers, this can be fairly predictable and thus we can take action to manage it based on the time of year. It looks like a marketing problem, with brand fit. Why are so many folks booking the hotel but canceling? They may solve the problem by instituting a simple no-refunds policy or a cancelation fee, some kind of negative incentive to prevent customers from canceling -- and thus, those who arent' serious about intending their stay are deterred from booking.
# They don't appear to have any problems with customer retensions, so I don't believe this strategy will create them any problems.
# 
# So what time of year do they experience the most cancellations? What months?
# We can intelligently interpret this to cycle with holidays, school starting, vacation times, etc and market accordingly. In what month do the most cancellations occur? Is this the same year to year? If it's cyclical, we can adjust our cancellation policy, and say, charge an extra cancellation fee or no-show fee around the holidays or other peak times. If it's not cyclical and is simply random, then we can look at other factors and adjust our strategies accordingly. It looks like they aren't getting a solid customer fit on the marketing side, else they'd be getting more loyalists.
# Do they have a customer advantage (like Points and rewards) program? This could come in after the deterrents to help build loyalty, though it seems like they're doing okay with that already.
# 
# It looks to me like new customers are booking it, sight unseen and not being held accountable or given any incentives not to cancel, or not to book in the first place.
# They may also improve upon this by adding in reputation and reviews to shift their customer base towards more loyalists from corporate, or consider adjusting their relationships with airline and online commission partners. I am sure they are paying a commission fee penalty too when there are cancellations. Perhaps they can investigate their commission rates or commission cancellation policies with third parties like Expedia, etc - whoever their airline booking partners are.
# This would be an interesting conversation to address, "can we select a better, more reliable airline booking partner?" "can we select a better online partner?". It is not clear how they are booking offline, and whether offline comes from partners or if they are booking directly with the hotel.
# 
# An anecdote:
# For example, I recently stayed at a hotel where I booked through Expedia, but I selected the wrong dates.
# Expedia had a no-problem free cancellation policy, but the hotel didn't.Even after calling Expedia AND the hotel manager, and escalating it to their managers,  I lost my $100+ booking simply becuase I chose the wrong dates. 
# I certainly learned my lesson and will be choosing the dates more carefully when I book.
# The hotel explained to me that it is because they pay the third party (Expedia in this case, my personal anecdote story) a commission fee for publicity and the sale, regardless of whether I show up or cancel. They were unwilling to take the loss, so it was on me.) This is one way to deter customers booking erroneously or in a non-committal way. From my customer experience, it was nice having a "get out of it, free" card from Expedia, but accountability from the hotel (I happened to already be staying at for multiple nights as a repeat guest). I was not able to transfer my booking to a different date. They gave me the option to change the booking for that night, but I had already paid. This came up, because my partner is a construction worker and we frequently stay in hotels and extend for 1-2 days at a time, ending up staying for weeks or months on end but booking in 1-2 day increments.
# In this case, I accidentally booked two weeks in advance, because Expedia had the dates set for two weeks out on their platform, rather than that night. It was 10:30AM and I hadn't had coffee, worked all night, just got out of night shift to extend my hotel before the 11AM checkout. We work nights and construction bossman said we had to stay another day because the work wasn't finished. I share this personal anecdote here because I see a HUGE number of construction workers doing the same thing (extending a few days at a time on short or no notice, last minute) due to the nature of the job and lifestyle! We fall into the corporate loyalist category with very few cancellations.

# **Let's analyze the customer who stayed for at least a day at the hotel.**

# In[234]:


stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data["total_days"] = (stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"])

stacked_barplot(stay_data, "total_days", "booking_status",figsize=(15,6))


# * The general trend is that the chances of cancellation increase as the number of days the customer planned to stay at the hotel increases.

# **As hotel room prices are dynamic, Let's see how the prices vary across different months**

# In[235]:


plt.figure(figsize=(10, 5))
sns.lineplot(y=data["avg_price_per_room"], x=data["arrival_month"], ci=None)
plt.show()


# * The price of rooms is highest in May to September - around 115 euros per room.

# ## **Data Preparation for Modeling**
# 
# - We want to predict which bookings will be canceled.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# **Separating the independent variables (X) and the dependent variable (Y)**

# In[236]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True) # Encoding the Categorical features


# **Splitting the data into a 70% train and 30% test set**
# 
# Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use the **stratified sampling** technique to ensure that relative class frequencies are approximately preserved in each train and validation fold.

# In[237]:


# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,stratify=Y, random_state=1)


# In[238]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# ## **Model Evaluation Criterion**
# 
# #### **Model can make wrong predictions as:**
# 
# 1. Predicting a customer will not cancel their booking but in reality, the customer will cancel their booking.
# 2. Predicting a customer will cancel their booking but in reality, the customer will not cancel their booking. 
# 
# #### **Which case is more important?** 
# 
# Both the cases are important as:
# 
# * If we predict that a booking will not be canceled and the booking gets canceled then the hotel will lose resources and will have to bear additional costs of distribution channels.
# 
# * If we predict that a booking will get canceled and the booking doesn't get canceled the hotel might not be able to provide satisfactory services to the customer by assuming that this booking will be canceled. This might damage brand equity. 
# 
# 
# 
# #### **How to reduce the losses?**
# 
# * The hotel would want the `F1 Score` to be maximized, the greater the F1  score, the higher the chances of minimizing False Negatives and False Positives. 

# **Also, let's create a function to calculate and print the classification report and confusion matrix so that we don't have to rewrite the same code repeatedly for each model.**

# In[239]:


# Creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Cancelled', 'Cancelled'], yticklabels=['Not Cancelled', 'Cancelled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# ## **Building the model**
# 
# We will be building 4 different models:
# 
# - **Logistic Regression**
# - **Support Vector Machine (SVM)**
# - **Decision Tree**
# - **Random Forest**

# ### **Question 4: Logistic Regression (6 Marks)**

# #### **Question 4.1: Build a Logistic Regression model (Use the sklearn library) (1 Mark)**

# In[240]:


# Fitting logistic regression model
lg = LogisticRegression()
lg.fit(X_train,y_train)


# #### **Question 4.2: Check the performance of the model on train and test data (2 Marks)**

# In[241]:


# Checking the performance on the training data
y_pred_train = lg.predict(X_train)
metrics_score(y_train, y_pred_train)


# ORIGINAL TRAIN MODEL: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# Let's check the performance on the test set

# In[242]:


# Checking the performance on the test dataset
y_pred_test = lg.predict(X_test)
metrics_score(y_test, y_pred_test)


# Using the model with the default threshold gives a decent precision but mediocre recall score.
# 
# TEST MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and recall score of 58%.
# We have been able to predict who will NOT cancel with a precision of 81% and a recall score of 89%.
# 
# TRAIN MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# 
# #### **Question 4.3: Find the optimal threshold for the model using the Precision-Recall Curve. (1 Mark)**
# 
# Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
# 
# Let's use the Precision-Recall curve and see if we can find a **better threshold.**
# 

# In[243]:


# Predict_proba gives the probability of each observation belonging to each class
y_scores_lg=lg.predict_proba(X_train)

precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# We want to choose a threshold that has a high recall while also having a good in precision. The optimal threshold looks to be a little bit over 0.4.

# In[244]:


# Setting the optimal threshold
optimal_threshold = 0.4


# #### **Question 4.4: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# In[245]:


# Creating confusion matrix
y_pred_train = lg.predict_proba(X_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)


# Yes, that's better -- we were able to achieve a significant improvement,
# 
# TRAIN MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 67% and recall score of 69%.
# We have been able to predict who will NOT cancel with a precision of 85% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced. 
# 
# for reference,
# TEST MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and recall score of 58%.
# We have been able to predict who will NOT cancel with a precision of 81% and a recall score of 89%.
# 
# TRAIN MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# Let's check the performance on the test set

# In[246]:


y_pred_test = lg.predict_proba(X_test)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)


# Yes, that's better -- we were able to achieve a significant improvement,
# THIS TEST MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 66% and recall score of 68%.
# We have been able to predict who will NOT cancel with a precision of 84% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced and is improved overall.
# 
# 
# TRAIN MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 67% and recall score of 69%.
# We have been able to predict who will NOT cancel with a precision of 85% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced. 
# 
# for reference,
# TEST MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and recall score of 58%.
# We have been able to predict who will NOT cancel with a precision of 81% and a recall score of 89%.
# 
# TRAIN MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# ### **Question 5: Support Vector Machines (11 Marks)**

# To accelerate SVM training, let's scale the data for support vector machines.

# In[247]:


scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)


# Let's build the models using the two of the widely used kernel functions:
# 
# 1.   **Linear Kernel**
# 2.   **RBF Kernel**
# 
# 

# #### **Question 5.1: Build a Support Vector Machine model using a linear kernel (1 Mark)**

# **Note: Please use the scaled data for modeling Support Vector Machine**

# In[248]:


svm = SVC(kernel='linear',probability=True) # Linear kernal or linear decision boundary
model = svm.fit(X=X_train_scaled, y = y_train)


# #### **Question 5.2: Check the performance of the model on train and test data (2 Marks)**

# In[249]:


y_pred_train_svm = model.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# The original model achieved a 72% precision and 58% recall.
#  achieved a 74% precision and 61% recall. It is an decrease of 1% in both categories as compared to the original linear model. This model is not better than our original linear model.
#  
# 
# THIS SVM MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 74% and recall score of 61%.
# We have been able to predict who will NOT cancel with a precision of 83% and a recall score of 90%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced and is improved overall.
# 
# 
# TEST MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 66% and recall score of 68%.
# We have been able to predict who will NOT cancel with a precision of 84% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced and is improved overall.
# 
# 
# TRAIN MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 67% and recall score of 69%.
# We have been able to predict who will NOT cancel with a precision of 85% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced. 
# 
# for reference,
# TEST MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and recall score of 58%.
# We have been able to predict who will NOT cancel with a precision of 81% and a recall score of 89%.
# 
# TRAIN MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# Checking model performance on test set

# In[250]:


y_pred_test_svm = model.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a recall score of 74% and a precision rate of 61%.
# 
# 
# The original training model achieved a 72% precision and 58% recall. This model achieved the same.
# The original testing model achieved a 72% precision and 58% recall. This model achieved the same.
#  
# 
# SVM MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 74% and recall score of 61%.
# We have been able to predict who will NOT cancel with a precision of 83% and a recall score of 90%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced and is improved overall.
# 
# 
# TEST MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 66% and recall score of 68%.
# We have been able to predict who will NOT cancel with a precision of 84% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced and is improved overall.
# 
# 
# TRAIN MODEL2: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 67% and recall score of 69%.
# We have been able to predict who will NOT cancel with a precision of 85% and a recall score of 83%.
#  =10% increase in recall while only a small 6% drop in precision. The model is more balanced. 
# 
# for reference,
# TEST MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and recall score of 58%.
# We have been able to predict who will NOT cancel with a precision of 81% and a recall score of 89%.
# 
# TRAIN MODEL1: We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 73% and recall score of 59%.
# We have been able to predict who will NOT cancel with a precision of 82% and a recall score of 89%.

# #### **Question 5.3: Find the optimal threshold for the model using the Precision-Recall Curve. (1 Mark)**
# 

# In[251]:


# Predict on train data
y_scores_svm=model.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# The optimal threshold is just over 0.4

# In[252]:


optimal_threshold_svm=0.4


# #### **Question 5.4: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# In[253]:


y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a recall score of 69% and a precision rate of 71%. It is more balanced than the other models.
# 
# The original linear training model achieved a 72% precision and 58% recall.
# The original linear testing model achieved a 72% precision and 58% recall. 
# 
# This model is more balanced with a 1% reduction in recall and an 11% increase in precision.
#  

# In[254]:


y_pred_test = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a recall score of 68% and a precision rate of 71%. It is more balanced.

# #### **Question 5.5: Build a Support Vector Machines model using an RBF kernel (1 Mark)**

# In[255]:


svm_rbf=SVC(kernel='rbf',probability=True)
svm_rbf.fit(X_train_scaled,y_train)


# #### **Question 5.6: Check the performance of the model on train and test data (2 Marks)**
# 
# 

# In[256]:


y_pred_train_svm = svm_rbf.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 79% and a recall score of 65%. It is more balanced than the other models and improved by 7% precision and 7% recall compared to the original.
# 
# The original linear training model achieved a 72% precision and 58% recall.
# The original linear testing model achieved a 72% precision and 58% recall. 

# #### Checking model performance on test set

# In[257]:


y_pred_test = svm_rbf.predict(X_test_scaled)

metrics_score(y_test, y_pred_test)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 78% and a recall score of 63%.

# In[258]:


# Predict on train data
y_scores_svm=svm_rbf.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[259]:


optimal_threshold_svm=0.4


# #### **Question 5.7: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# In[260]:


y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 69% and a recall score of 71%.

# In[261]:


y_pred_test = svm_rbf.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 72% and a recall score of 73%. It is more balanced than the other models and improved by 15% recall compared to the original.
# 
# The original linear training model achieved a 72% precision and 58% recall.
# The original linear testing model achieved a 72% precision and 58% recall. 

# ### **Question 6: Decision Trees (7 Marks)**

# #### **Question 6.1: Build a Decision Tree Model (1 Mark)**

# In[262]:


model_dt = DecisionTreeClassifier(random_state=1)
model_dt.fit(X_train, y_train)


# #### **Question 6.2: Check the performance of the model on train and test data (2 Marks)**

# In[263]:


# Checking performance on the training dataset
pred_train_dt = model_dt.predict(X_train)
metrics_score(y_train, pred_train_dt)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 100% and a recall score of 99%.  Oh, sweet! A perfect fit!
# But this means that we have probably overfit the model... let's check the test dataset.

# #### Checking model performance on test set

# In[264]:


pred_test_dt = model_dt.predict(X_test)
metrics_score(y_test, pred_test_dt)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 79% and a recall score of 79%.
# 
# If we look at the training model with near-pefect accuracy(0.99 1.0) from the previous step, we can conclude that this model is overfit. We have overfit the model, so we need to prune...

#  #### **Question 6.3: Perform hyperparameter tuning for the decision tree model using GridSearch CV (1 Mark)**

# **Note: Please use the following hyperparameters provided for tuning the Decision Tree. In general, you can experiment with various hyperparameters to tune the decision tree, but for this project, we recommend sticking to the parameters provided.**

# In[265]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 7, 2),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}


# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, cv=5, scoring='recall', n_jobs=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### **Question 6.4: Check the performance of the model on the train and test data using the tuned model (2 Mark)**

# #### Checking performance on the training set 

# In[266]:


# Checking performance on the training dataset
dt_tuned = estimator.predict(X_train)
metrics_score(y_train, dt_tuned)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 82% and a recall score of 68%. It is more balanced than the other models and improved by 3% precision and 3% recall compared to the original.
# 
# The original linear training model achieved a 72% precision and 58% recall.
# The original linear testing model achieved a 72% precision and 58% recall. 

# In[267]:


# Checking performance on the training dataset
y_pred_tuned = estimator.predict(X_test)
metrics_score(y_test, y_pred_tuned)


# We have been able to build a predictive model that can be used by the hotel company to predict the customers who are likely to cancel with a precision rate of 82% and a recall score of 67%.

# #### **Visualizing the Decision Tree**

# In[268]:


feature_names = list(X_train.columns)
plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,max_depth=3,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# #### **Question 6.5: What are some important features based on the tuned decision tree? (1 Mark)**

# In[269]:


# Importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# The most important feature is lead time, followed by the Online market segment as a lead source.
# From there, get into the number of special requests, average price per room, and number of adults, whether or not it's a weekend and what time of year(month) it is.
# These are the biggest factors used in our prediction.
# 
#  Offline market segment follows; not of much importance.
# It is almost negligible whether or not they need a parking spot or if it's a weeknight.

# ___

# ### **Question 7: Random Forest (4 Marks)**

# #### **Question 7.1: Build a Random Forest Model (1 Mark)**

# In[270]:


rf_estimator = RandomForestClassifier(random_state=1)

rf_estimator.fit(X_train, y_train)


# #### **Question 7.2: Check the performance of the model on the train and test data (2 Marks)**

# In[271]:


y_pred_train_rf = rf_estimator.predict(X_train)
metrics_score(y_train, y_pred_train_rf)


# We were able to build a model with 100% precision and 99% recall to predict which customers will cancel on the training data.
# 

# In[272]:


y_pred_test_rf = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test_rf)


# We were able to build a model with 88% precision and 80% recall to predict which customers will cancel on the test data.

# #### **Question 7.3: What are some important features based on the Random Forest? (1 Mark)**

# Let's check the feature importance of the Random Forest

# In[273]:


importances = rf_estimator.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index=columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

plt.figure(figsize = (13, 13))

sns.barplot(importance_df.Importance, importance_df.index)


# The most important feature is lead time. Interesting, the Online market segment as a lead source is not as important here. It comes after the weekend nights marker. The price per room comes in as the second most important factor after the lead time.
# 
# From there, we get into the number of special requests, the month and date, and number of nights. Weeknights are actually a more important factor than the weekends.
# 
# The online market segment is next, followed by number of adults, the year, and the offline market.
# 
# Next is meal plans (type 2), room type (type 4) and and meal plans (not selected).
# It makes sense that number of children follow next, when considering the clients' meal plans.
# Next, follows car... we can imagine that a family with a minivan and several children might want a meal plan or at least a space to park.
# 
# 
# Finally, we have the corporate market, and room types 2 and 6.
# Whether or not they are a repeated guest or have previously cancelled is almost irrlevant.
# The least important, and almost negligible factor, is room type 5.

# ### **Question 8: Conclude ANY FOUR key takeaways for business recommendations (4 Marks)**

# 1) LEAD TIME (most important factor) && WEEKNIGHTS + PRICE PER ROOM
# ==> when generating profit models, I would recommend that they build a competitive price per room model that changes based on the day of the week and number of days booked in advance.
# These are the top three most important factors in this model.
# By strategically optimizing the price per room based on the time component [f(PPR | time + date components)] they can maximize the income of the price model.
# ==> Institute a cancellation policy such as,
# - No Refunds, or charge a cancellation fee.
# - Insist that rather than cancel, they must rebook the room for another night at the same hotel, based on the pricing model. This policy will be in effect whether they book directly at the hotel, via an offline source, an online source, corporate or even aviation. 
# - There may be a case for negotiating directly with corporate (who tends to cancel the LEAST out of all of them!) to get some special rates, benefits or perks. These are the least risky of clients.
# - Based on the high rates of cancellation within Online, Offline and Aviation -- they may want to consider negotiating directly with their third-party partners. There may be a way to negotiate lower costs in publicity & marketing via reduced commission fees paid out to these third party bookings. It is worth optimizing there and addressing this policy, for if and when a guest cancels. If a guest cancels, does the hotel still have to pay a commission to the third party Aviation, Online market partner, etc? Lower these costs.
# - Then, institute a "no-refunds" policy with the guest, saying, regardless of whether you book online or offline or through an aviation [or optionally, a corporate booking] partner, the cancellation is non-refundable. Or, they may even seek to go a step further and institute a change fee. This may have an impact on the kind of clientele willing to stay at the hotel; depending on the market and price points, it may or may not be desirable. This might be more attractive to corporate or business clients, than say a vacationing family. I would inquire whether the hotel is an "extended stay, suite with kitchen" type of place, or a Motel 6 style nightly turnover with pets. The hotel may choose to cater to a certain type of Client.
# 
# 2) STRATEGIC MARKETING BASED ON MONTH OF THE YEAR
# ==> Plan in advance for Fall (the most popular months are August-October), consider setting a higher price point during those times, and market accordingly for slower months such as January.
# 
# 3) Require a minimum deposit instead or in addition to the cancellation fee, such as, "if you book a reservation, you will be required to place a credit card for a deposit and charged a minimum of one night's fees, regardless of whether you show up". Charge a minimum of one night when they make the booking. 
# 
# 4) Meal plans are a considerable cost; It's not just food costs but kitchen and cleanup staff + licensing and health code regulations + cook time and labor.
# They should look into the costs to cook the meals, and solicit visitor feedback about meals. Send user surveys after the guests visit, and ask guests in advance what their meal preferences are. There may be ways to strategically partner with local restaurants or third parties (such as Expedia or their Online/Aviation booking partners) to provide bulk rates, as well as share social data metrics.
# Can they call their Aviation/Online partners and get them to share their social user data on their customers, to find out more about what they like to eat? They may optimie this and even offer coupons or discounts with local eateries (or have their chefs in-house, if that's more efficient) to promote the local economy if guests aren't eating in house.


#Thanks for this great lesson!


# ## **Happy Learning!**

#@marykslong

