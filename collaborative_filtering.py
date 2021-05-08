#-------------------------------------------------------------------------
# AUTHOR: Charles Kypros
# FILENAME: collaborative_filtering.py
# SPECIFICATION: Collaborative Filtering
# FOR: CS 4200- Assignment #5
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
X_training = np.array(df.values)
X_training_floats = []
for val in X_training:
   float_val = []
   for digit in val:
      if isinstance(digit, float):
         float_val.append(digit)
   X_training_floats.append(float_val)

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
vec1 = np.array([X_training_floats[99]])
simularities = []
for i in range(99): 
   vec2 = np.array([X_training_floats[i]])
   simularities.append(cosine_similarity(vec1, vec2)[0][0])
simularities = sorted(simularities)

   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here
simularities_top_ten = simularities[-10:]

   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here
numerator_gallery = 0
numerator_restaurant = 0
denominator_gallery = 0
denominator_restaurant = 0

for simularity in simularities_top_ten:
   index = simularities.index(simularity)
   gallery_rating = X_training_floats[index][0]
   restaurant_rating = X_training_floats[index][5]
   average_rating = mean(X_training_floats[index])

   numerator_gallery += simularity * (gallery_rating - average_rating)
   numerator_restaurant += simularity * (restaurant_rating - average_rating)
   denominator_gallery += simularity
   denominator_restaurant += simularity
predicted_gallery = mean(X_training_floats[-1]) + (numerator_gallery / denominator_gallery)
predicted_restaurant = mean(X_training_floats[-1]) + (numerator_restaurant / denominator_restaurant)
print("Predicted gallery rating: ", predicted_gallery)
print("Predicted restaurant rating: ", predicted_restaurant)

   


