# Overview
This is a project for Data Science 80 at UCSD where I will be working with a raw dataset from food.com and perform data exploratoration to find the best prediction model
# Introduction
Cooking is an all rounded skill that teaches one about nourishing the body, nutritional awareness, and even as a stress or creative outlet. It gives us the ability to provide for ourselves and loved ones with our own hands, and only through practice and exploration of recipes and cuisines can we really learn about our preferences. Recipes will have everything laid out for us from the nutritional information down to the step by step processs; the knowledge gained from working with ingredients repeatedly and exposure to new ones supplies the user with the capability to go out and build a grocery list that caters to themselves. In this data exploration I will explore the **calories of a recipe**. The raw dataset from food.com consists of a recipes and ratings which date back to 2018. 
The <mark>recipes</mark> dataset has 83782 rows with each row representing a unique recipe. The 12 columns of <mark>recipes</mark> is shown below. 
|Column|Description|
|-----------|-----------|
|<mark>'name'</mark>|Recipe name|
|<mark>'id'</mark>|Recipe ID|
|<mark>'minutes'</mark>|Minutes to prepare recipe|
|<mark>'contributor_id'</mark>|User ID who submitted this recipe|
|<mark>'submitted'</mark>|Date recipe was submitted|
|<mark>'tags'</mark>|Food.com tags for recipe|
|<mark>'nutrition'</mark>|Nutrition information in the form \[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”|
|<mark>'n_steps'</mark>|Number of steps in recipe|
|<mark>'steps'</mark>|Text for recipe steps, in order|
|<mark>'description'</mark>|User-provided description|
|<mark>'ingredients'</mark>|Text of ingredients|
|<mark>'n_ingredients'</mark>|Number of ingredients in recipe|

The <mark>interactions</mark> dataset has 731927 with each row representing a review on a specified recipe. The 5 columns of <mark>interactions</mark> is shown below.
|Column|Description|
|-----------|-----------|
|<mark>'user_id'</mark>|User ID|
|<mark>'recipe_id'</mark>|Recipe ID|
|<mark>'date'</mark>|Date of interaction|
|<mark>'rating'</mark>|Rating given|
|<mark>'review'</mark>|Review text|
### First Part
This will be the data cleaning and exploration in discovering relationships between the features of our dataset. 
### Second Part 
This will be the assessment of Missingness of certain columns and then proceed to run a hypothesis test answering the question: **What is the relationship between calories and average rating of recipes?**
### Last Part 
This will be the creation of my prediction model focused on predicting the calories of a recipe. 
# Data Cleaning and Exploration 
## Cleaning 
1. Left merge <mark>recipes</mark> dataset with <mark>interactions</mark> dataset. Now the reviews of a recipe along with the rating of the recipe in our new dataframe <mark>food</mark> with __ rows.
2. Replace all 0 values in 'rating' column with np.nan values. This is important to do because a rating of 0 doesnt mean that it was rated very lowly therefore considered "bad" recipe, but rather that no rating exists for that particular recipe.
3. Find the average rating of a recipe and add that as a column. This is important to do because the same recipe can appear more than once because there may be multiple reviews thus multiple ratings from different users for one recipe.  
4. Shown below is the starting data type of all columns in our newly merged dataset <mark>food</mark>.

# Problem Identification
I plan to **predict the calories of a recipe** which is a regression problem. The response variable I chose is the calories of a recipe because for many people an important factor in creating meals is the amount of calories they are building in their meals; being able to identify the calories of a meal is a distinction that is of interest for some. Whether it's to be more nutritionally aware or watching weight. The metric I am using to evaluate my model is the **Root Mean Squared Error** because calories is a continuous numerical value. The information I know and is available to use before I train the model are all the columns I named above under **Intoduction**.
# Baseline Model 
For my baseline model I am untilizing a Linear Regression model and spliting the dataset into testing and training sets. The features in the baseline model are total fats, carbohydrates, and proteins in percentage of daily value \(PDV). All 3 features are quantitative however I chose to standardize these features because carbohydrates averaged about 12 while proteins and total fats averaged about 30. 
The RMSE of the model is **28.94** which is a good model because I recognize that RMSE is in the units of our original y data which is calories. So, our RMSE is telling me that the predicted values are about 28.94 calories off from the actual calorie values, which is not a lot in terms of calories. 
# Final Model 
The final model uses the standardized features <mark>total fats (PDV)</mark>, <mark>carbohydrates (PDV)</mark>, <mark>protein (PDV)</mark>, and <mark>sugar (PDV)</mark>; as well as the binarized feature <mark>n_ingredients</mark>. I found these as my best hyperparameters from performing an iterative **crossed validation score** with 5 folds. I incremented my features as follows: 
1. stdscalar total fats only
2. stdscalar total fats + carbs
3. stdscalar total fats + carbs + protein
4. stdscalar total fats + carbs + proteins + sugars
5. **stdscalar total fats + carbs + proteins + sugars + binarized n_ingredients**
6. stdscalar total fats + carbs + proteins + sugars + binarized n_ingredients + stdscalar minutes

I chose the obvious features of fats, carbohydrates, proteins, and sugars because calories are essentially the measure of the amount of these nutrients in food. Then I considered the number of ingredients as the next most important when predicting the calories of a recipe since more added material to a recipe will increase the calories; however, if the number of ingredients is high because of something like onions, garlic, seasonings, etc then those will not contribute to a higher calorie count, so I will train my model to see if the feature <mark>n_ingredients</mark> actually contributes to better predictions. The last feature I considered is how long a recipe takes to make, <mark>minutes</mark>. Generally, foods that are more nutritionally dense like chicken take longer to cook or desserts that have high amounts of sugar and are baked also take a long time to cook. On the flip side foods like steak are nutritionally dense but are done cooking fast, so <mark>minutes</mark> may not be an important feature to add in our model.

I continued to use the RMSE metric to evaluate my final model. The RMSE of my final model is **28.78** which is 0.16 better than the baseline model.

