# Overview
This is a project for Data Science 80 at UCSD where I will be working with a raw dataset from food.com and perform data exploratoration to find the best prediction model
# Introduction
Cooking is an all rounded skill that teaches one about nourishing the body, nutritional awareness, and even as a stress or creative outlet. It gives us the ability to provide for ourselves and loved ones with our own hands, and only through practice and exploration of recipes and cuisines can we really learn about our preferences. As nutritional awareness is one factor in deciding what to make for dinner; I wonder if recipes with higher calories will have more low ratings as a way of heeding caution or will it be the opposite with high ratings because more calories may be indicitive of better flavors or will there be no apparent relationship? In this data exploration I will explore the **relationship between calories and average rating of recipes**. The raw dataset from food.com consists of a recipes and ratings which date back to 2018. 
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

The <mark>interactions</mark> dataset has 731927 with each row representing a review on a specified recipe. The 5 columns of ==interactions is shown below.
|Column|Description|
|-----------|-----------|
|<mark>'user_id'</mark>|User ID|
|<mark>'recipe_id'</mark>|Recipe ID|
|<mark>'date'</mark>|Date of interaction|
|<mark>'rating'</mark>|Rating given|
|<mark>'review'</mark>|Review text|
### First Part
This will be the data cleaning and exploration in discovering possible relationships between the rating of a recipe and the varying types (and numbers) of features. 
### Second Part 
This will be the assessment of Missingness of certain columns and then proceed to run a hypothesis test answering the question: **What is the relationship between calories and average rating of recipes?**
### Last Part 
This will be the creation of my prediction model focused on predicting the rating of a recipe. 
# Data Cleaning and Exploration 
## Cleaning 
1. Left merge <mark>recipes</mark> dataset with <mark>interactions</mark> dataset. Now the reviews of a recipe along with the rating of the recipe in our new dataframe <mark>food</mark> with __ rows.
2. Replace all 0 values in 'rating' column with np.nan values. This is important to do because a rating of 0 doesnt mean that it was rated very lowly therefore considered "bad" recipe, but rather that no rating exists for that particular recipe.
3. Find the average rating of a recipe and add that as a column. This is important to do because the same recipe can appear more than once because there may be multiple reviews thus multiple ratings from different users for one recipe.  
4. Shown below is the starting data type all columns in our newly merged dataset <mark>food</mark>.
  - 
