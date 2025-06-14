# Data Analysis on Features of Food Recipe
This is a project for Data Science 80 at UCSD

Author: Tori Nguyen

# Introduction

Cooking is an all rounded skill that teaches one about nourishing the body, nutritional awareness, and even functions as a stress or creative outlet. It gives us the ability to provide for ourselves and our loved ones, and only through practice and exploration of recipes and cuisines can we really learn about our preferences. Recipes will have everything laid out for us from the nutritional information down to the step by step processs; the knowledge gained from working repeatedly with ingredients and exposure to new ones supplies the user with the capability to go out and build a grocery list that caters to themselves. In this data analysis I will primarily explore the **calories of a recipe**. 

The raw dataset from food.com consists of recipes and ratings which date back to 2018. The recipes dataset has 83782 rows with each row representing a unique recipe. The 12 columns of recipes is shown below. 

|Column|Description|
|-----------|-----------|
|'name'|Recipe name|
|'id'|Recipe ID|
|'minutes'|Minutes to prepare recipe|
|'contributor_id'|User ID who submitted this recipe|
|'submitted'|Date recipe was submitted|
|'tags'|Food.com tags for recipe|
|'nutrition'|Nutrition information in the form \[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”|
|'n_steps'|Number of steps in recipe|
|'steps'|Text for recipe steps, in order|
|'description'|User-provided description|
|'ingredients'|Text of ingredients|
|'n_ingredients'|Number of ingredients in recipe|

The interactions dataset has 731927 with each row representing a review on a specified recipe. The 5 columns of interactions is shown below.

|Column|Description|
|-----------|-----------|
|'user_id'|User ID|
|'recipe_id'|Recipe ID|
|'date'|Date of interaction|
|'rating'|Rating given|
|'review'|Review text|

### First Part

This will be the data cleaning and exploration in discovering relationships between the features of the dataset. 
### Second Part 

This will be the assessment of missingness of certain columns and then proceeding to run a hypothesis test answering the question: **What is the relationship between calories and average rating of recipes?**
### Last Part 

This will be the creation of my prediction model focused on predicting the calories of a recipe. 

# Data Cleaning and Exploration 

## Cleaning 

1. Left merge recipes dataset with interactions dataset. Now for every recipe there is a rating and review for it. The new dataframe **food** has **234429** rows.
2. Replace all 0 values in 'rating' column with null values. This is important to do because a rating of 0 doesn't mean that it was rated very lowly thus considered a "bad" recipe, but rather that no rating exists for that particular recipe.
3. Find the average rating of a recipe and add that as a column. This is also important to do since the same recipe can appear more than once because there may be multiple reviews thus multiple ratings from different users for one recipe.  
4. Convert ingredients, tags, and steps columns to lists values.
  - Although it looks like a list it is actually a string object. Clean up the string formatting and convert to a list.
5. Convert submitted and date columns to datetime objects
6. The nutrition column has the same issue of representing as a list but actually being a string object. We want to strip and clean up the string formatting then convert to a list.
  - nutrition is a list of specified nutritions: calories, total fat (PDV), sugar (PDV),	sodium (PDV),	protein (PDV),	saturated fat (PDV), and carbohydrates (PDV)
  - create columns for every unique value in these lists titled 'calories', 'total fat (PDV)', etc.

Below is the first few rows of our full food dataframe 

| |name|id|minutes|contributor_id|submitted|tags|n_steps|steps|description|ingredients|n_ingredients|user_id|recipe_id|date|rating|review|average rating|calories|total fat (PDV)|sugar (PDV)|sodium (PDV)|protein (PDV)|saturated fat (PDV)|carbohydrates (PDV)|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
|0|1 brownies in the world best ever|333281|40|985201|2008-10-27|[60-minutes-or-less, time-to-make, course, mai...|10|[heat the oven to 350f and arrange the rack in...|these are the most; chocolatey, moist, rich, d...|[bittersweet chocolate, unsalted butter, eggs,...|9|3.87e+05|333281.0|2008-11-19|4.0|These were pretty good, but took forever to ba...|4.0|138.4|10.0|50.0|3.0|3.0|19.0|6.0|
|1|1 in canada chocolate chip cookies|453467|45|1848091|2011-04-11|[60-minutes-or-less, time-to-make, cuisine, pr...|12|[pre-heat oven the 350 degrees f, in a mixing ...|this is the recipe that we use at my school ca...|[white sugar, brown sugar, salt, margarine, eg...|11|4.25e+05|453467.0|2012-01-26|5.0|Originally I was gonna cut the recipe in half ...|5.0|595.1|46.0|211.0|22.0|13.0|51.0|26.0|
|2|412 broccoli casserole|306168|40|50969|2008-05-30|[60-minutes-or-less, time-to-make, course, mai...|6|[preheat oven to 350 degrees, spray a 2 quart ...|since there are already 411 recipes for brocco...|[frozen broccoli cuts, cream of chicken soup, ...|9|2.98e+04|306168.0|2008-12-31|5.0|This was one of the best broccoli casseroles t...|5.0|194.8|20.0|6.0|32.0|22.0|36.0|3.0|
|3|412 broccoli casserole|306168|40|50969|2008-05-30|[60-minutes-or-less, time-to-make, course, mai...|6|[preheat oven to 350 degrees, spray a 2 quart ...|since there are already 411 recipes for brocco...|[frozen broccoli cuts, cream of chicken soup, ...|9|1.20e+06|306168.0|2009-04-13|5.0|I made this for my son's first birthday party ...|5.0|194.8|20.0|6.0|32.0|22.0|36.0|3.0|
|4|412 broccoli casserole|306168|40|50969|2008-05-30|[60-minutes-or-less, time-to-make, course, mai...|6|[preheat oven to 350 degrees, spray a 2 quart ...|since there are already 411 recipes for brocco...|[frozen broccoli cuts, cream of chicken soup, ...|9|7.69e+05|306168.0|2013-08-02|5.0|Loved this. Be sure to completely thaw the br...|5.0|194.8|20.0|6.0|32.0|22.0|36.0|3.0|

# Univariate Analysis

Here I will examine the distribution of single variables.

The plot below shows the number of reviews per rating category. We can see a high skew specifically in favor of ratings with a value of 5. This is likely due to the criteria of who is deemed credible to post a recipe to **food.com**. There needs to be some source of skill and knowledge with cuisine, and in general people gear to foods that align with their personal palette; so, this comes into play when deciding what recipe a person will spend their time preparing. Thus, less likely for someone to rate a recipe as "bad" when there is a good amount of consideration on the users' end. 

<iframe
  src="assets/univar1-reviews-ratings.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

The next plot below shows the probability distribution of calories. However, the calories column contains large outliers; therefore, I chose to filter out 10% of the outermost outliers where calories are greater than 750. The histogram peaks around **calories = 150** which means that a randomly selected recipe has a high probability of having a calorie value around 150. We also note that the histogram is right skewed which means the probability of selecting a recipe with calories above a 400 value is lower.

<iframe
  src="assets/univar2-calories.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

# Bivariate Analysis 

Here I will examine the statistics of a pair of columns to identify possible associations. 

The scatter plot shown below identifies the relationship between number of ingredients and time in minutes. However, the minutes column has large outliers so I chose to filter out 10% of the outermost outliers where minutes are greater than 130. Notice there is no strong, visible correlation (linear trend) between the number of ingredients and minutes of a recipe. There is also similar vertical stretch all throughout suggesting that there isn't more ingredients for recipes that take longer. Lastly, there is a dense cloud of data points in the lower half of the x-axis in which we can interpret as there being more recipes that take less time. 

<iframe
  src="assets/bivar-ingredients-minutes.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

# Interesting Aggregates 

I will explore the average amount of sugar per rating category. I binned the continuous numerical sugar column into discrete categorical intervals based on 4 bins: 1st quartile, 2nd quartile, 3rd quartile, and 4th quartile of the sugar data. Then I grouped with the two ordinal categorical columns, sugar quartile and rating, and found the average. 

|sugar quartile|1st quartile|2nd quartile|3rd quartile|4th quartile|
| --- | --- | --- | --- | --- |
|rating|  |  |  |  |
|1.0|61.09|864.83|1310.67|NaN|
|2.0|52.90|883.24|1461.00|NaN|
|3.0|48.27|814.55|NaN|NaN|
|4.0|43.67|806.66|1450.59|2152.0|
|5.0|47.29|806.18|1386.44|2156.5|

- Note: wherever there are null values means that there is no data for a given rating that falls under the specific sugar quartile

# Assessment of Missingness

## NMAR Analysis

By exploring the data I found that there are 4 columns with null values: description, rating, review, and average rating. NMAR missingness is dependent on the value itself. Refer to the introduction where I performed the data cleaning. I created the null values that exist in the rating column, and the average rating column was created using the rating column so if one contains null values then it makes sense that the other would too. 

The missingness of rating is NMAR because I created the null values wherever rating was 0. This is because a rating of 0 meant a rating does not exist for the given recipe, not that the recipe was lowly rated. Leaving the existence of 0 values would affect the data analysis. 

## Missingness Dependency

I will explore the missingness dependency of reviews against other columns using hypothesis testing with a significance level of **0.05**. 

### Review and Sodium

**Null Hypothesis:** Distribution of sodium with missing review values is the same as without missing review values 

**Alternative Hypothesis:** Distribution of sodium with missing review values is different from without missing review values 

**Test Statistic:** the plot of the distribution of sodium when reviews are missing and when they are not show that the two distributions are different shape but similar centers which led me to use the **KS test stat**.  

<iframe
  src="assets/MAR-sodium.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

I performed a permutation test and got a p-value of 0.17 which led me to keep the null hypothesis; therefore, the missingness of review is not dependent on sodium amount.

### Review and Number of Steps 

**Null Hypothesis:** Distribution of number of steps with missing review values is the same as without missing review values 

**Alternative Hypothesis:** Distribution of number of steps with missing review values is different from without missing review values 

**Test Statistic:** the plot of the distribution of number of steps when reviews are missing and when they are not show that the two distributions are different shape but similar centers which led me to use the **KS test stat**.

<iframe
  src="assets/MAR-nsteps.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

I performed a permutation test and got a p-value of 0.00007 which led me to reject the null hypothesis in favor of the alternative hypothesis; therefore, the missingness of review is dependent on the number of steps of a recipe.

# Hypothesis Testing

My main interest in exploring the food dataset is investigating the nutritional aspect of a recipe. The question I am focused on answering is **"What is the relationship between calories and average rating of a recipe?"**. Both calories and average rating are numerical continuous data. 

I decided to group the calories to create a categorical column based on where calories are greater than and less than the median of calories (298.2). I chose to go with the median because the calories column has large outliers that heavily skews the data to the right, in which the median would be good to use as it is robust to outliers. 

**Null Hypothesis:** The average rating for recipes with below median calories is the same as that for recipes with above median calories, therefore any observed differences is due to randomn chance 

**Alternative Hypothesis:** The average rating for recipes with below median calories is different from that for recipes with above median calories

**Test Statistic:** Absolute value of difference in mean between 'below median' calories and 'above median' calories
- I chose to use the absolute value because I care about whether these two distributions are different; therefore I care about the magnitude in difference not the direction 

**Significance Level:** 0.05

I chose a permutation test because I only have access to a sample of recipes from food.com and we want to compare if two distributions look similar or different (from same population or not). I performed 1000 repititions and shuffled the newly created categorical data column from grouping calories.

<iframe
  src="assets/permtest.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

The observed test statistic is represented by the bold vertical line. I got a p-value of **0.676** which led me to keep the null hypothesis which means any observed differences is due to randomness. 

# Problem Identification

I plan to **predict the calories of a recipe** which is a regression problem. The response variable I chose is the calories of a recipe because for many people an important factor in creating meals is the amount of calories they are building in their meals; being able to identify the calories of a meal is a distinction that is of interest for some; whether it's to be more nutritionally aware or to watch their weight. 

The metric I am using to evaluate my model is the **Root Mean Squared Error** because calories is a continuous numerical value. The information I know and is available to use before I train the model are all the columns I named above under **Intoduction**. I filtered the original **food** dataframe to remove the outermost 1% of outliers where calories >= 2500. I determined only a 1% removal because that will handle the situation of extreme cases and I won't lose too much potentially important data for my predictions. 

# Baseline Model 

For my baseline model I am untilizing a Linear Regression model and splitting the dataset into testing and training sets. The features in the baseline model are total fats, carbohydrates, and proteins in percentage of daily value \(PDV). All 3 features are quantitative however I chose to standardize these features because carbohydrates averaged about 12 while proteins and total fats averaged about 30. 

The RMSE of the training set is **29.35** which is a good model because I recognize that RMSE is in the units of our original y data which is calories. So, our RMSE is telling me that the predicted values are about 29.35 calories off from the actual calorie values, which is not a lot in terms of calories. 

# Final Model 

The final model uses the standardized features total fats, carbohydrates, protein, and sugar as well as the binarized feature n_ingredients. I found these as my best hyperparameters from performing an iterative **crossed validation score** with 5 folds. I incremented my features as follows: 
1. stdscalar total fats only
2. stdscalar total fats + carbs
3. stdscalar total fats + carbs + protein
4. stdscalar total fats + carbs + proteins + sugars
5. **stdscalar total fats + carbs + proteins + sugars + binarized n_ingredients**
6. stdscalar total fats + carbs + proteins + sugars + binarized n_ingredients + stdscalar minutes

I chose the obvious features of fats, carbohydrates, proteins, and sugars because calories are essentially the measure of the amount of these nutrients in food. Then I considered the number of ingredients as the next most important when predicting the calories of a recipe since more added material to a recipe will increase the calories. However, it wasn't one of my highest options because if the number of ingredients is high from something like onions, garlic, seasonings, etc then those will not contribute to a higher calorie count. The last feature I considered is how long a recipe takes to make (minutes column). Generally, foods that are more nutritionally dense like chicken take longer to cook or desserts that have high amounts of sugar and are baked also take a long time to cook. On the flip side foods like steak are nutritionally dense but are done cooking fast, so minutes may not be an important feature to add in our model.

I continued to use the RMSE metric to evaluate my final model. The RMSE of my final model is **29.32** which is 0.03 better than the baseline model.

# Fairness Analysis

For my fairness analysis I chose my two groups as above median minutes and below median minutes. I chose the median instead of mean because the distribution of minutes has many large outliers in which the mean is not robust to. Since my model is a regression model I chose the RMSE as my evaluation metric. 

**Null Hypothesis:** My model is fair. The RMSE for above median minutes group and below median minutes group are roughly the same and any differences are due to randomness

**Alternative Hypothesis:** My model is unfair. The RMSE for below median minutes group is lower than the RMSE for above median minutes group. 

**Test Statistic:** The difference in RMSE of (below median minutes group - above median minutes group)

**Significance Level:** 0.05

I performed a permutation test with 1000 repititions shuffling the labels of my new categorical data created from grouping minutes.

<iframe
  src="assets/fairness_distr.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>

I got a p-value of 0.0 which led me to reject the null hypothesis in favor of the alternative hypothesis. Therefore, I cannot say that the model is fair, and differences are not due to randomness; although, I cannot precisely define why. 
