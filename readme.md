# Car Price Prediction


### Problem Statement
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.

They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:

Which variables are significant in predicting the price of a car
How well those variables describe the price of a car
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.

### Business Goal
We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.



### Dataset
[https://www.kaggle.com/hellbuoy/car-price-prediction]


### Working Process

The working process is divided intoo 4 major part.

#### Data Understanding & Exploration

Try to find out the important of every colum with respect to it's given price.Visualize the cooraltion matrix and find that some columns are negetively co-related with others.


#### Data Cleaning

- find the missing values
- apply regular expression on car name
- some car names are ambiguous,so fixed them


#### Data Preparation

- divided the columns in train and test part.



#### Model Building

- Use statmodels to check which column are not important
- Finally use RFE for best columns and also check the coorelation matrix


#### Regularization

- use Ridge 
- use Lasso
