# Starbucks Capstone Project
## Introduction
For the capstone project, I selected the Starbucks Challenge. The dataset provided contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

![Starbucks image](https://c.pxhere.com/photos/a0/9f/starbucks_coffee_abstract_logo-1058040.jpg!d)

## Goal of Analysis
The end goal of the analysis was to determine which features have the most impact on determining whether a user completes and offer or not. We'll take this analysis one step further and see if there are variations between different offer types.

At the end of our overview, we'll have a strong overview of which features are most important in determining a successul offer completion.

We'll use a random forest classifier to determine which features matter most.

![random forest](https://github.com/datagy/Udacity-Data-Science-Program/blob/master/Capstone/images/tree.png)

## Data Sets Overview
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

### Data Cleaning
The data were of differing degrees of data cleanliness. For example, the `portfolio` dataset required very little cleaning. Others required more intensive cleaning, where data was structured as nested dictionaries within the resulting dataframes. Functions were developed to properly clean all data, one-hot encoding data as necessary.

### Dealing with Missing Values
One important consideration with any machine learning project is determining what to do with values. There are a number of typically considered options:

1. Drop missing values row-wise,
2. Drop missing values column-wise,
3. Impute missing values.

In our case, we encountered missing values in our `Profile` dataset. The values existed only in the `age` and `income` columns. We don't know enough to impute the values (and this analysis was outside the scope of the project). Because of this, it's best to drop those records. We'll drop the records, rather than the columns. If we had determined that these features had significant influence on our models, then we could return and impute the values.

### Determining Offer Success
In order to determine whether or not an offer was successul, some manipulation needed to be done to the cleaned `transcript` dataset. In particular, the decision was made to classify and offer as successful if and only if the user viewed the offer and completed the offer.

The following logic was applied to the `cleaned_transcript` dataframe:

```python
transcript_success = pd.pivot_table(data=cleaned_transcript, index=['person', 'offer_id'], columns=['event'], aggfunc='count').reset_index()
transcript_success.columns = ['person', 'offer_id', 'offer completed', 'offer received', 'offer viewed']
transcript_success['offer completed'] = transcript_success['offer completed'].apply(lambda x: 1 if x >= 1 else 0)
transcript_success['offer viewed'] = transcript_success['offer viewed'].apply(lambda x: 1 if x >= 1 else 0)
transcript_success['success'] = transcript_success['offer completed'] + transcript_success['offer viewed']
transcript_success['success'] = transcript_success['success'].apply(lambda x: 1 if x == 2 else 0)
transcript_success = transcript_success.drop(columns = ['offer completed', 'offer viewed', 'offer received'])
```

This gave us a unique `success` indicator for each person-offer pair. A 0 indicated that the offer was not successful, while a 1 indicated that the offer was successful.

### Final Pre-Processing
As a final step, we combined all relevant features of the three datasets into a single dataset.

The resulting dataframe allowed us to view per person-offer combination the following attributes: `'success', 'web', 'mobile', 'email', 'social', 'age', 'income', 'member_age', 'F', 'M', 'O', 'offer_type'`.

`offer_type` is included in the list only to be able to split this into various offer_type dataframes. Each offer type likely has differing drivers for what determines whether an offer is successful. Even if this doesn't turn out to be the case, it's incredibly useful information.

The dataframes can be pre-processed using the `get_dataframe()` function, which simply returns a single dataframe attributed to that offer type:

```python
def get_dataframe(df, offer):
    """Generates dataframes to run model against.
    INPUT: 
    df = dataframe of all data
    offer = offer type to filter to
    
    OUTPUT: 
    filtered = dataframe filtered to specific offer type"""
    if offer in ['bogo', 'discount', 'informational']:
        return df[df['offer_type'] == f'{offer}'].copy().drop(columns=['offer_type'])
    else:
        print("Incorrect offer_type inputted")
```

## Model Building, Fine-Tuning, and Evaluation
Sci-kit learn was used to develop the model, fine-tune it, and evaluate its attributes. Each offer_type dataframe is split into some key elements:
* **X, X_train, X_test**: train and test splits of targets
* **y, y_train, y_test**: train and test features
* **category_names**: the labels for the features

All features were scaled using `sklearn.preprocessing`'s StandardScaler function. This was especially important in normalizing the income feature, which had a large spread.

A function was developed to tune hyperparameters of different offer_type dataframes. sklearn's GridSearchCV was used to determine what the best parameters were. These parameters were returned by the function and fed into a function to return a refined model based on these parameters (as they can vary depending on the offer_type used).

```python
def build_evaluate_model(X, y):
    """Evaluates the model and returns a cv and returns a cv.
    
    OUTPUT = dictionary of parameters"""

    parameters={'max_features': ['auto', 'sqrt'],
                'max_depth' : [5, 10, 15],
                'n_estimators': [10, 30]}

    grid_search = GridSearchCV(RandomForestClassifier(random_state=2), param_grid=parameters)
    grid_search.fit(X, y)
    parameters = grid_search.best_params_
    
    return parameters
    
def refined_model(df, offer_type):
    """Returns a refined model based on other functions.
    INPUT:
    df = dataframe to use
    offer_type = uses 'bogo', 'informational', 'discount'
    
    OUTPUT:
    model = a fitted model
    """
    
    X, y, X_train,X_test,y_train, y_test, category_names = load_data(df, offer_type, test_size=0.20, random_state = 1)
    parameters = build_evaluate_model(X_train, y_train)
    model = RandomForestClassifier(random_state=1,
                                   max_depth=parameters.get('max_depth'), 
                                   max_features= parameters.get('max_features'),
                                   n_estimators=parameters.get('n_estimators'))
    model.fit(X_train, y_train)
    
    return model
```

From here, we were able to use the `features_importances_` attribute to be able to parse out for each offer_type model, which features were most influential.

These attributes were plotted against one another (as each feature was available across offer types).

## Analyzing Results
Let's take a look at the importance of different features across the `BOGO` and `Discount` offers. They have been mapped out as below:

![Feature importances for offer types](features_importances.png)

We can see that there is fairly significant variation between feature importance across the two different offer_types. Some features (such as the social medium) have significant impact on discount offer types, but much less on BOGOs. 

A key thing to note here is that `member_age` (i.e., how long a use has been a member), influences the success of an offer fairly dramatically for either offer type.

## What Does This Tell Us?
This is where things get interesting! We can see that the two features that have consistently high influence are member_age and income. Many others, such as gender and web ads have little impact. This means that we don't need to know a lot about a user directly to determine the influence of an offer campaign. Given that many users will be hesitant to share items such as age, income, and gender, we can determine which of these features would be worth investing into further (i.e., offer incentives to collect the information) or to use proxies to determine some of this information. 

For example, income has a fairly high importance, but many users may be uncomfortable sharing this information. We could suggest using location data to develop proxies for income based on median incomes of different neighbours from which users place mobile orders.

## Next Steps
In terms of next steps: since we now have a sense of what attributes are most important, we can remove some of the others that matter less. This allows us to prevent the model from being overfitted. We can then apply further models to a pipeline to see if this can be further refined.
