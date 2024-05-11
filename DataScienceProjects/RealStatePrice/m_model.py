import json
import pickle
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
matplotlib.rcParams['figure.figsize'] = (2, 1)

df1 = pd.read_csv('house_prices.csv')

# area_type   availability                  location       size  society total_sqft  bath  balcony   price
print(df1.groupby('area_type').agg('count'))
# data Cleaning
# for example lets assume that we don't need availability ,location,society, balcony not needed for pricing
df2 = df1[['area_type', 'location', 'size', 'bath', 'price', 'total_sqft']]
print(df2.isnull().sum())
# If we can update these na values manually then good or
# otherwise we have to update it with zero if this zero has no effect on price
# if it has effect on price better update it with mean value
# but for Text value can do it so better drop those nas
df3 = df2.dropna()
print(df3['size'].unique())
# 4 BHK and 4 Bedroom are the same value
# So we can take the first numberic value as the bed number

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.head(4))
df3.drop('size', axis='columns', inplace=True)
print(df3.head(4))

# check any abnormal data
print(df3['bhk'].unique())
# we have  19 16 43 bedrooms . so lets see whcich house and how much the price or area
print(df3[df3.bhk > 15])

# ite seems Munnekollal  40bath  660.0 price       2400sqft has 43beds and

# 1Hanuman Nagar  16bath  490.0 price       2000sqft   19 beds which seems is not correct '''Just common sense

# drop those rows by index
df3.drop([4684, 11559, 3379], inplace=True)

# lets check the total_sqft values
print(df3['total_sqft'].unique())
# here we can see that '1133 - 1384',1000Sq. Meter 1100Sq. Yards etc  data which is not formatted
# lets see which values are not float


def is_float(x):
    try:
        float(x)
    except:
        return False

    return True


def getDigit(x):
    multyplier = 1
    if (is_float(x)):
        return float(x)
    else:
        sqmeter_to_foot = 10.7639
        sqyard_to_foot = 9
        parch_to_foot = 272.25

        if "Meter" in x:
            multyplier = sqmeter_to_foot
        elif "Yards" in x:
            multyplier = sqyard_to_foot
        elif "Perch" in x:
            multyplier = parch_to_foot
        elif "-" in x:
            return (float(x.split('-')[0])+float(x.split('-')[1]))/2
        else:
            multyplier = 1

        d = [*x]
        n = ''
        for i in d:
            if (i.isnumeric()):
                n = n+i
            else:
                break

        return float(format(int(n)*multyplier, '.2f'))


# get the list of non float of totalsq
# ~ is use fro negate the posision
print(df3[~df3['total_sqft'].apply(is_float)].head(14))

df3['sqft'] = df3['total_sqft'].apply(lambda x: getDigit(x))
#    int(x.split('-')[0])+int(x.split('-')[1]))/2)

print(df3[~df3['total_sqft'].apply(is_float)].head(14))


# check any null
print(df3.isnull().sum())
# check sq feet
print(df3[df3.sqft == 0])
# as the price in lac value wich is 100000..  minimize it to per sq feet

df4 = df3.copy()
df4.drop('total_sqft', axis='columns', inplace=True)
df4['per_sqft'] = df3['price']*100000/df3['sqft']
print(df4.head(4))

# now check the location data

print(len(df4.location.unique()))
# 1302 location which is too much to handle as parameter input

# So we need to minimize


# fist clean the location data with any blank space
df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby(
    'location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

# lets see how many location has less then 10 data cause these data won't help us to predict location wise pricing
# so that we can group it as one location called 'Other Location'
location_10 = location_stats[location_stats <= 10]
print(len(location_10),
      ' of ', len(df4.location.unique()))
# 1050

# now we update the these location to Other Location
df4.location = df4.location.apply(lambda x: 'Other' if x in location_10 else x)

# now lets see unique location
print(len(df4.location.unique()))
print(df4.head(10))

# out liars like abnormal bedroom we already dropped
# as analyst find other outliars
# for example bedroom with area ie 1000 sqft with 6 bed room is abnormal
# lets assume per bedroom is minimum 100sqft and with kicthcen and other room it will take minimum 300sqft
# lets see which properties has below this area
print(df4[df4.sqft/df4.bhk < 300])
# lets remove or drop this rows
df5 = df4[~(df4.sqft/df4.bhk < 300)]

print(df4.shape, df5.shape)

# lets check abonormal persqft price
print(df5.per_sqft.describe())

# min          23.596850
# max      176470.588235
# above values  are abnormal which std        4167.157514
# we have to remove this daviation of std and others in per location


def remove_outliars(df):
    df_out = pd.DataFrame()
    all_location = df.groupby('location')

    for key, subdf in all_location:
        mn = np.mean(subdf.per_sqft)
        st = np.std(subdf.per_sqft)
        dv = mn-st
        dvp = mn+st
        df_reduce = subdf[(subdf.per_sqft > dv) & (subdf.per_sqft < dvp)]
        df_out = pd.concat([df_out, df_reduce], ignore_index=True)

    return df_out


df6 = remove_outliars(df5)

print(df6.per_sqft.describe())
# reduced  12486 to 10265

# now lets see bedroom  wise persqfeet  plotting on locations


def plot_bedrooms(df, locatoin):
    bed2 = df[(df.location == locatoin) & (df.bhk == 2)]
    bed3 = df[(df.location == locatoin) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (10, 8)
    plt.scatter(bed2.sqft, bed2.per_sqft, color='blue',
                marker='o', label='2 Beds')
    plt.scatter(bed3.sqft, bed3.per_sqft, color='green',
                marker='*', label='3 Beds')
    plt.xlabel('Total Area (sqrft)')
    plt.ylabel('Price per SQFT')
    plt.legend()
    plt.title(locatoin)
    plt.show()


plot_bedrooms(df6, 'Rajaji Nagar')
# here we can see that some 2 bed are higher then 3 bed for the same sqft
# so we have to find those rows which price is  lower then bed-1  price location wise


def remove_outliars_bhk(df):
    df_out = np.array([])
    all_location = df.groupby('location')

    for location, location_df in all_location:
        bhk_states = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_states[bhk] = {
                'mean': np.mean(bhk_df.per_sqft),
                'std': np.std(bhk_df.per_sqft),
                'count': bhk_df.shape[0]
            }
        # print('location ', location, ' states', bhk_states)

        for bhk, bhk_df in location_df.groupby('bhk'):
            f = bhk - 1
            if f in bhk_states:
                stats = bhk_states[f]
                if stats and stats['count'] > 5:
                    df_out = np.append(
                        df_out, bhk_df[bhk_df.per_sqft < (stats['mean'])].index.values)

    return df.drop(df_out, axis='index')


print(df6.per_sqft.describe())
df7 = remove_outliars_bhk(df6)
print(df7.shape)
# now lets see the improvement on plot

plot_bedrooms(df7, 'Rajaji Nagar')

# now lets see how many appartment are there according per_sqft field (group by range count)
# and plot it as histogram


plt.hist(df7.per_sqft, rwidth=0.8)
plt.xlabel('Per Sqr feet')
plt.ylabel('Apparments')
plt.legend()
plt.show()

# major data points are between 0-10000 per sqft

# now lets check bathroom
print(df7.bath.unique())

print(df7[df7.bath > 10])
# again lets plot it in hisogram
plt.hist(df7.bath, rwidth=0.8)
plt.xlabel('Bathroom')
plt.ylabel('Apparments')
plt.legend()
plt.show()
# most of the appartment has 2-6 bathroom
# we can see that there are 12, 13,16 bath .. lets see which appartment
# its seems ok... but if business manger says that bathroom can't exceed bedroom +2 then you have to remove those data
print(df7[df7.bath > df7.bhk+2])
#                 area_type       location  bath   price  bhk     sqft     per_sqft
# 1631        Built-up  Area  Chikkabanavar   7.0    80.0    4   2460.0  3252.032520
# 5251        Built-up  Area     Nagasandra   8.0   450.0    4   7000.0  6428.571429
# 5863  Super built-up  Area          Other   9.0  1000.0    6  11338.0  8819.897689
# 9033  Super built-up  Area    Thanisandra   6.0   116.0    3   1806.0  6423.034330

df8 = df7[df7.bath <= df7.bhk+2]

# now we have clean dataset now next step is Machine Learning

# lets drop unwanted varaible wich we don't need for predction perpose
# our prediction is price and input is location,sqft, bhk, bath
df9 = df8.drop(['area_type', 'per_sqft'], axis='columns')
print(df9.head(5))

# now make the location encoded with dummy
# we can drop actual location field and
# and also 'Other' dummy location field because all 0 will automatically set to 'Other' location
dummy = pd.get_dummies(df9.location)
df10 = pd.concat([df9.drop('location', axis='columns'),
                 dummy.drop('Other', axis='columns')], axis='columns')
print(df10.head(3))

# now training testing
X = df10.drop('price', axis='columns')
Y = df10.price

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=.2, random_state=10)

lrm = LinearRegression()
lrm.fit(x_train, y_train)
print(lrm.score(x_test, y_test))
# 0.783542852402141

# sufflinhg and using KFold check if any better model
cv = ShuffleSplit(n_splits=10, test_size=.2, random_state=0)
scores = cross_val_score(LinearRegression(), X, Y, cv=cv)
print(scores)

# lets try with lass and decission tree


def best_model(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {

            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['poisson',  'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],
                          cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


print(X.head(4))
df_b = best_model(X, Y)

print(df_b)
# so best model is inear_regression    70%
# we already try it with before comparing line: 292
print(lrm.score(x_test, y_test))
# now our model is final but location paramater is numeric . so we need to paramatarise it as text
# So we make a fuction which will predict the price  by taking location as text paramater


def predict_price(location, sqft, bath, bed):
    loc_column_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = bed
    x[2] = sqft
    if loc_column_index >= 0:
        x[loc_column_index] = 1

    return lrm.predict([x])[0]


# Lingadheeranahalli,3 bed,1521 sqft,3 bath price 95 lac
print(predict_price('Lingadheeranahalli', 1521, 3, 3))
# 98 lac
print(predict_price('Indira Nagar', 1521, 3, 3))
# 205 lac

# So it calc prime location pricing also

# now the model is ready to export and use it for production (http or other application)
# using  import pikkle

with open('appartment_priceing.pickle', 'wb') as f:
    pickle.dump(lrm, f)

# exporting column names
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
out_file = open("columns.json", "w")

json.dump(columns, out_file, indent=6)

out_file.close()
