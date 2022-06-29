import math
import os
from datetime import datetime
from pathlib import Path
from statistics import LinearRegression
from turtle import pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import urllib3
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn import preprocessing, neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from webdriver_manager.chrome import ChromeDriverManager
from mpl_toolkits.basemap import Basemap

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

def get_page_soup(url):
    """
    Returns the web page as an html object.
    """
    try:
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(url, verify=False, headers=agent)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup
    except Exception as e:
        print("[get_page_soup] Could not get page {}: \n {}".format(url, e))
    return None

def get_page_attributes_sel(url, feature_body):
    """
    This function is using Selenium in order to 'Click' the 'More properties'
    button and this allows us to extract all of the restaurant properties.
    """
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    button = None
    try:
        button = driver.find_element_by_css_selector('#site-main > div > div.place_and_rush_hours > div > div > div > div > small')
    except Exception as e:
        pass
    if(button):
        button.click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        attributes_body = soup.find('div', {'class':'pop-scroll-wrap'})
        if attributes_body:
            attributes_list = attributes_body.find_all('li')
            attributes = [attr.text.replace("\n","").strip() for attr in attributes_list]
            driver.quit()
            return attributes
        return []
    else:
        driver.quit()
        return get_page_attributes(feature_body)

def get_page_attributes(body):
    """
    Extract the restaurant parameters.
    """
    attributes_body = body.find('div', {'class':'place_info'})
    if(attributes_body):
        attributes_list = attributes_body.find_all('li')
        attributes = [attr.text for attr in attributes_list]
        return attributes
    return []

def get_type(feature_page):
    """
    gets the type of restaurant: Italian/
    :param body:
    :return:
    """
    try:
        name = feature_page.find("h6")  # , attrs={"class": "main_banner_content"}
        return name.text.split(',')[0].strip()  # take the name "name, location" and leave only the name
    except Exception as e:
        print("[get_type] error: ", e)

def get_number_of_reviews(body):
    """
    Extracts the number of the restaurant reviews.
    :param body:
    :return:
    """
    try:
        reviews_body = body.find('div', {'class':'raviews_box_item'})
        if(reviews_body):
            reviews_link = reviews_body.find('a')
            if(reviews_link):
                num = reviews_link.text.split(' ')[0]
                if(num.isdecimal()):
                    return num
    except Exception as e:
        print("[get_number_of_reviews] could not get number of reviews ", e)
    return 0

def get_name(feature_page):
    """
    :param feature_page:
    :return: name of the restaurant
    """
    try:
        name = feature_page.find("h1")  # , attrs={"class": "main_banner_content"}
        return name.text.split(',')[0].strip()  # take the name "name, location" and leave only the name
    except Exception as e:
        print("[get_name] error: ", e)

def get_stars(feature_page):
    """
    Extract the restaurant rating.
    :param feature_page:
    :return:number of stars
    """
    if(feature_page.find("div", attrs={"class":"reviews_wrap"})):
        stars = feature_page.find("div", attrs={"class":"reviews_wrap"}).find("span")  # , attrs={"class": "main_banner_content"}
    else:
        stars = None
    if(stars):
        return stars.text
    else:
        return None

def get_geolocation(feature_page):
    """
    Extract the location of the restaurant
    :param feature_page:
    :return: geolocation of rest
    """
    address = feature_page.find("h5")
    geolocator = Nominatim(user_agent="catuserbot")
    location = geolocator.geocode(address.text)
    if(location):
        return location.latitude, location.longitude
    else:
        return None

def save_df_to_csv(df):
    """
    creates csv based on the df and save it with a unique timestamp.
    """
    folder_name = "Resturants Output"
    if(not os.path.exists(folder_name)):
        os.mkdir(folder_name)
    file_name = "Rest df {}.csv".format(datetime.now().strftime("%d.%b.%Y %H-%M-%S"))
    filepath = Path("{}/{}".format(folder_name, file_name))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding = 'utf-8-sig', index=False)
    print("Your df is saved !")

def extract_page_attributes(page):
    """
    This function extract the restaurant features/ parameters using dedicated function for each parameter.
    Then it creates a dictionary that holds the restaurant data and add it to the data_local list.
    The function set the value of '1' for each parameter of the restaurant.
    :param page:
    :return: list of JSON of restaurants
    """
    feature_column = page.find_all("div", attrs={"class":"feature-column"})
    print("feature num ", len(feature_column))
    data_local = []  # list of restaurants on specific page
    for col in feature_column:  # runs on every restaurant page
        try:
            pageid = col.attrs["data-customer"]
            print("page id "+pageid)
            url = "https://www.rest.co.il/rest/" + pageid
            feature_body = get_page_soup(url)
            name = get_name(feature_body)
            type = get_type(feature_body)
            page_attributes = get_page_attributes_sel(url, feature_body)
            num_of_reviews = get_number_of_reviews(feature_body)
            stars = get_stars(feature_body)
            geolocation = get_geolocation(feature_body)
            resturant = {
                'id'             : pageid,
                'name'           : name,
                'type'           : type,
                'stars'          : stars,
                'location'       : geolocation,
                'num_of_reviews' :  num_of_reviews
            }
            for att in page_attributes:
                resturant[att] = '1'    
            print(resturant['name'])
            data_local.append(resturant)
        except Exception as e:
            print("[extract_page_attributes] error: ", e)
    return data_local

def get_data_for_pages(num):
    """
    This function gets number of pages to extract from the site.
    It goes over the site pages for num of times and using the  extract_page_attributes function
    to extract the page attributes indo the data variable
    """
    page = get_page_soup("https://www.rest.co.il/restaurants/israel")
    data.extend(extract_page_attributes(page))  # adding the page lst of JSON to global data DF
    if(num == 1):
        return data
    for i in range(2,num + 1):
        print("page ", i)
        page = get_page_soup("https://www.rest.co.il/restaurants/israel/page-{}/".format(i))
        if(page is None):
            break
        data.extend(extract_page_attributes(page))
    return data

def load_csv(file_name):
    """
    For testing. loading our saved df from CSV file.
    """
    return pd.read_csv(file_name, header=0, sep=',')

def fill_empty_binary_values(df):
    """
    Our DataFrame is built in a way that each restaurant is adding its on features/ parameters.
    This means we need to add the value '0' to all columns of the other restaurants in order to avoid missing cells.
    """
    df.fillna(value=0, inplace=True)

def heat_map(df):
    corr = df.corr()  # heat map
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  #
    plt.show()

def update_score(df):
    """
    This function calculates the restaurants score according to
    the T-Distribution formula and add it to the DataFrame.
    We also added a normalized column for convenience
    """
    df['score'] = df.apply(lambda row: row.stars - (1.96 * (1 / math.sqrt(row.num_of_reviews))), axis=1)  # update score column
    df['score_normalized'] = df.apply(lambda row: (row.score - min(df.score)) / (max(df.score) - min(df.score)), axis=1)  # normalized = (x-min(x))/(max(x)-min(x))

def split_loc(df):
    """
    splits the location into lat and lon columns and updates the df
    :param df:
    :return:
    """
    lat = []
    lon = []

    location = df['location'].tolist()
    for i in range(len(location)):
        if location[i] == '0' or location[i] == 0:
            lat.append(0)
            lon.append(0)
        else:
            temp = location[i].replace(" ", "").replace("(", "").replace(")", "").split(",")  # .trim()
            lat.append(temp[0][:4])
            lon.append(temp[1][:4])

    df['lat'] = lat
    df['lon'] = lon
    return df

def type_to_int(df):
    """
    turns the categorial type to numeric
    :param df:
    :return:
    """

    le = preprocessing.LabelEncoder()
    list = df['type']
    list = le.fit_transform(list)
    df.insert(loc=3, column='type_numeric', value=list)

    # le = preprocessing.LabelEncoder()
    # list = df['type']
    # list = le.fit_transform(list)
    # df.insert(loc=3, column='type_numeric', value=list)


    # df.type = pd.Categorical(df.type)
    # df['type_numeric'] = df.type.cat.codes

    # lbl = LabelEncoder()
    # df['type_numeric'] = lbl.fit_transform(df['type'])


    return df

def heat_map(df):
    """
    Create and show heat map of our DataFrame parameters.
    """
    corr = df.corr()  # heat map
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  #
    plt.show()

def geo_map(df):
    """
    Creates a map with visualization of the scores of each restaurant.
    """
    lat = df['lat'].values
    lon = df['lon'].values
    score = df['score_normalized'].values

    m = Basemap(projection='lcc', resolution='h',
                width=0.5E6, height=0.5E6,
                lat_0=31.6, lon_0=34.88, )

    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')

    m.scatter(lon, lat, latlon=True, c = score,s = 15, cmap='Reds', alpha=0.3)

    plt.colorbar(label='score')
    plt.clim(0, 1)

    x, y = m(32, 34)
    plt.plot(x, y, 'ok', markersize=2)
    plt.text(x, y, ' scores', fontsize=12)
    plt.show()

def show_histograms(df):
    """
        This function is calculating the 10 most common features and shows histograms for
        each one showing the resturant type score histogram
    """
    type_col = df['type_numeric'].to_list()
    values, counts = np.unique(type_col, return_counts=True)
    ind = np.argpartition(-counts, kth=10)[:10]

    for type in ind:  # show best 10 type graphs
        df_type = df.loc[df['type_numeric'] == type]
        name = (df_type.iloc[0]['type'])[::-1]
        x = df_type.index
        y = df_type['score_normalized'].to_list()

        ax = sns.displot(y, kde=True)
        ax.set(xlabel='score', ylabel='number of restaurants')
        ax.fig.suptitle(name)
        plt.ylim([0, 70])
        plt.show()



def show_boxplot(df):
    """
    Creates boxplot
    """
    sns.set_theme(style="white", palette="pastel")
    ax = sns.boxplot(x=df["score"])
    plt.show()


def get_highly_correlated_cols(df):
    """
    Returns correlations array and tupples array showing tuples of highly correlated columns/ parameters
    """
    list = df.corr().abs().unstack()
    correlations = []
    tuple_arr = []

    for index, value in list.items():
        if (index[0] == index[1]):
            continue
        col1 = df.columns.get_loc(index[0])
        col2 = df.columns.get_loc(index[1])
        if (value >= 0.5 and (col2, col1) not in tuple_arr):
            print("{} - {}".format(index[0], index[1]))
            correlations.append(value)
            tuple_arr.append((col1, col2))

    return correlations, tuple_arr


def lin_regression(df):
    """
    Running the linear regression model
    """
    score_normalized = df["score_normalized"]  # for pca
    #data = df.drop(columns=['score_normalized'])  # for pca
    data = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'score_normalized','type_numeric', 'num_of_reviews']).copy()  # num_of_reviews

    x_train, x_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.1)
    clf = LinearRegression()
    clf.fit(x_train, y_train)

    print("R2 linear regression: ", clf.score(data, score_normalized))  # r2 The coefficient of determination

def knn_regression(df):
    """
    Running the KNN regression model
    """
    score_normalized = df["score_normalized"]

    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type_numeric',
                            'num_of_reviews']).copy()  # num_of_reviews

    # data = df.drop(columns=['score_normalized']).copy()  # num_of_reviews

    x_train, x_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.1, random_state=42)

    # knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    knn = neighbors.KNeighborsRegressor(n_neighbors=5)

    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)

    print("R2 knn: ", model.score(x_test, y_test))

def network(df):
    """
    Running the neural network model
    """
    score_normalized = df["score_normalized"]
    # data = df.drop(columns=['score_normalized'])
    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type_numeric',
                            'num_of_reviews']).copy()  # num_of_reviews
    # score_normalized = df["score_normalized"]
    # data = df.drop(columns=['score_normalized'])

    X_train, X_test, y_train, y_test = train_test_split(data, score_normalized, random_state=1)
    # regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    # regr = MLPRegressor(hidden_layer_sizes = (25, 10, 5),max_iter = 200, activation = 'relu',solver = 'adam').fit(X_train, y_train)
    regr = MLPRegressor(random_state=42, max_iter=250).fit(X_train, y_train)
    # regr.predict(X_test)

    print("R2 network: ", regr.score(X_test, y_test))


def knn_graph(df):
    """
    Generating KNN graph in order to show different scores for different amount of neighbors
    """
    # df = pd.get_dummies(df)
    score_normalized = df["score_normalized"]

    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type_numeric',
                            'num_of_reviews']).copy()  # num_of_reviews
    #
    # data = df.drop(columns=['score_normalized']).copy()  # num_of_reviews

    x_train, x_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.1, random_state=42)

    # knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    params = {3, 7, 9, 13}
    xx = []
    yy = []

    for i in params:
        knn = neighbors.KNeighborsRegressor(n_neighbors=i)

        # model = GridSearchCV(knn, params, cv=5)
        knn.fit(x_train, y_train)
        # print(model.best_params_)
        # predicted = model.predict(x_test)
        # Score
        # print("R2 knn: ", metrics.mean_absolute_error(predicted, y_test))
        sc = knn.score(x_test, y_test)
        print("R2 knn ({} neighbors): {}".format(i, sc))
        xx.append(i)
        yy.append(sc)

    d = pd.DataFrame(columns={'k neigbors', 'score'})
    d['k neigbors'] = xx
    d['score'] = yy
    sns.barplot(data=d, x="k neigbors", y="score", palette='Blues_d')
    plt.show()


def dim_reduce_PCA(df, n):
    """
    Using the PCA technique in order to reduce dimension
    """
    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'num_of_reviews']).copy()
    features = data.columns
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['score_normalized']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents)

    # print(principalDf)
    finalDf = pd.concat([principalDf, df[['score_normalized']]], axis=1)
    finalDf = pd.concat([finalDf, df[['lat', 'lon', 'type_numeric']]], axis=1)
    return finalDf

# Crawling

data = get_data_for_pages(400)
df = pd.DataFrame.from_records(data)
save_df_to_csv(df)

# Handling data

df = load_csv("Resturants Output/6kdata.csv")

fill_empty_binary_values(df)
df = df[df.num_of_reviews != 0]
df = df.loc[:, (df != 0).any(axis=0)] #Removes columns with zeros only
df = type_to_int(df)
df = split_loc(df)
update_score(df)

# EDA

get_highly_correlated_cols(df)
heat_map(df)
show_rest_map(df)
show_histograms(df)
show_boxplot(df)


# Regression

knn_graph(df)

df1 = dim_reduce_PCA(df, 23)  # reduce the dimension of the feature matrix
lin_regression(df)
knn_regression(df)

network(df1)
