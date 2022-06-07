# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:36:58 2022

@author: Juan Diego
"""

import pandas as pd
import numpy as np
from numpy import median
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import matplotlib.pylab as plt2

    #Importing the data
    
file = "Grover Pricing.xlsx"
xls = pd.ExcelFile(file)
print(xls.sheet_names)

df = xls.parse(1, skiprows=[1], names = ["Ranking", "Cat", "Brand", "Rentals", "Returns", "Page_views", "Market_price", "1M_price", "3M_price", "6M_price", "12M_price"])

    #Analyzing the data globally, columns and datatypes

df.head()
df.shape
df.dtypes
df.info()
df.describe()

    #Do we have any null values we should be worried about?
    
df.isnull().sum()

    #We do, so let's replace null values for 0 in this case

df.loc[df.Returns.isnull(), ["Returns"]] = 0

    #Let's see if it worked
    
df.isnull().sum()

    #Perfect, no null values, what about duplicates?
    
duplicates = df.duplicated() #goes row by row, creates a Series with True if duplicate, False otherwise
duplicates.sum() #the Sum of the series is 0, which means there are no True values, and that there are no duplicate rows
df[duplicates] #trying to get the rows that are True, but we get an empty df

    #We need to change a couple of columns types with .astype, to work with correct data
    
df.dtypes
df["Ranking"] = df["Ranking"].astype("object")
df["Returns"] = df["Returns"].astype("int64")
df["1M_price"] = df["1M_price"].astype("str")
df["3M_price"] = df["3M_price"].astype("str")
df["6M_price"] = df["6M_price"].astype("str")
df["12M_price"] = df["12M_price"].astype("str")

df.dtypes
    
    #Let's split the Cat column to break it down into Category - Subcategory
    
df[["Category", "Subcategory"]] = df["Cat"].str.split("/",expand=True)

    #Make sure everything went smoothly, no null values and correct datatype

df["Category"].isnull().sum()
df["Subcategory"].isnull().sum()
df.dtypes
    
    #Let's count how many items have 1M, 3M, 6M and 12M discounts
    
items_1M_dis = df["1M_price"].str.contains(", ").sum() #146 items with discount
items_3M_dis = df["3M_price"].str.contains(", ").sum() #158 items with discount
items_6M_dis = df["6M_price"].str.contains(", ").sum() #162 items with discount
items_12M_dis = df["12M_price"].str.contains(", ").sum() #163 items with discount

    #Let's create a column that will tell you whether each item has a discount at 1M, 3M, 6M, 12M
    
df["1M_dis?"] ="No"
df["3M_dis?"] ="No"
df["6M_dis?"] ="No"
df["12M_dis?"] ="No"

df.dtypes

for r in df.index:
    if ", " in df.iloc[r, 7]:
        df.iloc[r, 13] = "Yes"

for r in df.index:
    if ", " in df.iloc[r, 8]:
        df.iloc[r, 14] = "Yes"

for r in df.index:
    if ", " in df.iloc[r, 9]:
        df.iloc[r, 15] = "Yes"

for r in df.index:
    if ", " in df.iloc[r, 10]:
        df.iloc[r, 16] = "Yes"

    #Let's split the price columns into full price - discounted price columns

m_prices = ["1M_price", "3M_price", "6M_price", "12M_price"]

for i in m_prices:
    df[[i + "_dis", i + "_full"]] = df[i].str.split(", ",expand=True)

    #Now, for the products that have NO discount, we need to assign them the same price
    #they have in the price_discount column, to get rid of None values

    #For 1M
    
for r in df.index:
    if (df.iloc[r, 13] == "No") & (df.iloc[r, 18] == None):
        df.iloc[r, 18] = df.iloc[r, 17 ]

    #For 3M

for r in df.index:
    if (df.iloc[r, 14] == "No") & (df.iloc[r, 20] == None):
        df.iloc[r, 20] = df.iloc[r, 19 ]

    #For 6M

for r in df.index:
    if (df.iloc[r, 15] == "No") & (df.iloc[r, 22] == None):
        df.iloc[r, 22] = df.iloc[r, 21 ]
   
    #For 12M        
   
for r in df.index:
    if (df.iloc[r, 16] == "No") & (df.iloc[r, 24] == None):
        df.iloc[r, 24] = df.iloc[r, 23 ]

    #Now let's see the % of how important the discounts are, but first we have to make
    #sure all of our prices are floats
    
df.dtypes

we_need_floats = ["1M_price_dis", "1M_price_full", "3M_price_dis", "3M_price_full", "6M_price_dis", "6M_price_full", "12M_price_dis", "12M_price_full"]

for f in we_need_floats:
    df[f] = df[f].astype("float")
    
df.dtypes

    #Now, let's create the new columns for the calculation of the discount in %

period_list = ["1M", "3M", "6M", "12M"]
discount = "_price_dis"
full = "_price_full"

for p in period_list:
    df[p + "_dis_%"] = round((df[p + full]-df[p + discount])/df[p + discount], 2)
    
    #Let's check for nulls because I saw some again
    
df["1M_dis_%"] = df["1M_dis_%"].fillna(0)
df["3M_dis_%"] = df["3M_dis_%"].fillna(0)
df["6M_dis_%"] = df["6M_dis_%"].fillna(0)
df["12M_dis_%"] = df["12M_dis_%"].fillna(0)

df.isnull().sum()

    #Let's create one more column that tells us whether the item has discount in all 4 plans, or none of them

df["dis_4_plans?"] = "No discount on all 4 plans"

for r in df.index:
    if (df.iloc[r, 13] == "Yes") & (df.iloc[r, 14] == "Yes") & (df.iloc[r, 15] == "Yes") & (df.iloc[r, 16] == "Yes"):
        df.iloc[r, 29] = "Discount on all 4 plans!"
        
df["dis_0_plans?"] = "No discount on any of the 4 plans"

for r in df.index:
    if (df.iloc[r, 13] == "Yes") | (df.iloc[r, 14] == "Yes") | (df.iloc[r, 15] == "Yes") | (df.iloc[r, 16] == "Yes"):
        df.iloc[r, 30] = "Discount on at least 1 plan!"
        
df.loc[(df["dis_4_plans?"] == "No discount on all 4 plans") & (df["dis_0_plans?"] == "Discount on at least 1 plan!"), :] #there are only 23 items that have promotions in some period, but no the 4 of them

    #now let's look at the natural discount we give based on the duration of contracts
    
df["nat_dis_%_1to3"] = round((df["1M_price_full"] - df["3M_price_full"]) / df["1M_price_full"], 2)
df["nat_dis_%_3to6"] = round((df["3M_price_full"] - df["6M_price_full"]) / df["3M_price_full"], 2)
df["nat_dis_%_6to12"] = round((df["6M_price_full"] - df["12M_price_full"]) / df["6M_price_full"], 2)

    #finally, let's look at how much the % of discount given varies with the subscription lengt
    
df["delta_dis_3vs1"] = df["3M_dis_%"]- df["1M_dis_%"]
df["delta_dis_6vs3"] = df["6M_dis_%"]- df["3M_dis_%"]
df["delta_dis_12vs6"] = df["12M_dis_%"]- df["6M_dis_%"]

    #Let's make a ratio to see the relationship between rentals and returns (how many times more rentals than returns)
    
df["rent_vs_rtrn"] = round(df["Rentals"]/df["Returns"], 2)
df["rent_vs_rtrn"].unique()

df["rent_vs_rtrn"] = df["rent_vs_rtrn"].fillna(0)
df.loc[df["rent_vs_rtrn"] == np.inf, ["rent_vs_rtrn"]]= 2

    #Now let's see how effective (or not) the webpage of each item is at sealing the deal
    
df["conversion"] = round(df["Rentals"]/df["Page_views"], 2)
df["conversion"].unique()

df["conversion"] = df["conversion"].fillna(0)
df.loc[df["conversion"] == np.inf, ["conversion"]]= 0


    #Let's create some ratios that compare the market price with the renting price
    
df["1M_vs_mp"] = df["1M_price_full"]/df["Market_price"]
df["3M_vs_mp"] = (df["3M_price_full"]*3)/df["Market_price"]
df["6M_vs_mp"] = (df["6M_price_full"]*6)/df["Market_price"]
df["12M_vs_mp"] = (df["12M_price_full"]*12)/df["Market_price"]

    #How many months at each price would one have to pay in order to be able to buy a new one?
    
df["subs_tobuy_new_1M"] = round(df["Market_price"]/df["1M_price_full"], 0)
df["subs_tobuy_new_3M"] = round(df["Market_price"]/(df["3M_price_full"]*3), 0)
df["subs_tobuy_new_6M"] = round(df["Market_price"]/(df["6M_price_full"]*6), 0)
df["subs_tobuy_new_12M"] = round(df["Market_price"]/(df["12M_price_full"]*12), 0)


    #Now, let's get into EDA.

df.Rentals.describe()
df.Returns.describe()
df.Page_views.describe()
df.Market_price.describe()
df["1M_price_full"].describe()
df["3M_price_full"].describe()
df["6M_price_full"].describe()
df["12M_price_full"].describe()
df["1M_dis_%"].describe()
df["3M_dis_%"].describe()
df["6M_dis_%"].describe()
df["12M_dis_%"].describe()
df["rent_vs_rtrn"].describe()
df["conversion"].describe()
df["1M_vs_mp"].describe()
df["3M_vs_mp"].describe()
df["6M_vs_mp"].describe()
df["12M_vs_mp"].describe()
df["subs_tobuy_new_1M"].describe()
df["subs_tobuy_new_3M"].describe()
df["subs_tobuy_new_6M"].describe()
df["subs_tobuy_new_12M"].describe()
    
sns.scatterplot(x="Page_views", y="Rentals", data=df.loc[df["Page_views"]<= 6000, :])
sns.scatterplot(x="Page_views", y="Rentals", data=df, hue="Category")
sns.scatterplot(x="Page_views", y="Rentals", data=df, hue="dis_4_plans?")
sns.scatterplot(x="Page_views", y="Rentals", data=df, hue="dis_0_plans?")

sns.scatterplot(x="Market_price", y="Rentals", data=df.loc[df["Rentals"]>= 0, :])
sns.scatterplot(x="Market_price", y="Rentals", data=df, hue="Category")
sns.scatterplot(x="Market_price", y="Rentals", data=df, hue="dis_4_plans?")
sns.scatterplot(x="Market_price", y="Rentals", data=df, hue="dis_0_plans?")

sns.countplot(x="Rentals", data=df[df["Rentals"] != 0])
sns.countplot(x="Returns", data=df[df["Returns"] != 0])
sns.countplot(x="Page_views", data=df)
sns.countplot(x="Market_price", data=df)
sns.countplot(x="Category", data=df)
sns.countplot(x="dis_4_plans?", data=df)
sns.countplot(x="dis_0_plans?", data=df)

sns.relplot(x="Market_price", y="Rentals", data=df, kind="scatter", col="Category")
sns.relplot(x="6M_price", y="Rentals", data=df, kind="scatter", col="Category")
sns.relplot(x="6M_price", y="Rentals", data=df, kind="scatter", col="dis_4_plans?")

sns.relplot(x="3M_price", y="Rentals", data=df, kind="scatter", size="3M_vs_mp")

sns.relplot(x="Category", y="Rentals", data=df, kind="line", ci="sd")

    #lines do not make sense, we do not have a time dimension here
    
sns.catplot(kind="count", x="Category", data=df, col="dis_4_plans?")

sns.catplot(kind="bar", x="Category", y="Rentals", data=df)
sns.catplot(kind="bar", x="Category", y="Rentals", data=df, ci=None) #you can order with order = ["a", "b"]

sns.catplot(kind="bar", x="dis_4_plans?", y="Rentals", data=df)
sns.catplot(kind="bar", x="dis_0_plans?", y="Rentals", data=df)

    #How about some boxplots?
    
sns.catplot(kind="box", x="Category", y="Rentals", data=df)
sns.catplot(kind="box", x="Category", y="Returns", data=df)
sns.catplot(kind="box", x="Category", y="Market_price", data=df)
sns.catplot(kind="box", x="Category", y="Page_views", data=df)
sns.catplot(kind="box", x="Category", y="1M_dis_%", data=df)
sns.catplot(kind="box", x="Category", y="3M_dis_%", data=df)
sns.catplot(kind="box", x="Category", y="6M_dis_%", data=df)
sns.catplot(kind="box", x="Category", y="12M_dis_%", data=df)
sns.catplot(kind="box", x="Category", y="conversion", data=df)

sns.catplot(kind="box", x="Category", y="Rentals", data=df, whis=[5, 95])

sns.catplot(kind="point", x="Category", y="Returns", data=df, hue="Category", estimator=median)

    #Distribution plots

sns.displot(df["Category"])
sns.displot(df["Subcategory"])
sns.displot(df["Brand"])
sns.displot(df["1M_dis?"])
sns.displot(df["3M_dis?"])
sns.displot(df["6M_dis?"])
sns.displot(df["12M_dis?"])
sns.displot(df["dis_4_plans?"])
sns.displot(df["dis_4_plans?"])
sns.displot(df["Rentals"], bins=50)
sns.displot(df["Returns"], bins=50)
sns.displot(df["Page_views"], bins=50)
sns.displot(df["Market_price"], bins=20)


    #Regression plots
    
sns.regplot(data=df, x="Market_price", y="Rentals")
sns.lmplot(data=df, x="Market_price", y="Rentals")
sns.lmplot(data=df, x="Market_price", y="Rentals", hue="Category", row="Category")

    #Now, let's start asking questions that might be relevant and finding
    #The answer on python, being with just numbers or visualizations
    
    # How many different items we have in the dataset?
    
total_items = df.shape[0]

    # How are these items distributed in terms of Category
    
'''df["Category"].value_counts()
cat_order = list(df["Category"].value_counts().index)

plt.figure(figsize=(10,6))
by_cat = sns.countplot(data=df, x="Category", order = cat_order, palette = "Set2")
for bar in by_cat.patches:
    by_cat.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=14, xytext=(0, 8),
                   textcoords='offset points')
by_cat.set_xticklabels(by_cat.get_xticklabels(), rotation=80, fontsize=12)'''

by_cat = df.groupby(by=["Category"], as_index=False).Rentals.count().sort_values(by="Rentals", ascending=False)

by_cat["Category"] = pd.Categorical(by_cat.Category, categories =["computers", "audio-and-music", "cameras", "phones-and-tablets", "home-entertainment", "wearables", "smart-home", "gaming-and-vr", "drones", "e-mobility"]) 

ggplot(aes(x="Category", y="Rentals", fill="Category"), by_cat) +\
    geom_bar(stat="identity")\
        + theme_classic() +\
            labs(title = "Qty items by Category", x = "Category", y = "Qty")\
                + theme(axis_text_x=element_text(size=9, weight="bold", color="black", rotation=75), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="Rentals"), size=10, color="black")\
                        + scale_fill_brewer(type = "qual", palette = "Set3" )

by_cat["%_of_total"] = round(by_cat["Rentals"]/total_items, 2)

ggplot(aes(x="Category", y="Rentals", fill="Category"), by_cat) +\
    geom_bar(stat="identity")\
        + theme_classic() +\
            labs(title = "Qty items by Category", x = "Category", y = "Qty")\
                + theme(axis_text_x=element_text(size=9, weight="bold", color="black", rotation=75), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="%_of_total"), size=10, color="black")\
                        + scale_fill_brewer(type = "qual", palette = "Set3" )
                    
by_cat["cumm_%"] = round(np.cumsum(by_cat["%_of_total"]), 2)

ggplot(aes(x="Category", y="cumm_%", fill="Category"), by_cat) +\
    geom_bar(stat="identity", fill=["#d45555" if p <= 0.5 else "#46b06a" if  p <=0.86 else "#404040" for p in list(by_cat["cumm_%"])])\
        + theme_classic() +\
            labs(title = "Qty items by Category", x = "Category", y = "Cumm_items")\
                + theme(axis_text_x=element_text(size=9, weight="bold", color="black", rotation=75), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="cumm_%"), size=10, color="black")
                    
        # In terms of subcategory
        
by_subcat = df.groupby(by=["Subcategory"], as_index=False).Rentals.count().sort_values(by="Rentals", ascending=False)

sublist = list(df.groupby(by=["Subcategory"]).Rentals.count().sort_values(ascending=False).index)

by_subcat["Subcategory"] = pd.Categorical(by_subcat.Subcategory, categories = sublist)

ggplot(aes(x="Subcategory", y="Rentals"), by_subcat) +\
    geom_bar(stat="identity")\
        + theme_classic() +\
            labs(title = "Qty items by Subcategory", x = "Subategory", y = "Qty")\
                + theme(axis_text_x=element_text(size=8, color="black", rotation=90), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="Rentals"), size=8, color="black")

by_subcat["%_of_total"] = round(by_subcat["Rentals"]/total_items, 2)

ggplot(aes(x="Subcategory", y="Rentals"), by_subcat) +\
    geom_bar(stat="identity")\
        + theme_classic() +\
            labs(title = "Qty items by Subcategory", x = "Subategory", y = "Qty")\
                + theme(axis_text_x=element_text(size=8, color="black", rotation=90), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="%_of_total"), size=8, color="black")
                    
by_subcat["cumm_%"] = round(np.cumsum(by_subcat["%_of_total"]), 2)

ggplot(aes(x="Subcategory", y="cumm_%"), by_subcat) +\
    geom_bar(stat="identity", fill=["#d45555" if p <= 0.5 else "#46b06a" if  p <=0.86 else "#404040" for p in list(by_subcat["cumm_%"])])\
        + theme_classic() +\
            labs(title = "Qty items by Subcategory", x = "Subategory", y = "Qty")\
                + theme(axis_text_x=element_text(size=8, color="black", rotation=90), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=14, hjust = 0.5))\
                    + geom_text(aes(label="cumm_%"), size=8, color="black")
                    
        # You have 43 subcats. With 3 of them you reach 25% of your different products, with 7 50% of your different products.
                    
        # How many different brands we have?
        
df.Brand.nunique() #191 different brands for 2139 different items

        # Amount of items per brand
        
by_brand = df.groupby(by=["Brand"], as_index=False).Rentals.count().sort_values(by="Rentals", ascending=False)

        # Let's see % of total and distribution

by_brand["%_of_total"] = round(by_brand["Rentals"]/total_items, 4)
by_brand["cumm_%"] = round(np.cumsum(by_brand["%_of_total"]), 2)

        # You have 191 different brands, and 2139 items. With just the 3 top brands (in terms
        # of number of items) you get 26% of all items, and with your top 10 brands (just 5% of 
        #total brands) you get 50% of your different items.
        
        # Let's give brands a ranking based on total number of items
        
by_brand["items_ranking"] = [i+1 for i in range(191)]

by_brand_top = by_brand.loc[by_brand["items_ranking"] <= 35, :]

sublist2 = list(by_brand_top["Brand"])

by_brand_top["Brand"] = pd.Categorical(by_brand_top.Brand, categories = sublist2)

ggplot(aes(x="Brand", y="cumm_%"), by_brand_top) +\
    geom_bar(stat="identity", fill=["#d45555" if p <= 3 else "#46b06a" if  p <=10 else "#404040" for p in list(by_brand_top.items_ranking)])\
        + theme_classic() +\
            labs(title = "Cummulated items by brand", x = "Brand", y = "% total items")\
                + theme(axis_text_x=element_text(size=8, color="black", rotation=90), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=12, hjust = 0.5))\
                    + geom_text(aes(label="cumm_%"), size=8, color="black")
                    
        #Let's analyze rentals now

        #How many rentals in the period 
        
total_rentals = df["Rentals"].sum()
avg_rentals = df["Rentals"].mean()
max_rentals = df["Rentals"].max()
min_rentals = df["Rentals"].min()

        #Top items with the most rentals
        
df_rentals = df.sort_values(by="Rentals", ascending=False)
df_rentals["%_of_total"] = round(df_rentals["Rentals"]/total_rentals, 2)
df_rentals["cumm_%"] = round(np.cumsum(df_rentals["%_of_total"]), 2)
df_rentals = df_rentals.loc[df_rentals["cumm_%"] <= 0.5, :]
df_rentals["rental_ranking"] = [i+1 for i in range(38)]

ggplot(aes(x="rental_ranking", y="cumm_%"), df_rentals) +\
    geom_bar(stat="identity", fill=["#d45555" if p <= 0.25 else "#46b06a" if  p <=0.4 else "#404040" for p in list(df_rentals["cumm_%"])])\
        + theme_classic() +\
            labs(title = "Cummulated rentals by item", x = "Items with the most rentals", y = "% of total rents cummulated")\
                + theme(axis_text_x=element_text(size=7, color="black", rotation=90), axis_text_y=element_text(size=10, color="black") \
                        ,axis_title=element_text(size=12),\
                        plot_title=element_text(weight='bold', size=12, hjust = 0.5))\
                    + geom_text(aes(label="cumm_%"), size=8, color="black")
                    
        # There are 2139 items. If we sorted them from the ones with the most rentals to the ones 
        # with the least rentals and sort them: 9 items are 20% of all rentals,
        # 13 items cummulate 25% of all rentals in the period, and 38 items cummulate 50% of all rentals
        # This means that the first 38 items represent 50% of rentals, while the other 2101 items cummulate the other 50%.
        
        #How many items with 0 rentals? And with less than 10, 20?
        
df_worst = df.loc[df["Rentals"] <= 20, :]

df_worst["low_rental"] = pd.cut(df_worst.Rentals, bins=[-100, 0, 1, 5, 10, np.inf], labels = ["Zero", "Unused", "Between 1 and 5", "Between 5 and 10", "Between 10 and 20"])

df_worst.value_counts("low_rental")

        # Out of the the 2139 items we have, 2010 have had 20 or less rentals (so only 129 have more than 20 rentals in the period):
        # 1358 (63% of total items!) have had 0 rentals in the period.
        # 414 (19% of total items) had between 1 and 5 rentals
        # 125 (6% of total items) had between 5 and 10 rentals
        # 113 (5% of total items) had between 10 and 20 rentals

        # Items with the best and worst rental-return ratio?
        
df["rent_vs_rtrn"].mean() #average was 0.64, which means rentals were 0.64 of returns, on average
df["rent_vs_rtrn"].min()
df["rent_vs_rtrn"].max() #max was 19.5, which means there was an item that was rented 19 times more than returned. (38 rentals, 2 returns)

df_ratio = df.sort_values(by="rent_vs_rtrn", ascending = False)
df_ratio["rr_ratio"] = pd.cut(df_ratio.rent_vs_rtrn, bins=[-100, 0, 1, 2, 5, 10, np.inf], labels = ["Zero", "btwn 0-1", "btwn 1 and 2", "btwn 2 and 5", "btwn 5 and 10", "More than 10"])
top100_ratio = df_ratio.loc[:, ["Ranking", "Category", "Subcategory", "Brand", "Rentals", "Returns", "rent_vs_rtrn", "Page_views", "Market_price", "12M_vs_mp"]].head(100)

df_ratio.value_counts("rr_ratio")

        # Out of the 2139 items, I made a rental-return ratio. I classified it, results:
        # 1358 (63%) items have a ratio of 0, because of 0 rentals.
        # 267 (18%) items have a ratio between 0 and 1, which means items with more returns than rentals 
        # 389 (12%) items have a ratio between 1 and 2, which means items with almost x2 more rentals than returns
        # 108 (5%) items have a ratio between 2 and 5, which means items with almost x5 more rentals than returns
        # 13 (0.6%) items have a ratio between 5 and 10
        # 4 (0.2%) items have a ratio bigger than 10

    # Rental-return ratio by category, subcategory, brand
    
rr_cat = df_ratio.groupby(by="Category", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False)
rr_subcat = df_ratio.groupby(by="Subcategory", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False)
rr_brand = df_ratio.groupby(by="Brand", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False).head(20)

    # For the top 100 products with the better rr_ratio, how do they look by cat, subcat, brand?
    
top100_ratio.groupby(by="Category", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False)
top100_ratio.groupby(by="Subcategory", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False)
top100_ratio.groupby(by="Brand", as_index=False).rent_vs_rtrn.mean().sort_values(by="rent_vs_rtrn", ascending=False)

    # Let's take a look at rentals by Cat, Subcat, Brand, dis_0_plans
    
df.groupby(by="Category", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)
df.groupby(by="Subcategory", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)
df.groupby(by="Brand", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)
df.groupby(by="dis_0_plans?", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)

    # Let's look at rentals by price ratio between renting for 12 months and buying it

df_vs_market = df.loc[~df["12M_vs_mp"].isin([0, np.inf]), :]
df_vs_market["12M_vs_mp"].describe()
df_vs_market["Price_class"] = pd.cut(df_vs_market["12M_vs_mp"], bins=[0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, np.inf], labels = ["<0.3 of mp", "btwn 0.3-0.5 of mp", "btwn 0.5-0.7 of mp", "btwn 0.7-0.9 of mp", "btwn 0.9-1 of mp", "btwn 1-1.5 of mp", "more than double"])
df_vs_market["Price_class"] = pd.Categorical(df_vs_market["Price_class"], categories = ["<0.3 of mp", "btwn 0.3-0.5 of mp", "btwn 0.5-0.7 of mp", "btwn 0.7-0.9 of mp", "btwn 0.9-1 of mp", "btwn 1-1.5 of mp", "more than double"])
df_2 = df_vs_market.groupby(by="Price_class", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)
df_2["% of total"] = round(df_2["Rentals"]/total_rentals, 2)
print(df_2)

    # Key finding there, be sure to include it
    # What about sum of rentals for market price buckets?
    
df.Market_price.describe()

df_3 = df.loc[~df["Market_price"].isin([0]), :]
df_3["Price_bucket"] = pd.cut(df_3.Market_price, bins=[0, 100, 200, 300, 500, 750, 1000, 1500, 2000, np.inf], labels = ["<100", "btwn 100-200", "btwn 200-300", "btwn 300-500", "btwn 500-750", "btwn 750-1000", "btwn 1000-1500", "btwn 1500-2000", "More than 2000"])

df_3=df_3.groupby(by="Price_bucket", as_index=False).Rentals.sum().sort_values(by="Rentals", ascending=False)
df_3["% of total"] = round(df_3["Rentals"]/total_rentals, 2)
print(df_3)

    # Key finding here as well
    
    # Now let's look at the returns figures. Start with returns by item

total_returns = df["Returns"].sum()
top_returned = df.sort_values(by="Returns", ascending=False).head(50).loc[:, ["Ranking", "Category", "Subcategory", "Brand", "Rentals", "Returns", "rent_vs_rtrn", "Page_views", "Market_price", "12M_vs_mp"]]
top_returned["%_oftotal"] = round(top_returned["Returns"]/total_returns, 2)
top_returned["cumm_%"] = round(np.cumsum(top_returned["%_oftotal"]), 2)
top_returned["return_ranking"] = [i+1 for i in range(50)]

    # If you analyze the items with most returns, and sort the top 50, with just 
    # 10 items you get 20% of all total returns, with 40 you get 50% of total returns.
    
    # For the top 50 most returned items, what is their category? Subcat? Brand?
    
top_returned.groupby(by="Category", as_index=False).Returns.sum().sort_values(by="Returns", ascending=False)
top_returned.groupby(by="Subcategory", as_index=False).Returns.sum().sort_values(by="Returns", ascending=False)
top_returned.groupby(by="Brand", as_index=False).Returns.sum().sort_values(by="Returns", ascending=False)

top_returned["Price_class"] = pd.cut(top_returned["12M_vs_mp"], bins=[0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, np.inf], labels = ["<0.3 of mp", "btwn 0.3-0.5 of mp", "btwn 0.5-0.7 of mp", "btwn 0.7-0.9 of mp", "btwn 0.9-1 of mp", "btwn 1-1.5 of mp", "more than double"])
top_returned["Price_bucket"] = pd.cut(top_returned.Market_price, bins=[0, 100, 200, 300, 500, 750, 1000, 1500, 2000, np.inf], labels = ["<100", "btwn 100-200", "btwn 200-300", "btwn 300-500", "btwn 500-750", "btwn 750-1000", "btwn 1000-1500", "btwn 1500-2000", "More than 2000"])

top_returned.groupby(by="Price_class", as_index=False).Returns.sum().sort_values(by="Returns", ascending=False)
top_returned.groupby(by="Price_bucket", as_index=False).Returns.sum().sort_values(by="Returns", ascending=False)

    # Now let's look at Page views. Let's figure out total page views
    
total_views = df.Page_views.sum()

    # Now let's build the dataframe, and start by looking at % of total views
    
viewed = df.sort_values(by="Page_views", ascending=False).loc[:, ["Ranking", "Category", "Subcategory", "Brand", "Rentals", "Returns", "rent_vs_rtrn", "Page_views", "Market_price", "12M_vs_mp", "conversion"]]
viewed["%_of_total"] = viewed["Page_views"]/total_views
viewed["cumm_%"] = np.cumsum(viewed["%_of_total"])
viewed["views_ranking"] = [i+1 for i in range(2139)]

    # The 12 most viewed products make up 20% of all views, and the top 76
    # make up 50% of all views. This means that the other 2036 make up the other 50%
    
    # Let's analyze page views(all of them) by Category, Subcategory, Brand
    
df3 = viewed.groupby(by="Category", as_index=False).Page_views.sum().sort_values(by="Page_views", ascending=False)
df3["%_of_total"] = df3["Page_views"]/total_views
print(df3)

df4 = viewed.groupby(by="Subcategory", as_index=False).Page_views.sum().sort_values(by="Page_views", ascending=False)
df4["%_of_total"] = df4["Page_views"]/total_views
print(df4)

df5 = viewed.groupby(by="Brand", as_index=False).Page_views.sum().sort_values(by="Page_views", ascending=False)
df5["%_of_total"] = df5["Page_views"]/total_views
print(df5)

    # Key takeaway, INCLUDE THIS ABOVE. What about conversion now?
    
ggplot(aes(x="Page_views", y="Rentals"), df.loc[df.Page_views <= 4000, :]) + geom_point() +theme_bw()

avg_conversion = viewed["conversion"].mean()

    # How does the average conversion of all items vary against the conversion of the most viewed items (50% views)

top76_conversion = viewed.loc[viewed["views_ranking"] <= 76 , :].conversion.mean()

    # Average conversion by cat, subcat, brand, price
    
viewed.groupby(by="Category", as_index=False).conversion.mean().sort_values(by="conversion", ascending=False)
viewed.groupby(by="Subcategory", as_index=False).conversion.mean().sort_values(by="conversion", ascending=False)
viewed.groupby(by="Brand", as_index=False).conversion.mean().sort_values(by="conversion", ascending=False)

viewed["Price_class"] = pd.cut(viewed["12M_vs_mp"], bins=[0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, np.inf], labels = ["<0.3 of mp", "btwn 0.3-0.5 of mp", "btwn 0.5-0.7 of mp", "btwn 0.7-0.9 of mp", "btwn 0.9-1 of mp", "btwn 1-1.5 of mp", "more than double"])
viewed["Price_bucket"] = pd.cut(viewed.Market_price, bins=[0, 100, 200, 300, 500, 750, 1000, 1500, 2000, np.inf], labels = ["<100", "btwn 100-200", "btwn 200-300", "btwn 300-500", "btwn 500-750", "btwn 750-1000", "btwn 1000-1500", "btwn 1500-2000", "More than 2000"])

viewed.groupby(by="Price_class", as_index=False).conversion.mean().sort_values(by="conversion", ascending=False)
viewed.groupby(by="Price_bucket", as_index=False).conversion.mean().sort_values(by="conversion", ascending=False)

    # Finally, let's analzyze Price. Let's start with creating the price buckets, and then counting how many items in each one

df["Price_class"] = pd.cut(df["12M_vs_mp"], bins=[0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, np.inf], labels = ["<0.3 of mp", "btwn 0.3-0.5 of mp", "btwn 0.5-0.7 of mp", "btwn 0.7-0.9 of mp", "btwn 0.9-1 of mp", "btwn 1-1.5 of mp", "more than double"])
df["Price_bucket"] = pd.cut(df.Market_price, bins=[0, 100, 200, 300, 500, 750, 1000, 1500, 2000, np.inf], labels = ["<100", "btwn 100-200", "btwn 200-300", "btwn 300-500", "btwn 500-750", "btwn 750-1000", "btwn 1000-1500", "btwn 1500-2000", "More than 2000"])
df["rr_ratio"] = pd.cut(df.rent_vs_rtrn, bins=[-100, 0, 1, 2, 5, 10, np.inf], labels = ["Zero", "btwn 0-1", "btwn 1 and 2", "btwn 2 and 5", "btwn 5 and 10", "More than 10"])

df6 = df.groupby(by="Price_class", as_index=False).Ranking.count().sort_values(by="Ranking", ascending=False)
df6["%_of_total"] = df6["Ranking"]/total_items
print(df6)

df7 = df.groupby(by="Price_bucket", as_index=False).Ranking.count().sort_values(by="Ranking", ascending=False)
df7["%_of_total"] = df7["Ranking"]/total_items
print(df7)

    # Let's start with Avg market price, 1M, 3M, 6M, 12M, 12M_vs_mp for all rows
    
avg_mkt_price = df["Market_price"].mean()
avg_1M_price = df["1M_price_full"].mean()
avg_3M_price = df["3M_price_full"].mean()
avg_6M_price = df["6M_price_full"].mean()
avg_12M_price = df["12M_price_full"].mean()
avg_price_ratio = df.loc[~df["12M_vs_mp"].isin([0, np.inf])]["12M_vs_mp"].mean()

    # Now, avg price by Category, subcategory, brand y relacion mp con 12m price para mkt price, 1M, 3M, 6M, 12M y 12M_vs_mp
    
    # First, market price
    
df.groupby(by="Category", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)
df.groupby(by="Subcategory", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)
df.groupby(by="Brand", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)
df.groupby(by="Price_class", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False) #very interesting result here!
df.groupby(by="dis_4_plans?", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)
df.groupby(by="dis_0_plans?", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)
df.groupby(by="rr_ratio", as_index=False).Market_price.mean().sort_values(by="Market_price", ascending=False)

    # Now, let's do 1M, 3M, 6M, 12M and 12M_vs_mp with a loop
    
looping = ["1M_price_full", "3M_price_full", "6M_price_full", "12M_price_full", "12M_vs_mp"]

for c in looping:
    print("This is the START of", c, "tables\n")
    print(df.groupby(by="Category", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="Subcategory", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="Brand", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="Price_class", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="dis_4_plans?", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="dis_0_plans?", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print(df.groupby(by="rr_ratio", as_index=False)[c].mean().sort_values(by=c, ascending=False))
    print("This is the END of", c, "tables\n")
    
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="Category", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="Subcategory", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="Brand", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="Price_class", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="dis_0_plans?", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
df.loc[~df["12M_vs_mp"].isin([0, np.inf]),:].groupby(by="rr_ratio", as_index=False)["12M_vs_mp"].mean().sort_values(by="12M_vs_mp", ascending=False)
    
    # How does price affects rentals?
    
ggplot(aes(x="Market_price", y="Rentals"), df) + geom_point() +theme_bw()

    # Let's divide into items that had over 100 rentals, and those under less
    
ggplot(aes(x="Market_price", y="Rentals"), df.loc[df.Rentals >= 100, :]) + geom_point() +theme_bw() #for very popular items, price doesnt seem to be so relevant
ggplot(aes(x="Market_price", y="Rentals"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #for everyting else, there seems to be a weak connection

    # Let's look at how the other prices (and ratio) are related to the rentals

ggplot(aes(x="1M_price_full", y="Rentals"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #you can see it here as well
ggplot(aes(x="3M_price_full", y="Rentals"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #you can see it here as well
ggplot(aes(x="6M_price_full", y="Rentals"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #you can see it here as well
ggplot(aes(x="12M_price_full", y="Rentals"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #you can see it here as well
ggplot(aes(x="12M_vs_mp", y="Rentals", color="Price_class"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw() #BINGO, when the 12m_fare is closer to the market price or above it, demand falls rapidly
ggplot(aes(x="12M_vs_mp", y="Rentals",color="rr_ratio"), df.loc[(df.Rentals < 100) & (df.Rentals != 0), :]) + geom_point() +theme_bw()

    # Are there categories that are more and less elastic?

h=list(df["Category"].unique())

for i in h:
    print("THIS IS THE GRAPH FOR", i)
    print(ggplot(aes(x="12M_vs_mp", y="Rentals", color="Price_class"), df.loc[(df.Rentals < 100) & (df.Rentals != 0) & (df.Category == i), :]) + geom_point() +theme_bw() + ggtitle(i))
    print("THIS WAS THE LAST GRAPH FOR", i)

    # The answer is YES, check the graphs above and single out results
    
    # Let's look at average discounts for 1M, 3M, 6M, 12M, for all items WITH DISCOUNT, and by cat, subcat, brand, absolute price
    
avg_1M_dis = df.loc[df["1M_dis_%"] != 0, :]["1M_dis_%"].mean()
avg_3M_dis = df.loc[df["3M_dis_%"] != 0, :]["3M_dis_%"].mean()
avg_6M_dis = df.loc[df["6M_dis_%"] != 0, :]["6M_dis_%"].mean()
avg_12M_dis = df.loc[df["12M_dis_%"] != 0, :]["12M_dis_%"].mean()

    # Now let's look at the avg discounts by cat, subcat, brand, price_bucket & price_class
    
df.loc[df["1M_dis_%"] != 0, :].groupby(by="Category", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False)
df.loc[df["1M_dis_%"] != 0, :].groupby(by="Subcategory", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False)
df.loc[df["1M_dis_%"] != 0, :].groupby(by="Brand", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False)
df.loc[df["1M_dis_%"] != 0, :].groupby(by="Price_class", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False) #from 1 to 1.5 only 7%?
df.loc[df["1M_dis_%"] != 0, :].groupby(by="Price_bucket", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False) #strongest discount from 500 to 1000
df.loc[df["1M_dis_%"] != 0, :].groupby(by="rr_ratio", as_index=False)["1M_dis_%"].mean().sort_values(by=["1M_dis_%"], ascending=False) #check this one out!

df.loc[df["3M_dis_%"] != 0, :].groupby(by="Category", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False)
df.loc[df["3M_dis_%"] != 0, :].groupby(by="Subcategory", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False)
df.loc[df["3M_dis_%"] != 0, :].groupby(by="Brand", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False)
df.loc[df["3M_dis_%"] != 0, :].groupby(by="Price_class", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False) #from 1 to 1.5 only 12%? every other is higher
df.loc[df["3M_dis_%"] != 0, :].groupby(by="Price_bucket", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False) #strongest avg discount from 500 to 1000
df.loc[df["3M_dis_%"] != 0, :].groupby(by="rr_ratio", as_index=False)["3M_dis_%"].mean().sort_values(by=["3M_dis_%"], ascending=False) #the higher the discount, the better ratio

df.loc[df["6M_dis_%"] != 0, :].groupby(by="Category", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False)
df.loc[df["6M_dis_%"] != 0, :].groupby(by="Subcategory", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False)
df.loc[df["6M_dis_%"] != 0, :].groupby(by="Brand", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False)
df.loc[df["6M_dis_%"] != 0, :].groupby(by="Price_class", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False) #from 1 to 1.5 only 16%? smallest avg discount again on this
df.loc[df["6M_dis_%"] != 0, :].groupby(by="Price_bucket", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False) #strongest avg discount on 500-750
df.loc[df["6M_dis_%"] != 0, :].groupby(by="rr_ratio", as_index=False)["6M_dis_%"].mean().sort_values(by=["6M_dis_%"], ascending=False) # the higher the discount, the better ratio

df.loc[df["12M_dis_%"] != 0, :].groupby(by="Category", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False)
df.loc[df["12M_dis_%"] != 0, :].groupby(by="Subcategory", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False)
df.loc[df["12M_dis_%"] != 0, :].groupby(by="Brand", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False)
df.loc[df["12M_dis_%"] != 0, :].groupby(by="Price_class", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False) # finally good discount on 1-1.5 (30%)
df.loc[df["12M_dis_%"] != 0, :].groupby(by="Price_bucket", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False) #higher discount on 500-750
df.loc[df["12M_dis_%"] != 0, :].groupby(by="rr_ratio", as_index=False)["12M_dis_%"].mean().sort_values(by=["12M_dis_%"], ascending=False) #The higher the avg discount, the better that rr ratio

    # Now, let's look at the discounts vs qty rentals (for items with discount)
    
ggplot(aes(x="1M_dis_%", y="Rentals"), df.loc[df["1M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="3M_dis_%", y="Rentals"), df.loc[df["3M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="6M_dis_%", y="Rentals"), df.loc[df["6M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="12M_dis_%", y="Rentals"), df.loc[df["12M_dis_%"] != 0, :]) + geom_point() + theme_bw()

    # On the discounts is not too clear that higher discounts lead to higher rentals, there are too few items to judge
    
    # Now, let's see if the discounts are correlated with the qty of returns (for items with discount)
    
ggplot(aes(x="1M_dis_%", y="Returns"), df.loc[df["1M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="3M_dis_%", y="Returns"), df.loc[df["3M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="6M_dis_%", y="Returns"), df.loc[df["6M_dis_%"] != 0, :]) + geom_point() + theme_bw()
ggplot(aes(x="12M_dis_%", y="Returns"), df.loc[df["12M_dis_%"] != 0, :]) + geom_point() + theme_bw() 

    # In this case, you kind of can see that the higher the discount, maybe the less returns of the item
    
    # Now, let's see, for each value in subscriptions to buy new, what is the sum of rentals
    
df.groupby(by="subs_tobuy_new_1M", as_index=False).Rentals.sum().sort_values(by="subs_tobuy_new_1M", ascending=True) #the sweetspot happens when the monthly sub is 1/8 of the market price. After 8 subscriptions (8 months), rentals rapidly fall
df.groupby(by="subs_tobuy_new_3M", as_index=False).Rentals.sum().sort_values(by="subs_tobuy_new_3M", ascending=True) #the sweetspot happens when the quarterly sub is 1/4 of the market price. After 4 subscriptions(of 3 months, so 1 year), rentals rapidly fall
df.groupby(by="subs_tobuy_new_6M", as_index=False).Rentals.sum().sort_values(by="subs_tobuy_new_6M", ascending=True) #the sweetspot happens when the bi-annual sub is 1/2 of the market price. After 2 subscriptions(of 2 semesters, so 1 year), rentals rapidly fall
df.groupby(by="subs_tobuy_new_12M", as_index=False).Rentals.sum().sort_values(by="subs_tobuy_new_12M", ascending=True) #the sweetspot happens when the annual sub is close to the market price. After 1 subscriptions (of 12 months, so a year), rentals rapidly fall

df8=df.loc[:, ["Ranking", "Rentals", "Returns", "conversion", "rent_vs_rtrn", "Price_class", "rr_ratio"]]





































































































































