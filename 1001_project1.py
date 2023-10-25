import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

df_data = pd.read_csv("movieReplicationSet.csv")
df_movies = df_data.iloc[:,0:400]

#1) Are movies that are more popular (operationalized as having more ratings) rated higher than movies that
#are less popular? [Hint: You can do a median-split of popularity to determine high vs. low popularity movies]

rating_counts = df_movies.count()
high_popularity_movies = rating_counts[rating_counts > rating_counts.median()].index
low_popularity_movies = rating_counts[rating_counts <= rating_counts.median()].index

high_popularity_rating = df_movies[high_popularity_movies].mean()
low_popularity_rating = df_movies[low_popularity_movies].mean()

t1, p1 = stats.ttest_ind(high_popularity_rating,low_popularity_rating)


#2) Are movies that are newer rated differently than movies that are older? [Hint: Do a median split of year of
#release to contrast movies in terms of whether they are old or new]

df_movies_col_mean = df_data.iloc[:,0:400].apply(lambda col: col.fillna(col.mean()), axis=0)

movie_years = df_movies_col_mean.columns.str.extract(r'(\d{4})')[0].astype(int)
new_movies = movie_years[movie_years > movie_years.median()].index
old_movies = movie_years[movie_years <= movie_years.median()].index

new_movies_rating = df_movies_col_mean.iloc[:,new_movies].mean()
old_movies_rating = df_movies_col_mean.iloc[:,old_movies].mean()

t2, p2 = stats.ttest_ind(new_movies_rating,old_movies_rating)


#3) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?

df_movies_row_mean = df_data.iloc[:,0:400].apply(lambda row: row.fillna(row.mean()), axis=1)

male_rating_Shrek = df_movies_row_mean[df_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]['Shrek (2001)']
female_rating_Shrek = df_movies_row_mean[df_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]['Shrek (2001)']

t3, p3 = stats.ttest_ind(male_rating_Shrek, female_rating_Shrek)


#4) What proportion of movies are rated differently by male and female viewers?

df_male_rating = df_movies_row_mean[df_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]
df_female_rating = df_movies_row_mean[df_data['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]

results = [
    stats.ttest_ind(df_male_rating.iloc[:, i], df_female_rating.iloc[:, i])
    for i in range(df_male_rating.shape[1])
]
t4, p4 = zip(*results)
proportion4 = sum(1 for p in p4 if p <= 0.005) / df_male_rating.shape[1]


#5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?

OnlyChildren_rating_theLionKing = df_movies_row_mean[df_data['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] == 1]['The Lion King (1994)']
WithSiblings_rating_theLionKing = df_movies_row_mean[df_data['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] == 0]['The Lion King (1994)']

t5, p5 = stats.ttest_ind(OnlyChildren_rating_theLionKing, WithSiblings_rating_theLionKing)


#6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings
#vs. those without?

df_OnlyChildren_rating = df_movies_row_mean[df_data['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] == 1]
df_WithSiblings_rating = df_movies_row_mean[df_data['Are you an only child? (1: Yes; 0: No; -1: Did not respond)'] == 0]

results = [
    stats.ttest_ind(df_OnlyChildren_rating.iloc[:, i], df_WithSiblings_rating.iloc[:, i])
    for i in range(df_OnlyChildren_rating.shape[1])
]
t6, p6 = zip(*results)
proportion6 = sum(1 for p in p6 if p <= 0.005) / df_OnlyChildren_rating.shape[1]


#7) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who
#prefer to watch them alone?

socially_rating_WallStreet = df_movies_row_mean[df_data['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 0]['The Wolf of Wall Street (2013)']
alone_rating_WallStreet = df_movies_row_mean[df_data['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 1]['The Wolf of Wall Street (2013)']

t7, p7 = stats.ttest_ind(socially_rating_WallStreet, alone_rating_WallStreet)


#8) What proportion of movies exhibit such a “social watching” effect?

df_socially_rating = df_movies_row_mean[df_data['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 0]
df_alone_rating = df_movies_row_mean[df_data['Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'] == 1]

results = [
    stats.ttest_ind(df_socially_rating.iloc[:, i], df_alone_rating.iloc[:, i])
    for i in range(df_socially_rating.shape[1])
]
t8, p8 = zip(*results)
proportion8 = sum(1 for p in p8 if p <= 0.005) / df_socially_rating.shape[1]


#9) Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?

ks = stats.ks_2samp(df['Home Alone (1990)'], df['Finding Nemo (2003)'])

#10) There are ratings on movies from several franchises ([‘Star Wars’,‘Harry Potter’,‘The Matrix’,‘Indiana
#Jones’,‘Jurassic Park’,‘Pirates of the Caribbean’,‘Toy Story’,‘Batman’]) in this dataset. How many of these
#are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks
#featured in this question to identify the movies that are part of each franchise]

#starwar
sw1 = df['Star Wars: Episode 1 - The Phantom Menace (1999)']
sw2 = df['Star Wars: Episode II - Attack of the Clones (2002)']
sw3 = df['Star Wars: Episode IV - A New Hope (1977)']
sw4 = df['Star Wars: Episode V - The Empire Strikes Back (1980)']
sw5 = df['Star Wars: Episode VI - The Return of the Jedi (1983)']
sw6 = df['Star Wars: Episode VII - The Force Awakens (2015)']

f1, p1 = stats.f_oneway(sw1, sw2, sw3, sw4, sw5, sw6)
print(f1, p1)

#harrypotter
hp1 = df['Harry Potter and the Chamber of Secrets (2002)']
hp2 = df['Harry Potter and the Deathly Hallows: Part 2 (2011)']
hp3 = df['Harry Potter and the Goblet of Fire (2005)']
hp4 = df['Harry Potter and the Sorcerer\'s Stone (2001)']

f2, p2 = stats.f_oneway(hp1, hp2, hp3, hp4)
print(f2, p2)

#matrix
m1 = df['The Matrix (1999)']
m2 = df['The Matrix Reloaded (2003)']
m3 = df['The Matrix Revolutions (2003)']

f3, p3 = stats.f_oneway(m1, m2, m3)
print(f3, p3)

#indianajones
ij1 = df['Indiana Jones and the Kingdom of the Crystal Skull (2008)']
ij2 = df['Indiana Jones and the Last Crusade (1989)']
ij3 = df['Indiana Jones and the Raiders of the Lost Ark (1981)']
ij4 = df['Indiana Jones and the Temple of Doom (1984)']

f4, p4 = stats.f_oneway(ij1, ij2, ij3, ij4)
print(f4, p4)

#jurassicpark
jp1 = df['Jurassic Park (1993)']
jp2 = df['Jurassic Park III (2001)']
jp3 = df['The Lost World: Jurassic Park (1997)']

f5, p5 = stats.f_oneway(jp1, jp2, jp3)
print(f5, p5)

#pirates of the carribbean
pc1 = df['Pirates of the Caribbean: At World\'s End (2007)']
pc2 = df['Pirates of the Caribbean: Dead Man\'s Chest (2006)']
pc3 = df['Pirates of the Caribbean: The Curse of the Black Pearl (2003)']

f6, p6 = stats.f_oneway(pc1, pc2, pc3)
print(f6, p6)

#toystory
ts1 = df['Toy Story (1995)']
ts2 = df['Toy Story 2 (1999)']
ts3 = df['Toy Story 3 (2010)']

f7, p7 = stats.f_oneway(ts1, ts2, ts3)
print(f7, p7)

#batman
bm1 = df['Batman & Robin (1997)']
bm2 = df['Batman (1989)']
bm3 = df['Batman: The Dark Knight (2008)']

f8, p8 = stats.f_oneway(bm1, bm2, bm3)
print(f8, p8)
