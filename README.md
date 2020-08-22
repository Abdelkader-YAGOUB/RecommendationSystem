# Recommendation System

## Introduction
During the last few decades, with the rise of Youtube, Amazon, Netflix and many other such web services, recommender systems have taken more and more place in our lives. From e-commerce (suggest to buyers articles that could interest them) to online advertisement (suggest to users the right contents, matching their preferences), recommender systems are today unavoidable in our daily online journeys.

The basic premise of recommendations is that there are significant dependencies between user-centric and item-centric activity. For example, a user who is interested in a historical documentary is more likely to be interested in another historical documentary or educational program rather than an action film. Alternatively, dependencies can be present at the finer granularity of individual items rather than categories. These dependencies can be learned in a data-driven fashion from the scoring matrix and the resulting model is used to make predictions for target users. The more rated items available to a user, the easier it is to make reliable predictions about future user behavior. In practice, recommendation systems can be more complex and richer in data, with a wide variety of auxiliary data types.

In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).

## Evaluating
A big part of why recommender systems are as much art as they are science is that it’s difficult to measure how good they are. There is a certain aesthetic quality to the results they give you, and it’s hard to say whether a person considers a recommendation to be good or not especially if you’re developing algorithms offline.

People have come up with a lot of different ways to measure the quality of a recommender system, and often different measurements can be at odds with each other. 

Through this project, we review various methods of evaluating recommendation systems using a wonderful framework called SurpriseLib.

## Recommendation system techniques ⚙️
We are going to simple explanation the major paradigms of recommender systems : content-based, collaborative and Hybrid methode.

##### Content-Based Filtering :
The most simple approach recommending items just based on the attributes of those items themselves, instead of trying to use aggregate user behavior data. For example, it can be effective to just recommend movies in the same genre as movies we know someone enjoys. Even for more advanced recommender systems that are based on machine learning, baking in some knowledge about the content itself can make them even better.

##### Neighborhood-Based Collaborative Filtering :
This is the idea of leveraging the behavior of others to inform what you might enjoy. At a very high level, it means finding other people like you and recommending stuff they liked. Or, it might mean finding other things similar to the things you like that is, recommending stuff people bought who also bought the stuff you liked. Either way, the idea is taking cues from people like your “neighborhood” if you will and recommending stuff based on things they liked that you haven’t tried yet. That’s why we call it “collaborative” filtering it’s recommending stuff based on other peoples’ collaborative behavior.

##### Hybrid Recommenders :
There’s no need to choose a single algorithm for your recommender system. Each algorithm has its own strengths and weaknesses, and you may end up with a better system by combining many algorithms together in what we call a hybrid approach.

## Getting Set Up & Run 🚀

##### Tools :
- Environment for Python 3 installed, you’ll need to get one. I recommend Anaconda  it’s free and widely used.  Download Link : https://anaconda.com/download
- We need to install a Python package that makes developing recommender systems easier, called “Surprise”. open up a terminal and run : 
``` conda install –c conda-forge scikit-surprise ```

##### DataSet :
- We’re going to build up a large project that recommends movies in many different ways. So we’re going to need data to work with.
MovieLens data set. It’s a subset of 100,000 real movie ratings from real people, along with some information about the movies themselves. Included with the project, Also, there is a way to download it through the following link : https://grouplens.org/datasets/movielens/

## Other
- Surprise’ documentation : https://surprise.readthedocs.io/en/stable/
- Surprise GitHub : https://github.com/NicolasHug/Surprise

## Thanks for reading! 🙏 

![alt text](https://yagoub.net/github_img/RecommendationSystemEnd.png)
