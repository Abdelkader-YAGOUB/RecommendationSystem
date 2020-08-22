from MovieLens import MovieLens
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics

# Color for text OUtPUT
BeginRED = '\033[91m'
EndRED = '\033[0m'

BeginBgRED = '\033[41m'
EndBgRED = '\033[0m'

BeginGREEN = '\033[92m'
EndGREEN = '\033[0m'

BeginBgGREEN ='\033[42m'
EndBgGREEN ='\033[0m'

BeginYELLO = '\033[93m'
EndYELLO = '\033[0m'

BeginBgYELLO = '\033[43m'
EndBgYELLO = '\033[0m'

BeginBLUE = '\033[94m'
EndBLUE = '\033[0m'

BeginBgBLUE = '\033[44m'
EndBgBLUE = '\033[0m'

ml = MovieLens()

print(BeginGREEN + "Loading movie ratings..." + EndGREEN)
data = ml.loadMovieLensLatestSmall()

print(BeginGREEN + "Computing movie popularity ranks so we can measure novelty later..." + EndGREEN)
rankings = ml.getPopularityRanks()

print(BeginGREEN + "Computing item similarities so we can measure diversity later..." + EndGREEN)
fullTrainSet = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

print(BeginGREEN + "Building recommendation model..." + EndGREEN)
trainSet, testSet = train_test_split(data, test_size=.25, random_state=1)

algo = SVD(random_state=10)
algo.fit(trainSet)

print(BeginGREEN +"Computing recommendations..." + EndGREEN)
predictions = algo.test(testSet)

print(BeginGREEN + "Evaluating accuracy of model...\n" + EndGREEN)
print(BeginBgBLUE + "# Root Mean Squared Error. Lower values mean better accuracy. #" + EndBgBLUE)
print(BeginBLUE + "RMSE: " + EndBLUE, BeginYELLO + "", RecommenderMetrics.RMSE(predictions) ,"" + EndYELLO)
print(BeginBgBLUE + "# Mean Absolute Error. Lower values mean better accuracy. #" + EndBgBLUE)
print(BeginBLUE + "MAE: " + EndBLUE, BeginYELLO + "", RecommenderMetrics.MAE(predictions),"" + EndYELLO)

print(BeginGREEN + "\nEvaluating top-10 recommendations..." + EndGREEN)

LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainSet, testSet in LOOCV.split(data):
    print(BeginGREEN + "Computing recommendations with leave-one-out..." + EndGREEN)

    # Train model without left-out ratings
    algo.fit(trainSet)

    # Predicts ratings for left-out ratings only
    print(BeginGREEN + "Predict ratings for left-out set..." + EndGREEN)
    leftOutPredictions = algo.test(testSet)

    # Build predictions for all ratings not in the training set
    print(BeginGREEN + "Predict all missing ratings..." + EndGREEN)
    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)

    # Compute top 10 recs for each user
    print(BeginGREEN + "Compute top 10 recs per user..." + EndGREEN)
    topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n=10)
    print("\n")
    # See how often we recommended a movie the user actually rated
    print(BeginBgBLUE + "# Hit Rate; how often we are able to recommend a left-out rating. Higher is better. #" + EndBgBLUE)
    print(BeginBLUE + "Hit Rate: " + EndBLUE, BeginYELLO +"", RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions),"" + EndYELLO)

    print("\n")
    # Break down hit rate by rating value
    print(BeginBgBLUE + "# rating Hit Rate #" + EndBgBLUE)
    print(BeginBLUE + "rHR (Hit Rate by Rating value): "  + EndBLUE)
    RecommenderMetrics.RatingHitRate(topNPredicted, leftOutPredictions)

    print("\n")
    # See how often we recommended a movie the user actually liked
    print(BeginBgBLUE + "# Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better. #" + EndBgBLUE)
    print(BeginBLUE +"cHR (Cumulative Hit Rate, rating >= 4): " + EndBLUE, BeginYELLO + "", RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions, 4.0),"" + EndYELLO )

    print("\n")
    # Compute ARHR
    print(BeginBgBLUE + "# Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better. #" + EndBgBLUE)
    print(BeginBLUE +"ARHR (Average Reciprocal Hit Rank): " + EndBLUE, BeginYELLO + "", RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions),"" + EndYELLO)

print(BeginGREEN + "\nComputing complete recommendations, no hold outs..." + EndGREEN)
algo.fit(fullTrainSet)
bigTestSet = fullTrainSet.build_anti_testset()
allPredictions = algo.test(bigTestSet)
topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n=10)

print("\n")
# Print user coverage with a minimum predicted rating of 4.0:
print(BeginBgBLUE + "# Ratio of users for whom recommendations above a certain threshold exist. Higher is better. #" + EndBgBLUE)
print(BeginBLUE + "User coverage: " + EndBLUE, BeginYELLO + "", RecommenderMetrics.UserCoverage(topNPredicted, fullTrainSet.n_users, ratingThreshold=4.0),"" + EndYELLO)

print("\n")
# Measure diversity of recommendations:
print(BeginBgBLUE + "# 1-S, where S is the average similarity score between every possible pair of recommendations " + EndBgBLUE)
print(BeginBgBLUE + "for a given user. Higher means more diverse. #" + EndBgBLUE)
print(BeginBLUE + "Diversity: " + EndBLUE, BeginYELLO + "", RecommenderMetrics.Diversity(topNPredicted, simsAlgo),"" + EndYELLO)

print("\n")
# Measure novelty (average popularity rank of recommendations):
print(BeginBgBLUE + "# Average popularity rank of recommended items. Higher means more novel. #" + EndBgBLUE)
print(BeginBLUE + "Novelty (average popularity rank): " + EndBLUE, BeginYELLO +  "",RecommenderMetrics.Novelty(topNPredicted, rankings),"" + EndYELLO)
