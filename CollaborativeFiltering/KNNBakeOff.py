from MovieLens import MovieLens
from surprise import KNNBasic
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

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


def LoadMovieLensData():
    ml = MovieLens()
    print(BeginGREEN + "Loading movie ratings..." + EndGREEN)
    data = ml.loadMovieLensLatestSmall()
    print(BeginGREEN + "Computing movie popularity ranks so we can measure novelty later..." + EndGREEN)
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator 
evaluator = Evaluator(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, BeginBgBLUE + "User KNN" + EndBLUE)

# Item-based KNN
ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, BeginBgBLUE + "Item KNN" + EndBLUE)

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, BeginBgBLUE + "Random" + EndBLUE)

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
