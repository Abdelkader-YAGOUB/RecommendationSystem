from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

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

# User Id which we are testing on    
testSubject = '85'
k = 10

print(BeginBgBLUE + "User Collaborative Filtering for user Id : [" + testSubject + "]" + EndBgBLUE)

# Load our data set and compute the user similarity matrix
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
print("\n")
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(BeginBLUE + "" + ml.getMovieName(int(movieID)) + " | " + EndBLUE, ratingSum)
        pos += 1
        if (pos > 10):
            break

        



