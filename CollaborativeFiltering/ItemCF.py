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

print(BeginBgBLUE + "item Collaborative Filtering for user Id : [" + testSubject + "]" + EndBgBLUE)

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K items we rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)
    
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

    


