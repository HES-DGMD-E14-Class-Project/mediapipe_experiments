import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


data_dict = pickle.load(open('./data.pickle', 'rb'))
# print("data_dict variable",data_dict )


data = data_dict['data'] #np.array(data_dict['data'])
data2 = []

# Make sure all of the datasets have the same dimensions:

for index in range(len(data)):
    if len(data[index]) != 42:
        data[index] = data[index][:42]
    data2.append(data[index][:42])


for item in data2:
    if len(item) != 42:
        print(len(item))
        print(item)

data2 = np.array(data2)        

labels = np.array(data_dict['labels']) 

x_train, x_test, y_train, y_test = train_test_split(data2, labels, test_size=0.2, shuffle=True, stratify=labels)

"""
inverse_reg_strengths = [ 1 * (10 ** index) for index in range(-6, 10) ]
reg_strengths = [1/ inv for inv in inverse_reg_strengths]
accs = [0.4637964774951076, 0.4637964774951076,  0.45792563600782776, 0.49, 0.54, 0.5714285714285714, 0.68, 0.7, 0.81, 0.8571428571428571, 0.8671428571428571, 0.84, 0.87, 0.88, 0.9, 0.86, 0.85]


print(inverse_reg_strengths)
ax = plt.subplot()
ax.plot([index for index in range(len(accs))], accs, '-o')
ax.set_xticklabels(inverse_reg_strengths)

ax.set_title("Accuracies for various logistic regression models")

ax.set_xlabel("Regularization Strength")
ax.set_ylabel("Test Accuracy")
plt.show()
"""

# for strength in inverse_reg_strengths:
#     logit_model = LogisticRegression(C = strength, max_iter = 500).fit(x_train, y_train)
#     logit_acc = logit_model.score(x_test, y_test)
#     print(f"Our logistic model's score: {logit_acc}")
    
# logit_dict = {}
    

# """
# We will use a random forest classifier since we are dealing with a multi-class classification problem
best_accurracies = {index : 0 for index in range(0, 53, 4)}
best_depth = {index : 0 for index in range(1, 6)}
for num in range(0, 53, 4):
    curr_best_depth = 0
    curr_best_depth_acc = 0
    curr_best_model = None
    for depth in range(1, 6):
        model = RandomForestClassifier(n_estimators = num+1, max_depth = depth)
        model.fit(x_train, y_train) # Training
        y_predict = model.predict(x_test) # predict the test scores
        score = accuracy_score(y_predict, y_test) # Test on the test set
        if curr_best_depth_acc < score:
            curr_best_depth_acc = score
            curr_best_depth = depth
            curr_best_model = model
            
        #Print the score out
        print(f"Depth {depth} at num {num}")
        print('{}% of samples were classified correctly !'.format(score * 100))
    best_accurracies[num] = curr_best_depth_acc
    best_depth[num] = curr_best_depth

print(best_accurracies)
print(best_depth)


plt.plot(list(best_accurracies.keys()), list(best_accurracies.values()), '-o')
plt.xlabel("Number of Trees/Estimators")
plt.ylabel("Test Accuracy")
plt.title("Accuracies for Various Random Forest Models")
plt.show()


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
# """







