
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
print(aaron_judge.columns)
print(aaron_judge.description.unique())
print(aaron_judge.type.unique())

def find_strike_zone(dataset):
    dataset['type'] = dataset.type.map({'S':1,'B':0})
    print(dataset.type.unique())

    print(dataset['plate_x'])

    dataset = dataset.dropna(subset=['plate_x','plate_z','type'])

    fig, ax = plt.subplots()
    plt.scatter(x = dataset['plate_x'],y = dataset['plate_z'], c=dataset['type'],cmap=plt.cm.coolwarm, alpha=0.25)
    training_set, validation_set = train_test_split(dataset, random_state = 1)

    largest ={'value': 0 ,'gamma':1,'C':1}
    for gamma in range(1,5):
      for C in range(1,5):
        classifier = SVC(kernel='rbf', gamma=gamma, C= C)
        classifier.fit(training_set[['plate_x', 'plate_z']],training_set['type'])

        score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
        if (score > largest['value']):
          largest['value']=score
          largest['gamma']=gamma
          largest['C']=C
        
    print(largest)

    draw_boundary(ax, classifier)
    plt.show()

find_strike_zone(david_ortiz)
