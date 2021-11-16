import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# split into input (X) and output (Y) variables
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
#print(X_train)
#print(Y_train)

def distance(one,two):
    return numpy.linalg.norm(one-two)

 
def get_cluster(predict):
    res = sum(predict)/len(predict)
    if res > 0.5:
        return 1
    else:
        return 0

# This function makes a prediction based on K neihbour(s).
# It calculates the min distance for each points, to identify its cluster.
def prediction(x,x_rest,y_rest,k):
    data=[]
    data_X=[]

    for i in range(len(x_rest)):
        data.append((distance(x,x_rest[i]), y_rest[i]))
        data_X.append(distance(x,x_rest[i]))

    k_predict=[]
    for i in range (k):
        ind = data_X.index(min(data_X))
        k_predict.append(data[ind][1])
        data.pop(ind)
        data_X.pop(ind)


    predict = get_cluster(k_predict)
    return predict


TP = 0
TN = 0
FP = 0
FN = 0
data = []
k = 9
for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    pred = prediction(x,X_train,Y_train,k)
    
    if(y==1 and pred ==1):
        TP += 1

    if(y==0 and pred ==0):
        TN += 1

    if(y==1 and pred ==0):
        FN += 1

    if(y==0 and pred ==1):
        FP += 1
print("K = ", k)
print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
print("Recall",TP/(TP+FN))
print("Precision",TP/(TP+FP))
print("F1",(2*TP)/(2*TP+FP+FN))




