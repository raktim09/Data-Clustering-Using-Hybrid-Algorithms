import  numpy  as  numpy
import  scipy  as  scipy
from  sklearn  import  cluster
import  matplotlib.pyplot  as  plt
from sklearn import metrics
import time
import random
def sequence_cluster(Output):
    l  =  list(set(Output))
    for i in range(len(l)):
        for k in range(len(Output)):
            if Output[k] == l[i]:
                Output[k] = i+1
    return Output            
     
def  set2List(NumpyArray):
    list  =  []
    for  item  in  NumpyArray:
        list.append(item.tolist())
    return  list   
  
def  member(value,MinumumPoints,Max_Points):
    if  value<=MinumumPoints:
        return  0
    elif  value<Max_Points:
        return  (value  -  MinumumPoints)/(Max_Points  -  MinumumPoints)
    else:
        return  1    
def Max_Distance(i,PointNeighbors,DistanceMatrix):
    maximum = 0
    for k in PointNeighbors:
        if DistanceMatrix[i][k] > maximum:
            maximum = DistanceMatrix[i][k]
    return maximum
            
def calculate_Membership(maximum,distance,Epsilon):
    # return ((2.718*(maximum*2))/(2.718(maximum*2)+2.718*(distance*2)))   #new1
    # return (1 - 2.718*(2*distance)/2.718*(2*maximum))                      #new2
    # return ((maximum*2 - distance2)/(maximum2 + distance*2))            #new3
    if Epsilon<0.5:
        return (maximum*2/(maximum2+distance*2))                           #new4
    else:
        return ((2.718*(maximum*2))/(2.718(maximum*2)+2.718*(distance*2))) #new1
def  DBSCAN(Dataset,Epsilon, Points,DistanceMethod  =  'euclidean'):
    m,n = Dataset.shape
    Type = numpy.zeros(m)
    Membership = []
    for i in range(m):
        Membership.append({})
#      -1  noise    outlier
#    0  border
#    1  core
    ClustersList=[]
    Cluster=[]
    PointClusterNumber=numpy.zeros(m)
    PointClusterNumberIndex=1
    PointNeighbors=[]
    DistanceMatrix  =  scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset,DistanceMethod))
    
    BorderPoint = []
    for  i  in  range(m):
        BorderPoint = []
        if  len(Membership[i]) == 0:
            PointNeighbors = numpy.where(DistanceMatrix[i]<=Epsilon)[0]
            PointNeighbors = set2List(PointNeighbors)

            Maximum = Max_Distance(i,PointNeighbors,DistanceMatrix)

            for k in PointNeighbors:
                if DistanceMatrix[i][k] == Maximum:        
                    BorderPoint.append(k)

            if len(PointNeighbors) >= Points:
                Membership[i][PointClusterNumberIndex] = 1
                Type[i] = 1
                ExpandCluster(DistanceMatrix,i,Epsilon,Points,PointNeighbors,BorderPoint,Membership,PointClusterNumberIndex,Type,Maximum)    
                PointClusterNumberIndex += 1
            else:
                Type[i] = -1
    # print(PointClusterNumberIndex)
    return Membership,Type,PointClusterNumberIndex-1

def ExpandCluster(DistanceMatrix,PointtoExpand,Epsilon,Points,PointNeighbors,BorderPoint,Membership,PointClusterNumberIndex,Type,Maximum):
    Neighbors = []
    for i in PointNeighbors:
        if i not in  BorderPoint:
            Membership[i][PointClusterNumberIndex] = round(calculate_Membership(Maximum,DistanceMatrix[i][PointtoExpand],Epsilon),3)
    for i in BorderPoint:
        for j in Neighbors:
            if (j not in BorderPoint) and (PointClusterNumberIndex not in Membership[j]):
                Membership[j][PointClusterNumberIndex] = round(calculate_Membership(Maximum,DistanceMatrix[j][PointtoExpand],Epsilon),3)
            elif j not in BorderPoint:
                Membership[j][PointClusterNumberIndex] = round(max(Membership[j][PointClusterNumberIndex],calculate_Membership(Maximum,DistanceMatrix[j][PointtoExpand],Epsilon)),3)

        Neighbors = numpy.where(DistanceMatrix[i]<=Epsilon)[0] 
        Neighbors = set2List(Neighbors)

        Maximum = Max_Distance(i,Neighbors,DistanceMatrix)

        for k in Neighbors:
            if DistanceMatrix[i][k] == Maximum and k != PointtoExpand:
                try:
                    BorderPoint.index(k)
                except ValueError:    
                    BorderPoint.append(k) 
        if len(Neighbors) >= Points:
            Membership[i][PointClusterNumberIndex] = 1
            Type[i] = 1
        else:
            Membership[i][PointClusterNumberIndex] = 0
            Type[i] = 0        
        PointtoExpand = i
    return

def  calculate_PC(Data_len,cluster_size,Membership_value):
    pc  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                pc  +=  Membership_value[i][j+1]**2
    return  pc/Data_len  

def  calculate_FPI(Data_len,cluster_size,Membership_value):
    value  =  0
    for  i  in  range(Data_len):
        for j in range(cluster_size):
            if (j+1) in Membership_value[i]:
                value  +=  (Membership_value[i][j+1]**2)/Data_len
    if cluster_size == 1:
        return 0
    else:
        FPI  =  1  -  (cluster_size/(cluster_size-1))*(1-value)
        return  FPI          

def  F_recall(Output,Membership_final,k):
    membership_sum = 0
    c = 0
    for i in  range(len(Membership_final)):
        if k in Membership_final[i] and Output[i] == k:
            membership_sum += Membership_final[i][k]
    for i in Output:
        if k == i:
            c += 1
    if membership_sum == 0:
        return 0,c    
    return membership_sum/c,c            

def  F_Precision(Membership_final,k):
    membership_sum = 0
    c = 0
    for i in  range(len(Membership_final)):
        if k in Membership_final[i] and Output[i] == k:
            membership_sum += Membership_final[i][k]
    for i in Membership_final:
        if k in i:
            c += 1
    if c == 0:
        return 0,c        
    return membership_sum/c,c

def Combine_clusters(Membership_value,size,Points):
    cluster_size = size
    m = 1
    l = 1
    for j in range(1,size+1):
        for k in range(1,j):
            count = 0
            flag = 0
            for i in Membership_value:
                if (j in i) and (k in i):
                    count += 1
                if count == Points:
                    flag = 1
                    break
            if flag == 1:
                cluster_size -= 1
                l = min(j,k)
                for i in Membership_value:
                    if (j in i) and (k in i):
                        i[l] = max(i[j],i[k])
                        if l!=k:    
                            del i[k]
                        if l!=j:    
                            del i[j]
                    elif k in i:
                        i[l] = i[k]
                        if l!=k:    
                            del i[k]
                    elif j in i:
                        i[l] = i[j]
                        if l!=j:    
                            del i[j]                  

    return Membership_value,cluster_size                            
def membership_set(Membership,r1):
    set_value = []
    for i in range(len(Membership)):
        for j in range(r1+1):
            if j in Membership[i]:
                set_value.append(j)
    return set_value            

def mapping_cluster(Membership,set_result,Output,set_output):
    count = {}
    Result = []
    c =0
    for i in range(len(Membership)):
        Result.append(0)
    for i in set_result:
        # for m in set_output:
        #         count[m] = 0
        max1 = 0
        pos = 0     
        for j in set_output:
            c = 0
            for k in range(len(Membership)):
                if i in Membership[k] and j == Output[k]:
                    c += 1
            if c>max1:
                max1 = c
                pos = j
        for j in range(len(Membership)):
            if i in Membership[j]:
                Result[j] = pos
                Membership[j][pos] = Membership[j][i]
                if pos != i:
                    del Membership[j][i]
    return Membership,Result      

def accuracy(Membership,Output):
    count = 0
    for i in range(len(Output)):
        if Output[i] in Membership[i]:
            count+=1
    return count/len(Output)
def Plot_cluster(Membership,Data,Output,file_name):
    m = max(Output)
    color = {1:'red',2:'blue',3:'yellow',4:'brown',5:'green',6:'orange',7:'cyan',8:'olive',9:'gray',10:'pink',11:'lime',12:'blueviolet',13:'gold',14:'aqua',15:'maroon',16:'wheat',17:'peru',18:'darkkhaki',19:'red',20:'blue',21:'yellow',22:'brown',23:'green',24:'orange',25:'cyan',26:'olive',27:'gray',28:'pink',29:'lime',30:'blueviolet',31:'gold'} 
    plt.style.use('ggplot')
    for i in range(len(Output)):
        if Output[i] in Membership[i]:
            plt.scatter(Data[i][0],Data[i][1],color = color[Output[i]])
        else:
            plt.scatter(Data[i][0],Data[i][1],color ='black')
    plt.title(file_name)
    plt.show()


file_name = "joe"
X  =  numpy.loadtxt(file_name+".txt",dtype  =  'float')
Output  =  len(X)*[0]
for i in range(len(X)):
  if i<1000:
    Output[i]=1
  elif i<2000:
    Output[i]=2
  elif i<3000:
    Output[i]=3
  elif i<4000:
    Output[i]=4
  elif i<5000:
    Output[i]=5
  else:
    Output[i]=6            
Target = sequence_cluster(Output)
Epsilon = 0.2
Points = 100
t0 = time.time()
Membership_value,Type,cluster_len=DBSCAN(X,Epsilon,Points)
Membership,c_len1 = Combine_clusters(Membership_value,cluster_len,Points) 

Membership,c_len2 = Combine_clusters(Membership,cluster_len,Points)
t1 = time.time()

set_value = membership_set(Membership,cluster_len)  
Membership_final,Result = mapping_cluster(Membership,set_value,Output,list(set(Output)))        

# for i in range(len(Membership_final)):
#     print(Membership_final[i],Output[i],"\n")

PC = calculate_PC(len(X),cluster_len,Membership_final)
FPI = calculate_FPI(len(X),cluster_len,Membership_final)

F_measure  =  0
l  =  list(set(Output))
for i in l:
    try:
        f_Precision,Cj = F_Precision(Membership_final,i)
        f_recall,Dc = F_recall(Output,Membership_final,i)
        f_measure = 2*(f_Precision*f_recall)/(f_recall+f_Precision)
        F_measure += (Cj/len(X))*f_measure
    except:
        F_measure +=0
Accuracy =  accuracy(Membership,Output)       
k = list(set(Output))
print("Time_complexity:  ",t1-t0)
print("Silhoutte :- ")
label=[]
for i in Membership:
  if(len(i)!=1):
    label.append(0)
  else:
    for key,value in i.items():
      label.append(key)
print(metrics.silhouette_score(X,label))
Plot_cluster(Membership,X,Output,file_name)
