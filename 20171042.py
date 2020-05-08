#!/usr/bin/env python
# coding: utf-8


# 
# ## Datasets 
# - 3 datasets are provided. Load the data from the drive [link](!https://drive.google.com/file/d/1ujsKv9W5eidb4TXt1pnsqwDKVDFtzZTh/view?usp=sharing).

# In[3]:


# Installing Libraries
get_ipython().system('pip install scikit-learn matplotlib Pillow scipy')


# In[1]:


# Basic Imports
import os
import sys
import warnings
import numpy as  np
import pandas as pd
from scipy import linalg

import warnings
warnings.filterwarnings('ignore')
# Loading and plotting data
from PIL import Image
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Features
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import _class_means,_class_cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn import svm

plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Parameters
# - Image size: Bigger images create better representation but would require more computation. Choose the correct image size based on your Laptop configuration. 
# - is_grayscale: Should you take grayscale images? Or rgb images? Choose whichever gives better representation for classification. 

# In[2]:


opt = {
    'image_size': 32,
    'is_grayscale': False,
    'val_split': 0.75
}


# ### Load Dataset

# In[3]:


cfw_dict = {'Amitabhbachan': 0,
    'AamirKhan': 1,
    'DwayneJohnson': 2,
    'AishwaryaRai': 3,
    'BarackObama': 4,
    'NarendraModi': 5,
    'ManmohanSingh': 6,
    'VladimirPutin': 7}

imfdb_dict = {'MadhuriDixit': 0,
     'Kajol': 1,
     'SharukhKhan': 2,
     'ShilpaShetty': 3,
     'AmitabhBachan': 4,
     'KatrinaKaif': 5,
     'AkshayKumar': 6,
     'Amir': 7}

# Load Image using PIL for dataset
def load_image(path):
    im = Image.open(path).convert('L' if opt['is_grayscale'] else 'RGB')
    im = im.resize((opt['image_size'],opt['image_size']))
    im = np.array(im)
    im = im/256
    return im

# Load the full data from directory
def load_data(dir_path):
    image_list = []
    y_list = []
    
    if "CFW" in dir_path:
        label_dict = cfw_dict

    elif "yale" in dir_path.lower():
        label_dict = {}
        for i in range(15):
            label_dict[str(i+1)] = i
    elif "IMFDB" in dir_path:
        label_dict = imfdb_dict
    else:
        raise KeyError("Dataset not found.")
    
    
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".png"):
            im = load_image(os.path.join(dir_path,filename))
            y = filename.split('_')[0]
            y = label_dict[y] 
            image_list.append(im)
            y_list.append(y)
        else:
            continue

    image_list = np.array(image_list)
    y_list = np.array(y_list)

    print("Dataset shape:",image_list.shape)

    return image_list,y_list

# Display N Images in a nice format
def disply_images(imgs,classes,row=1,col=2,w=64,h=64):
    fig=plt.figure(figsize=(8, 8))
    for i in range(1, col*row +1):
        img = imgs[i-1]
        fig.add_subplot(row, col, i)
        
        if opt['is_grayscale']:
            plt.imshow(img , cmap='gray') 
        else:
            plt.imshow(img)
        
        plt.title("Class:{}".format(classes[i-1]))
        plt.axis('off')
    plt.show()


# In[4]:


# Loading the dataset
# eg.
dirpath = './dataset/IMFDB/'
X,y = load_data(dirpath)
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
#print(N,H,W,X)


# In[5]:


# Show sample images
ind = np.random.randint(0,y.shape[0],6)
disply_images(X[ind,...],y[ind], row=2,col=3)


# # Features
#     You are provided 6 Features. These features are:
#    - Eigen Faces / PCA 
#    - Kernel PCA
#    - Fisher Face / LDA
#    - Kernel Fisher Face
#    - VGG Features 
#    - Resnet Features
# 
# **VGG and Resnet features are last layer features learned by training a model for image classification**
#     
# ---
# 

# In[6]:


# Flatten to apply PCA/LDA
X = X.reshape((N,H*W*C))


# ###  1. Eigen Face:
# Use principal component analysis to get the eigen faces. 
# Go through the [documentation](!http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) on how to use it

# In[7]:


def get_pca(X,k):
    """
        Get PCA of K dimension using the top eigen vectors 
    """
    pca = PCA(n_components=k)
    X_k = pca.fit_transform(X)
    return X_k,pca


# ###  2. Kernel Face:


# In[8]:


def get_kernel_pca(X, k,kernel='rbf', degree=3):
    """
        Get PCA of K dimension using the top eigen vectors 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
        @param: kernel => which kernel to use (“linear” | “poly” | “rbf” | “sigmoid” | “cosine” )
        @param: d => Degree for poly kernels. Ignored by other kernels
    """
    kpca = KernelPCA(n_components=k,kernel=kernel,degree=degree)
    X_k = kpca.fit_transform(X)
    return X_k


# ### 3. Fisher Face


# In[9]:


def get_lda(X_train,X_test,y, k):
    """
        Get LDA of K dimension 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
    """
    lda = LDA(n_components=k).fit(X_train,y)
    X_tr = lda.transform(X_train)
    X_te = lda.transform(X_test)
    return X_tr,X_te


# ### 4. Kernel Fisher Face
# Use LDA using different kernels similiar to KernelPCA. Here the input is directly transformed instead of using the kernel trick.  

# In[10]:


def get_kernel_lda(X_train,X_test,y,k,kernel='rbf',degree=3):
    """
        Get LDA of K dimension 
        @param: X => Your data flattened to D dimension
        @param: k => Number of components
        @param: kernel => which kernel to use ( “poly” | “rbf” | “sigmoid”)
    """
    # Transform  input
    if kernel == "poly":
        X_train_transformed = X_train**degree
        X_test_transformed = X_test**degree
    elif kernel == "rbf":
        var = np.var(X_train)
        var1 = np.var(X_test)
        X_train_transformed= np.exp(-X_train/(2*var))
        X_test_transformed= np.exp(-X_test/(2*var1))
    elif kernel == "sigmoid":
        X_train_transformed = np.tanh(X_train)
        X_test_transformed = np.tanh(X_test)
    else: 
        raise NotImplementedError("Kernel {} Not defined".format(kernel))
        
    klda = LDA(n_components=k).fit(X_train_transformed,y)
    X_tr = klda.transform(X_train_transformed)
    X_te = klda.transform(X_test_transformed)
    return X_tr,X_te


# ### 5. VGG Features
# VGG Neural Networks a 19 layer CNN architecture introduced by Andrew Zisserman([Link](!https://arxiv.org/pdf/1409.1556.pdf) to paper). We are providing you with the last fully connected layer of this model.
# 
# The model was trained for face classification on each dataset and each feature the dimension of 4096.

# In[11]:


def get_vgg_features(dirpath):
    features = np.load(os.path.join(dirpath,"VGG19_features.npy"))
    return features


# ### 6. Resnet Features
# 
# [Residual neural networks](!https://arxiv.org/pdf/1512.03385.pdf) are CNN with large depth, to effectively train these netwrorks they utilize skip connections, or short-cuts to jump over some layers. This helps solving [vanishing gradient problem](!https://en.wikipedia.org/wiki/Vanishing_gradient_problem) 
# 
# A 50 layer resnet model was trained for face classification on each dataset. Each feature the dimension of 2048

# In[12]:


def get_resnet_features(dirpath):
    features = np.load(os.path.join(dirpath,"resnet50_features.npy"))
    return features



# In[13]:


# Compute your features 
# eg.

X_3D = get_pca(X.T,0.95)


# In[14]:


#Create a scatter plot  
# #eg.
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
hi,_ = get_pca(X,0.95)
ax.scatter(hi[:,0],hi[:,1],hi[:,2],c=y)
plt.show()
    


# In[15]:



def eigen_spectrum(path,a):
    X,y = load_data(path)
    N,H,W = X.shape[0:3]
    C = 1 if opt['is_grayscale'] else X.shape[3]
    X = X.reshape((N,H*W*C))
    
    #calculating eigen values and eigen vec
    eig_val,eig_vec = np.linalg.eig(np.cov(X))
    size = eig_val.shape
    #print(size)
    sum_eigval = np.sum(eig_val)
    #print(eig_val,sum_eigval)
    sum = 0
    for i in range(0,size[0]):
        sum = sum + eig_val[i]
        if sum/sum_eigval >= 0.95:
            break
    if path == './dataset/IMFDB/':
        if a == 1:
            plt.title("eigen spectrum for IMFDB data set")
    
    if path == './dataset/IIIT-CFW/' :
        if a==1:
            plt.title("eigen spectrum for IIIT-CFW data set")
        
    if path == './dataset/Yale_face_database/':
        if a==1:
            plt.title("eigen spectrum for Yale_face dataset")
    if a==1:
        plt.stem(eig_val[0:100]/sum_eigval)
        plt.show()
    return i


# In[16]:


# Plot the eigen value spectrum 
ans = eigen_spectrum('./dataset/IMFDB/',1)
print("minimum number of eigen values required for IMFDB data set is - ",ans)

ans = eigen_spectrum('./dataset/IIIT-CFW/',1)
print("minimum number of eigen values required for IIIT-CFW data set is - ",ans)

ans = eigen_spectrum('./dataset/Yale_face_database/',1)
print("minimum number of eigen values required for Yale_face_database data set is - ",ans)

    


# 1(c). Reconstruct  the  image  back for each case
# 

# In[17]:


def reconstruct_images(path,j):
    """
        
        @params: 
                Input parameters

        @return reconstructed_X => reconstructed image
        
    """
    X,y = load_data(path)
    N,H,W = X.shape[0:3]
    C = 1 if opt['is_grayscale'] else X.shape[3]
    X = X.reshape((N,H*W*C))
    if j==0:
        required_eigenvalues = eigen_spectrum(path,0)
    #temp = get_pca(X.T, required_eigenvalues)
        pca = PCA(n_components = required_eigenvalues)
        temp = pca.fit_transform(X.T)
        reconstruct_X = pca.inverse_transform(temp).T
        return reconstruct_X
    else:
        pca = PCA(n_components = j)
        temp = pca.fit_transform(X.T)
        reconstruct_X = pca.inverse_transform(temp).T
        return reconstruct_X

    
    pass
    reconstruct_X = None
    
    return reconstruct_X    


# In[18]:


# Display results 
X_reconstruced = reconstruct_images('./dataset/IMFDB/',0)

X_reconstruced = X_reconstruced.reshape(-1, 32, 32, 3)
# Display random images
X,y = load_data('./dataset/IMFDB/')

ind = np.random.randint(0,y.shape[0],6)

disply_images(X_reconstruced[ind,...],y[ind],row=2,col=3)

# Show the reconstruction error

#print(X.shape)

print("error for IMFDB-",np.sqrt(np.mean((X - X_reconstruced)**2)))






X1_reconstruced = reconstruct_images('./dataset/IIIT-CFW/',0)

X1_reconstruced = X1_reconstruced.reshape(-1, 32, 32, 3)
# Display random images
X1,y1 = load_data('./dataset/IIIT-CFW/')

ind = np.random.randint(0,y1.shape[0],6)

disply_images(X1_reconstruced[ind,...],y1[ind],row=2,col=3)

# Show the reconstruction error

#print(X.shape)

print("error for IIIT-CFW-",np.sqrt(np.mean((X1 - X1_reconstruced)**2)))






X2_reconstruced = reconstruct_images('./dataset/Yale_face_database/',0)

X2_reconstruced = X2_reconstruced.reshape(-1, 32, 32, 3)
# Display random images
X2,y2 = load_data('./dataset/Yale_face_database/')

ind = np.random.randint(0,y2.shape[0],6)

disply_images(X2_reconstruced[ind,...],y2[ind],row=2,col=3)

# Show the reconstruction error


print("error for Yale_face_database-",np.sqrt(np.mean((X2 - X2_reconstruced)**2)))


# In[19]:


# code goes here
X,y = load_data('./dataset/IMFDB/')
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
X = X.reshape((N,H*W*C))
print("data set taken is-","IMFDB")
k_values = [9,20,102]
max = np.max(y)

for j in k_values:
    print("value of K taken-",j)
    classes = []
    for i in range(max+1):
        itr = 0
        itr1 = 0
        for cl in y:
            if cl == i:
                if itr == 0:
                    start = itr1
                    itr = itr+1
                else:
                    end = itr1
            itr1= itr1+1
        X_reconstructed = reconstruct_images('./dataset/IMFDB/',j)
        classes.append(np.mean((X[start:end,:] - X_reconstructed[start:end,:])**2))
    print(classes)
    order = np.argsort(classes)
    #print(order)
    
    for a in imfdb_dict.keys():
        if order[max] == imfdb_dict[a] :
            required = a
    print("so the image class with maximum error-",order[max],"person is-",required)


# In[20]:


# code goes here
X,y = load_data('./dataset/IIIT-CFW/')
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
X = X.reshape((N,H*W*C))
print("data set taken is-","IIIT-CFW")
k_values = [9,20,102]

max = np.max(y)

for j in k_values:
    print("value of K taken-",j)
    classes = []
    for i in range(max+1):
        itr = 0
        itr1 = 0
        for cl in y:
            if cl == i:
                if itr == 0:
                    start = itr1
                    itr = itr+1
                else:
                    end = itr1
            itr1= itr1+1
        X_reconstructed = reconstruct_images('./dataset/IIIT-CFW/',j)
        classes.append(np.mean((X[start:end,:] - X_reconstructed[start:end,:])**2))
    order = np.argsort(classes)
    #print(order)
    for a in cfw_dict.keys():
        if order[max] == cfw_dict[a] :
            required = a
    print("so the image class with maximum error-",order[max],"person is-",required)


# In[21]:


X,y = load_data('./dataset/Yale_face_database/')
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
X = X.reshape((N,H*W*C))
print("data set taken is-",'Yale_face_database')
k_values = [9,20,102]
max = np.max(y)

for j in k_values:
    print("value of K taken-",j)
    classes = []
    for i in range(max+1):
        itr = 0
        itr1 = 0
        for cl in y:
            if cl == i:
                if itr == 0:
                    start = itr1
                    itr = itr+1
                else:
                    end = itr1
            itr1= itr1+1
        X_reconstructed = reconstruct_images('./dataset/Yale_face_database/',j)
        classes.append(np.mean((X[start:end,:] - X_reconstructed[start:end,:])**2))
    order = np.argsort(classes)
    print(order)
    print("so the image class with maximum error-",order[max])


# In[22]:


# Define your classifier here. You can use libraries like sklearn to create your classifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix,classification_report,f1_score,recall_score,precision_score
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



class Classifier():
    def __init__(self,classifier):
        #super.__init__()
        if classifier == "Decision Trees":
            self.classify =  DecisionTreeClassifier()
        elif classifier == "SVM":
            self.classify = SVC(kernel='linear',C=0.25)
        elif classifier == "Logistic Regression":
            self.classify =  LogisticRegression()
        else:
            self.classify = MLPClassifier(hidden_layer_sizes=(256,128), batch_size=256,early_stopping = True)
        
    
    # Define your parameters eg, W,b, max_iterations etc. 
    
    def classify1(self,X):
        """
            Given an input X classify it into appropriate class. 
        """
        prediction = self.classify.predict(X)
        
        return prediction
        
    def confusion_matrix(self,pred,y):
        """
            A confusion matrix is a table that is often used to describe the performance of a classification
            model (or “classifier”) on a set of test data for which the true values are known.
            
            
            @return confusion_matrix => num_classesxnum_classes martix 
                where confusion_matrix[i,j] = number of prediction which are i and number of ground truth value equal j 
        
        """
        return confusion_matrix(pred,y)
        
    def train(self,X_train,y_train):
        """
            
            @param X_train => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your classifier will be trained. 
                                It can be any combination of features provided above.

            @param y_train => N vector. Ground truth label 
    
            @return Nothing
        """
        return self.classify.fit(X_train,y_train)
        
    def validate(self,X_validate,y_validate):
        """
            
            @param X_validate => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your classifier validated. 
                                It can be any combination of features provided above.

            @param y_validate => N vector. Ground truth label 
            
        """
        y_pred = self.classify1(X_validate)
        
        length = y_pred.shape
        count = 0
        for i in range(0,length[0]):
            if y_pred[i] != y_validate[i]:
                count +=1
        


    
    
        return self.confusion_matrix(y_validate,y_pred),accuracy_score(y_validate,y_pred),f1_score(y_validate,y_pred,average="macro"),count/length[0]


# In[23]:

def train_test_split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2) 
    return X_train,X_test,y_train,y_test


# In[24]:



import pandas

def table(path,a):
    X,y = load_data(path)
    N,H,W = X.shape[0:3]
    C = 1 if opt['is_grayscale'] else X.shape[3]
    X = X.reshape((N,H*W*C))
    
    min_eigenvalues = eigen_spectrum(path,0)
    
    
    # dictionaries for accuracy, f1-score,reduced space
    
    accuracy = {}
    f1_score = {}
    reduced_space = {}
    error = {}
    n = {}
    confu_matrix = {}
    features = {"pca":get_pca(X,32),"kernel_pca":get_kernel_pca(X,32,kernel='rbf', degree=4),
                 "lda":get_lda(X,X,y,5),"kernel_lda":get_kernel_lda(X,X,y,5,kernel='rbf',degree=4),"vgg":get_vgg_features(path),
                "resnet":get_resnet_features(path),"lda+pca+kernelpca+kernellda":get_resnet_features(path),"klda+kpca":get_resnet_features(path)
               }
    all_classifiers = ["MLP","SVM","Decision Trees","Logistic Regression"]
    
    for i in features:
        temp_accuracy = 0
        temp_f1_score = 0
        
        best_accuracy = 0
        best_f1_score = 0
        #print(i)
        
        if i == "pca":
            new_x,_ = features[i]
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)

        elif i=="lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
           # print(X_train.shape,X_test.shape)
            
            X_train,X_test = get_lda(X_train,X_test,y_train,5)
            new_x = X_train
    
        elif i =="kernel_lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_train,X_test = get_kernel_lda(X_train,X_test,y_train,5)
            new_x = X_train
            
        
        elif i == "lda+pca+kernelpca+kernellda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_lda_train,X_lda_test = get_lda(X_train,X_test,y_train,7)
            X_pca_train,_ = get_pca(X_train,32)
            X_pca_test,_ = get_pca(X_test,32)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_lda_train,X_pca_train,X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_lda_test,X_pca_test,X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
            
        elif i=="klda+kpca":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
            

        
        else:
            new_x = features[i] 
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)

            
        # getting train and test data by calling function
        
        
        for cls in all_classifiers:
            res = Classifier(cls)
            res.train(X_train,y_train)
            temp_confusion,temp_accuracy,temp_f1_score,temp_error = res.validate(X_test,y_test)
            if best_accuracy < temp_accuracy:
                best_confusion,best_accuracy,best_f1_score,best_error = temp_confusion,temp_accuracy,temp_f1_score,temp_error
                name = cls
           # print(cls,"-",temp_accuracy)
        #print(best_accuracy)
        accuracy[i],f1_score[i],reduced_space[i],n[i],error[i],confu_matrix[i]=best_accuracy*100,best_f1_score*100,new_x.shape[1],name,best_error*100,best_confusion
        
    table = {'Reduced Space':reduced_space,'Accuracy':accuracy,'F1_score':f1_score,'classifier':n,'error':error}

    if a==1:
        return table
    else:
        return confu_matrix,n,accuracy
    
    
print("data-IMFDB")
pd.DataFrame(table('./dataset/IMFDB/',1))
      
        


# In[25]:


print("data-IIIT-CFW")

pd.DataFrame(table('./dataset/IIIT-CFW/',1))


# In[26]:


print("data-Yale_face")

pd.DataFrame(table('./dataset/Yale_face_database/',1))


# In[27]:


# For each dataset confusion matrix for the best model 
matrix,name,l = table('./dataset/IMFDB/',0)
#print(type(accuracy))
#print(np.argmax(np.array(accuracy)))
import operator
max = 0

for i in l.keys():
    if max < l[i]:
        required = i
print(required)

print(matrix[required])


# In[28]:


matrix,name,l = table("./dataset/Yale_face_database/",0)
#print(type(accuracy))
#print(np.argmax(np.array(accuracy)))
import operator
max = 0

for i in l.keys():
    if max < l[i]:
        required = i
print(required)

print(matrix[required])


# In[29]:


matrix,name,l = table('./dataset/IIIT-CFW/',0)
#print(type(accuracy))
#print(np.argmax(np.array(accuracy)))
import operator
max = 0

for i in l.keys():
    if max < l[i]:
        required = i
print(required)

print(matrix[required])



# In[30]:


# Compute TSNE for different features and create a scatter plot

X,y= load_data("./dataset/Yale_face_database/")  # feature 
N,H,W = X.shape[0:3]
X = X.reshape(N,H*W*C)
k = 3
TSNE_X = TSNE(k).fit_transform(X)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TSNE_X[:,0],TSNE_X[:,1],TSNE_X[:,2],c=y)
plt.show()
plt.scatter(TSNE_X[:,0],TSNE_X[:,1],c=y)








# In[31]:


# Compute TSNE for different features and create a scatter plot

X,y= load_data('./dataset/IIIT-CFW/')  # feature 
N,H,W = X.shape[0:3]
X = X.reshape(N,H*W*C)
k=3
TSNE_X = TSNE(k).fit_transform(X)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TSNE_X[:,0],TSNE_X[:,1],TSNE_X[:,2],c=y)
plt.show()
plt.scatter(TSNE_X[:,0],TSNE_X[:,1],c=y)


# In[32]:



# Compute TSNE for different features and create a scatter plot

X,y= load_data('./dataset/IMFDB/')  # feature 
N,H,W = X.shape[0:3]
X = X.reshape(N,H*W*C)
k=3
TSNE_X = TSNE(k).fit_transform(X)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TSNE_X[:,0],TSNE_X[:,1],TSNE_X[:,2],c=y)
plt.show()
plt.scatter(TSNE_X[:,0],TSNE_X[:,1],c=y)


# In[33]:


X,y= load_data('./dataset/IMFDB/')  # feature 
N,H,W = X.shape[0:3]
X = X.reshape(N,H*W*C)


X1,y1= load_data('./dataset/IIIT-CFW/')  # feature 
N1,H1,W1 = X1.shape[0:3]
C1 = 1 if opt['is_grayscale'] else X1.shape[3]

X1 = X1.reshape(N1,H1*W1*C1)


X2,y2= load_data("./dataset/Yale_face_database/")  # feature 
N2,H2,W2 = X2.shape[0:3]
C2 = 1 if opt['is_grayscale'] else X2.shape[3]

X2 = X2.reshape(N2,H2*W2*C2)

y = [0]*400
y1 = [1]*672
y2 = [2]*165

new_X,new_Y = np.concatenate((X,X1,X2),axis=0),np.concatenate((y,y1,y2),axis=0)

k=3
TSNE_X = TSNE(k).fit_transform(new_X)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TSNE_X[:,0],TSNE_X[:,1],TSNE_X[:,2],c=new_Y)
plt.show()
plt.scatter(TSNE_X[:,0],TSNE_X[:,1],c=new_Y)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier

class FaceVerification():
    def __init__(self,classifier):
        #super.__init__()
        if classifier == "KNN":
            self.classify = KNeighborsClassifier(n_neighbors=3)

    
    # Define your parameters eg, W,b, max_iterations etc. 

    
    def verify(self,X,class_id):
        """
            Given an input X find if the class id is correct or not.
            
            @return verfication_results => N vector containing True or False. 
                    If the class-id matches with your prediction then true else false.   
        """
        verification_results = (self.classify.predict(X)==class_id)
        return verfication_results
        
    def train(self,X_train,y_train):
        """
            Given your training data, learn the parameters of your classifier
            
            @param X_train => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your verification system will be trained. 
                                It can be any combination of features provided above.

            @param y_train => N vector. Ground truth label 
    
            @return Nothing
        """
        return self.classify.fit(X_train,y_train)
        
    def validate(self,X_validate,y_validate):
        """
            How good is your system on unseen data? Use the function below to calculate different metrics. 
            Based on these matrix change the hyperparmeters
            
            @param X_validate => NxD tensor. Where N is the number of samples and D is the dimension. 
                                It can be any combination of features provided above.

            @param y_validate => N vector. Ground truth label 
            
        """
        y_pred = self.classify.predict(X_validate)  
        return accuracy_score(y_validate,y_pred),f1_score(y_validate,y_pred, average = "macro"),recall_score(y_validate, y_pred, average="macro"),precision_score(y_validate,y_pred,average="macro")
        # Calculate F1-score
    


# In[35]:


# Create a train and validation split and show your results 
def train_test_split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2) 
    return X_train,X_test,y_train,y_test


# In[36]:


from sklearn.neighbors import KNeighborsClassifier

class FaceVerification1():
    def __init__(self,classifier):
        #super.__init__()
        if classifier == "KNN":
            self.classify = KNeighborsClassifier(n_neighbors=5)

    
    # Define your parameters eg, W,b, max_iterations etc. 

    
    def verify(self,X,class_id):
        """
            Given an input X find if the class id is correct or not.
            
            @return verfication_results => N vector containing True or False. 
                    If the class-id matches with your prediction then true else false.   
        """
        verification_results = (self.classify.predict(X)==class_id)
        return verfication_results
        
    def train(self,X_train,y_train):
        """
            Given your training data, learn the parameters of your classifier
            
            @param X_train => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your verification system will be trained. 
                                It can be any combination of features provided above.

            @param y_train => N vector. Ground truth label 
    
            @return Nothing
        """
        return self.classify.fit(X_train,y_train)
        
    def validate(self,X_validate,y_validate):
        """
            How good is your system on unseen data? Use the function below to calculate different metrics. 
            Based on these matrix change the hyperparmeters
            
            @param X_validate => NxD tensor. Where N is the number of samples and D is the dimension. 
                                It can be any combination of features provided above.

            @param y_validate => N vector. Ground truth label 
            
        """
        y_pred = self.classify.predict(X_validate)  
        return accuracy_score(y_validate,y_pred),f1_score(y_validate,y_pred, average = "macro"),recall_score(y_validate, y_pred, average="macro"),precision_score(y_validate,y_pred,average="macro")
        # Calculate F1-score
    


# In[37]:


import pandas

def table(path):
    X,y = load_data(path)
    N,H,W = X.shape[0:3]
    C = 1 if opt['is_grayscale'] else X.shape[3]
    X = X.reshape((N,H*W*C))
    
    min_eigenvalues = eigen_spectrum(path,0)
    
    
    # dictionaries for accuracy, f1-score,reduced space
    
    accuracy = {}
    f1_score = {}
    reduced_space = {}
    error = {}
    precision = {}
    recall_score = {}
    n = {}
    features = {"pca":get_pca(X,32),"kernel_pca":get_kernel_pca(X,32,kernel='rbf', degree=4),
                 "lda":get_lda(X,X,y,5),"kernel_lda":get_kernel_lda(X,X,y,5,kernel='rbf',degree=4),"vgg":get_vgg_features(path),
                "resnet":get_resnet_features(path),"lda+pca+kernelpca+kernellda":get_resnet_features(path),"klda+kpca":get_resnet_features(path)
               }
    
    all_classifiers = ["KNN"]
    
    for i in features:
        #print(i)
        
        if i == "pca":
            new_x,_ = features[i]
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)

        elif i=="lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            #print(X_train.shape,X_test.shape)
            
            X_train,X_test = get_lda(X_train,X_test,y_train,5)
            new_x = X_train
    
        elif i =="kernel_lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_train,X_test = get_kernel_lda(X_train,X_test,y_train,5)
            new_x = X_train
        
        elif i == "lda+pca+kernelpca+kernellda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_lda_train,X_lda_test = get_lda(X_train,X_test,y_train,7)
            X_pca_train,_ = get_pca(X_train,32)
            X_pca_test,_ = get_pca(X_test,32)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_lda_train,X_pca_train,X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_lda_test,X_pca_test,X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
        elif i=="klda+kpca":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
            
        else:
            new_x = features[i] 
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
 
        for cls in all_classifiers:
            res = FaceVerification(cls)
            res.train(X_train,y_train)
            accuracy[i],f1_score[i],recall_score[i],precision[i] = res.validate(X_test,y_test)
            accuracy[i] = 100*accuracy[i]
            precision[i] = 100*precision[i]
            #print(accuracy[i])
            name = cls
            reduced_space[i],n[i],error[i]=new_x.shape[1],name,100-accuracy[i]
        
    table = {'Reduced Space':reduced_space,'Accuracy':accuracy,'F1_score':f1_score,'classifier':n,'error':error,'recall_score':recall_score,'precision':precision}
    return table
    
    
print("data-IMFDB")
print("k=3")
pd.DataFrame(table('./dataset/IMFDB/'))
      


# In[38]:


print("k=3")
print("CFW DATA SET")
pd.DataFrame(table('./dataset/IIIT-CFW/'))


# In[39]:


print("k=3")
print("Yale Data Base")
pd.DataFrame(table("./dataset/Yale_face_database/"))


# In[40]:


#lets create a dictionary for classifier


import pandas

def table(path):
    X,y = load_data(path)
    N,H,W = X.shape[0:3]
    C = 1 if opt['is_grayscale'] else X.shape[3]
    X = X.reshape((N,H*W*C))
    
    min_eigenvalues = eigen_spectrum(path,0)
    
    
    # dictionaries for accuracy, f1-score,reduced space
    
    accuracy = {}
    f1_score = {}
    reduced_space = {}
    error = {}
    precision = {}
    recall_score = {}
    n = {}
    features = {"pca":get_pca(X,32),"kernel_pca":get_kernel_pca(X,32,kernel='rbf', degree=4),
                 "lda":get_lda(X,X,y,5),"kernel_lda":get_kernel_lda(X,X,y,5,kernel='rbf',degree=4),"vgg":get_vgg_features(path),
                "resnet":get_resnet_features(path),"lda+pca+kernelpca+kernellda":get_resnet_features(path),"klda+kpca":get_resnet_features(path)
               }
    
    all_classifiers = ["KNN"]
    
    for i in features:
        #print(i)
        
        if i == "pca":
            new_x,_ = features[i]
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)

        elif i=="lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            #print(X_train.shape,X_test.shape)
            
            X_train,X_test = get_lda(X_train,X_test,y_train,5)
            new_x = X_train
    
        elif i =="kernel_lda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_train,X_test = get_kernel_lda(X_train,X_test,y_train,5)
            new_x = X_train
        
        elif i == "lda+pca+kernelpca+kernellda":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_lda_train,X_lda_test = get_lda(X_train,X_test,y_train,7)
            X_pca_train,_ = get_pca(X_train,32)
            X_pca_test,_ = get_pca(X_test,32)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_lda_train,X_pca_train,X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_lda_test,X_pca_test,X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
        elif i=="klda+kpca":
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
            X_kernel_pca_train,X_kernel_pca_test = get_kernel_pca(X_train,32,kernel='rbf'),get_kernel_pca(X_test,32,kernel='rbf')
            X_kernel_lda_train,X_kernel_lda_test = get_kernel_lda(X_train,X_test,y_train,7)
            X_train,X_test,y_train,y_test = np.concatenate((X_kernel_lda_train,X_kernel_pca_train),axis=1),np.concatenate((X_kernel_lda_test,X_kernel_pca_test),axis=1),y_train,y_test
            new_x = X_train
            
        else:
            new_x = features[i] 
            X_train,X_test,y_train,y_test = train_test_split_data(X,y)
 
        for cls in all_classifiers:
            res = FaceVerification1(cls)
            res.train(X_train,y_train)
            accuracy[i],f1_score[i],recall_score[i],precision[i] = res.validate(X_test,y_test)
            accuracy[i] = 100*accuracy[i]
            precision[i] = 100*precision[i]
            #print(accuracy[i])
            name = cls
            reduced_space[i],n[i],error[i]=new_x.shape[1],name,100-accuracy[i]
        
    table = {'Reduced Space':reduced_space,'Accuracy':accuracy,'F1_score':f1_score,'classifier':n,'error':error,'recall_score':recall_score,'precision':precision}
    return table
    
    
print("data-IMFDB")
print("k=5")
pd.DataFrame(table('./dataset/IMFDB/'))
      
        


# In[41]:


print("CFW Data")
print("k=5")
pd.DataFrame(table('./dataset/IIIT-CFW/'))


# In[42]:


print("Yala Data base")
print("k=5")
pd.DataFrame(table("./dataset/Yale_face_database/"))


#cartoon vs real images
#         Use a combination of IIIT-CFW and other dataset. 
# In[43]:


# Load data
X,y = load_data('./dataset/IMFDB/')
N,H,W = X.shape[0:3]
C = 1 if opt['is_grayscale'] else X.shape[3]
X = X.reshape((N,H*W*C))

X1,y1 = load_data('./dataset/IIIT-CFW/')
N1,H1,W1 = X1.shape[0:3]
C = 1 if opt['is_grayscale'] else X1.shape[3]
X1 = X1.reshape((N1,H1*W1*C))

y = [0]*400

y1 = [1]*672

new_X,new_Y = np.concatenate((X,X1),axis=0),np.concatenate((y,y1),axis=0)


# In[44]:


# Define your features


class Classifier():
    def __init__(self,classifier):
        #super.__init__()
        if classifier == "Decision Trees":
            self.classify =  DecisionTreeClassifier()
        elif classifier == "SVM":
            self.classify = SVC(kernel='linear',C=0.25)
        elif classifier == "Logistic Regression":
            self.classify =  LogisticRegression()
        elif classifier == "KNN":
            self.classify = KNeighborsClassifier(n_neighbors=3)
        else:
            self.classify = MLPClassifier(hidden_layer_sizes=(256,128), batch_size=256,early_stopping = True)
    
    # Define your parameters eg, W,b, max_iterations etc. 
    
    def classify1(self,X):
        """
            Given an input X classify it into appropriate class. 
        """
        prediction = self.classify.predict(X)
        
        return prediction
        
    def confusion_matrix(self,pred,y):
        """
            A confusion matrix is a table that is often used to describe the performance of a classification
            model (or “classifier”) on a set of test data for which the true values are known.
            
            
            @return confusion_matrix => num_classesxnum_classes martix 
                where confusion_matrix[i,j] = number of prediction which are i and number of ground truth value equal j 
        
        """
        return confusion_matrix(pred,y)
        
    def train(self,X_train,y_train):
        """
            Given your training data, learn the parameters of your classifier
            
            @param X_train => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your classifier will be trained. 
                                It can be any combination of features provided above.

            @param y_train => N vector. Ground truth label 
    
            @return Nothing
        """
        return self.classify.fit(X_train,y_train)
        
    def validate(self,X_validate,y_validate,j):
        """
            How good is the classifier on unseen data? Use the function below to calculate different metrics. 
            Based on these matrix change the hyperparmeters and judge the classification
            
            @param X_validate => NxD tensor. Where N is the number of samples and D is the dimension. 
                                it is the data on which your classifier validated. 
                                It can be any combination of features provided above.

            @param y_validate => N vector. Ground truth label 
            
        """
        y_pred = self.classify1(X_validate)
        count = 0
        length = y_pred.shape
        X3 = []
        Y3 = []
        X4 = []
        Y4 = []
        for i in range(0,length[0]):
            if y_pred[i] != y_validate[i]:
                X3.append(X_validate[i])
                Y3.append(y_validate[i])
                count +=1
        #print(j)
        if  j != "lda" :
            if j!="kernel_lda":
                length = y_pred.shape
                count = 0
                X3 = []
                Y3 = []
                count1 = 0
                count = 0
                for i in range(0,length[0]):
                    if y_pred[i] != y_validate[i]:
                        X3.append(X_validate[i])
                        Y3.append(y_validate[i])
                        count +=1
                    else:
                        X4.append(X_validate[i])
                        Y4.append(y_validate[i])
                        count1 +=1
                        
                #print(Y3)

                X3 = np.array(X3)
                #print(X3.shape)

                N,H,W, C = count,32,32,3
                X3 = X3.reshape((N,H,W,C))


                Y3 = np.array(Y3)
                print("wrong classified images")
                plt.imshow(X3[0])
                plt.show()
                plt.imshow(X3[1])
                plt.show()
                
                X4 = np.array(X4)
                #print(X3.shape)

                N,H,W, C = count1,32,32,3
                X4 = X4.reshape((N,H,W,C))
                print("correct classified images")


                Y4 = np.array(Y4)
                plt.imshow(X4[0])
                plt.show()
                plt.imshow(X4[1])
                plt.show()
                
              #  plt.imshow(X3[2])
        X3 = []
        Y3 = []
        X4 = []
        Y4 = []


        
    
        return self.confusion_matrix(y_validate,y_pred),accuracy_score(y_validate,y_pred),f1_score(y_validate,y_pred,average="macro"),count/length[0],precision_score(y_validate,y_pred,average="macro")


# In[45]:

def train_test_split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2) 
    return X_train,X_test,y_train,y_test


# In[46]:



import pandas

def table(a):
    
    # dictionaries for accuracy, f1-score,reduced space
    
    accuracy = {}
    f1_score = {}
    reduced_space = {}
    error = {}
    n = {}
    confu_matrix = {}
    precision = {}
    features = {"pca":get_pca(new_X,40),"kernel_pca":get_kernel_pca(new_X,40,kernel='rbf', degree=4),
                "lda":get_lda(new_X,new_X,new_Y,5),"kernel_lda":get_kernel_lda(new_X,new_X,new_Y,5,kernel='rbf',degree=4)}
   
    all_classifiers = ["MLP","SVM","Decision Trees","Logistic Regression"]
    
    for i in features:
      #  print(i)
        temp_accuracy = 0
        temp_f1_score = 0
        
        best_accuracy = 0
        best_f1_score = 0
        temp_precision = 0
        best_precision = 0
        #print(i)
        
        if i == "pca":
            new_x,_ = features[i]
            X_train,X_test,y_train,y_test = train_test_split_data(new_X,new_Y)

        elif i=="lda":
            X_train,X_test,y_train,y_test = train_test_split_data(new_X,new_Y)
            
            X_train,X_test = get_lda(X_train,X_test,y_train,5)
          #  print(X_train.shape,X_test.shape)
            
            new_x = X_train
    
        elif i =="kernel_lda":
            X_train,X_test,y_train,y_test = train_test_split_data(new_X,new_Y)
            X_train,X_test = get_kernel_lda(X_train,X_test,y_train,5)
            new_x = X_train

        
        else:
            new_x = features[i] 
            X_train,X_test,y_train,y_test = train_test_split_data(new_X,new_Y)

        for cls in all_classifiers:
            res = Classifier(cls)
            res.train(X_train,y_train)
            print(cls)

            temp_confusion,temp_accuracy,temp_f1_score,temp_error,temp_precision = res.validate(X_test,y_test,i)
            if best_accuracy < temp_accuracy:
                best_confusion,best_accuracy,best_f1_score,best_error,best_precision = temp_confusion,temp_accuracy,temp_f1_score,temp_error,temp_precision
                name = cls
           # print(cls,"-",temp_accuracy)
        #print(best_accuracy)
        accuracy[i],f1_score[i],reduced_space[i],n[i],error[i],confu_matrix[i],precision[i]=best_accuracy*100,best_f1_score*100,new_x.shape[1],name,best_error*100,best_confusion,best_precision
        if a==2:
            print(confu_matrix)
            
    table = {'Reduced Space':reduced_space,'Accuracy':accuracy,'F1_score':f1_score,'classifier':n,'error':error,'precision':precision}

    if a==1:
        return table
    elif a==0:
        return confu_matrix,n,accuracy
    
    
#print("data-IMFDB")
pd.DataFrame(table(1))
      
        


# In[47]:


print("pca-tsne plot")
pca_X,_ = get_pca(new_X,10)
k=3

TSNE_X = TSNE(k).fit_transform(pca_X)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TSNE_X[:,0],TSNE_X[:,1],TSNE_X[:,2],c=new_Y)
plt.show()
plt.scatter(TSNE_X[:,0],TSNE_X[:,1],c=new_Y)


# In[48]:


print("7fold-accuracy for normal data")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, new_X, new_Y, cv=7)
print(scores)                                              
print("Accuracy: %0.2f" % (scores.mean()))

print("7fold-accuracy for pca applied feature data")
clf = svm.SVC(kernel='linear', C=1)
new_X,_ = get_pca(new_X,10)
scores = cross_val_score(clf, new_X, new_Y, cv=7)
print(scores)                                              
print("Accuracy: %0.2f" % (scores.mean()))


# In[ ]:




