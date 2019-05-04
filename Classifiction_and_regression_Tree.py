
######################################################################################
##################################################################################
#This function splits data into train and test dataset 
# Input to the function
# 1. datafrmae 
# 2. portion of the data you want as train (integer or number between 1 to1).
###################################################################################

import numpy as np
import pandas as pd
import random

def train_test_split(df,portion):
    if portion<1:
        portion=portion*len(df)
    
    portion=int(portion)
    sample=random.sample(range(len(df)),portion)
    
    train_df =df.loc[sample].reset_index(drop=True)
    test_df  =df.drop(sample,axis=0).reset_index(drop=True)
    return train_df,test_df

#################################################################################################################
# This function check the purity of subset if it contains one class it will return true else False 
# Input for the function																		   
# 1.Data frame with all columns																	   
# 2.Y variable column Name 
# 3.Min width(no of element allowed on node) after this you will prune the tree 
######################################################################################################
import numpy as np
def data_check(data,Y_column_name):
    label_column=data[Y_column_name].values
    
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
		
		
#################################################################################################
#This function classify the subset based on max count and in clase of regression mean of Y variable
# Input for the function																		   
# 1.Data frame with all columns																	   
# 2.Y variable column Name 
##################################################################################################
def classify(data,Y_column_name):
    label_column=data[Y_column_name].values
    
#     checking if the Y_column_name is continuous and catagorical if catagorical then classify else regress
    
    if len(np.unique(label_column))<10 or isinstance(label_column[0],str):
          
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]

        return classification
    
    else:
        regression=label_column.mean()
        
        return regression
		
#################################################################################################
# This function generate's all possible split for both catagorical and numerical data
# Input for the function
# 1.Data frame with all columns
# 2.Y variable column Name 
##################################################################################################

def possibe_splits(tr,Y_column_name):
    rf=tr.drop(Y_column_name,axis=1)
    d={}
    for column in rf.columns:
        a=[]
## Checking for the catagorical and numerical variables
        if len(np.unique(rf[column].values))<10 or isinstance(rf[column].values[0],str):
            
            a=np.unique(rf[column].values)

        else:
            mm=np.sort(np.unique(tr[column].values))
            a=[]
            for i in range(len(mm)-1):
                a.append((mm[i]+mm[i+1])/2)
           
        d[column]=a  
		
#################################################################################################
### Splitting the data based on given index and col_name
# Input for the function
# 1.Dataframe with all columns
# 2.column name on which you want to split the dataframe
# 3.row no on which you want to split the DataFrame
##################################################################################################

def splits_data(tr,col_name,row_val):
    if len(np.unique(tr[col_name].values))<10 or isinstance(tr[col_name].values[0],str):
        df_upper=tr[tr[col_name]==row_val].reset_index(drop=True)
        df_lower=tr[tr[col_name]!=row_val].reset_index(drop=True)
    else:
        df_upper=tr[tr[col_name]>row_val].reset_index(drop=True)
        df_lower=tr[tr[col_name]<=row_val].reset_index(drop=True)
    
    return df_upper,df_lower
	
#################################################################################################
# Getting the entropy of the Y variable on given Datafram 
# Input for the function
# 1.Dataframe with all columns
# 2.column name of Y variable
##################################################################################################
	
############################################## For Classification Problem     ###########################
def entropy(tr,Y_column_name):
    M=np.unique(tr[Y_column_name].values,return_counts=True)[1]
    dat_entropy=-np.array([i*np.log2(i) for i in M/len(tr)]).sum()
    return dat_entropy

def Gini(tr,Y_column_name):
    M=np.unique(tr[Y_column_name].values,return_counts=True)[1]
    dat_Gini=1-np.array([i*i for i in M/len(tr)]).sum()
    return dat_Gini


#######################################  For Regression Problem  ########################################

##### For Regression you need to to check if we are losing in variance or not 
##### This we can do by just checking if the variance is less then previous split or not 
##### Here both MSE and variance are equal since we will be predicting the value of mean of subset

def Variance_MSE(tr,Y_column_name):
    M=0
    M=np.unique(tr[Y_column_name].values,return_counts=True)[1]
    MSE=np.var(M)
    return MSE

#################################################################################################
# Check lest entropy subset and provide column and row value 
# Input for the function
# 1.Data frame with all columns
# 2.column name of Y variable
# 3. All split dictionary generated earlier 
##################################################################################################


#######################################################For calassification#############################################

def bestsplit(tr,all_splits,Y_column_name):
    ent=999
    for i in all_splits.keys():
        for j in all_splits[i]:
            dataup,datalow=splits_data(tr,i,j)
            
            enup=len(dataup)*entropy(dataup,Y_column_name)
            endown=len(datalow)*entropy(datalow,Y_column_name)
            
            entro=(enup+endown)/len(tr)
# Here I had issue on code (less then and equal to) will work, if I apply only less then it will not work correctly you can try            
            if entro<=ent:
                ent=entro
                col=i
                row=j
                
    return(col,row)

############################################################ For Regression ###########################################


def bestsplit_Regress(tr,all_splits,Y_column_name):
    ent=999
    for i in all_splits.keys():
        for j in all_splits[i]:
            dataup,datalow=splits_data(tr,i,j)
            
            enup=len(dataup)*Variance_MSE(dataup,Y_column_name)
            endown=len(datalow)*Variance_MSE(datalow,Y_column_name)
            
            entro=(enup+endown)/len(tr)
# Here I had issue on code (less then and equal to) will work, if I apply only less then it will not work correctly you can try            
            if entro<=ent:
                ent=entro
                col=i
                row=j
                
    return(col,row)

	
#################################################################################################
# This is the final classification algorithm which generates tree with help of recursion  
# Input for the function
# 1.Data frame with all columns
# 2.column name of Y variable
# 3.Min no of element in subset at which you want to prune the tree
# 4.Max length of the tree after which you want to prune the tree 
##################################################################################################


def decision_tree_classifire(pr,Y_column_name,min_width=2,max_depth=3,count=0):
    
    if (data_check(pr,Y_column_name) or (len(pr) < min_width) or (count == max_depth)):
        return classify(pr,Y_column_name)

    else:
        count+=1
        all_splits=possibe_splits(pr,Y_column_name)
        best_col,best_row=bestsplit(pr,all_splits,Y_column_name)
        data_up,data_down=splits_data(pr,best_col,best_row)
        
        # checking if column is catagorical or continuous
        
        if len(np.unique(pr[best_col].values))<10 or  isinstance(pr[best_col].values[0],str):
            
            question = "{} = {}".format(best_col,best_row)
        else:
            question = "{} <= {}".format(best_col,best_row)
        sub_tree = {question: []}
        
        yes_answer = decision_tree_classifire(data_down,Y_column_name, min_width,max_depth,count)
        no_answer = decision_tree_classifire(data_up,Y_column_name, min_width,max_depth,count)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
		
####################################################################################################
# This is the final regression algorithm which generates tree with help of recursion  
# Input for the function
# 1.Data frame with all columns
# 2.column name of Y variable
# 3.Min no of element in subset at which you want to prune the tree
# 4.Max length of the tree after which you want to prune the tree 
####################################################################################################

##########################################################   Decision Tree Regressor  ############


def decision_tree_regressor(pr,Y_column_name,min_width=2,max_depth=3,count=0):
    
    if (data_check(pr,Y_column_name) or (len(pr) < min_width) or (count == max_depth)):
        return classify(pr,Y_column_name)

    else:
        count+=1
        all_splits=possibe_splits(pr,Y_column_name)
        best_col,best_row=bestsplit_Regress(pr,all_splits,Y_column_name)
        data_up,data_down=splits_data(pr,best_col,best_row)
        
        # checking if column is catagorical or continuous
        
        if len(np.unique(pr[best_col].values))<10 or  isinstance(pr[best_col].values[0],str):
            question = "{} = {}".format(best_col,best_row)
        else:
            question = "{} <= {}".format(best_col,best_row)
        sub_tree = {question: []}
        
        yes_answer = decision_tree_regressor(data_down,Y_column_name, min_width,max_depth,count)
        no_answer  = decision_tree_regressor(data_up,Y_column_name, min_width,max_depth,count)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
		
		

####################################################################################################
# This is the predict function it will take the predictors and will predict the value of Y variable
# This function is same for classification and regression   
# Input for the function
# 1.df.loc[ ]  value which will contain all predictors
# 2.Tree generated by classifier and regressors trees
####################################################################################################	

def predict(inputs,tree):
    ask=list(tree.keys())[0]
    feature_name, comparison_operator, value = ask.split(" ")
    if comparison_operator == "<=":
        if inputs[feature_name] <= float( value):
            answer = tree[ask][0]
        else:
            answer = tree[ask][1]
    else:
        if inputs[feature_name] == float( value):
            answer = tree[ask][0]
        else:
            answer = tree[ask][1]
        
            
        
    if isinstance(answer,dict):
        rev_tree= answer
        return predict(inputs,rev_tree)
    else:
        return answer	
		
		
		

