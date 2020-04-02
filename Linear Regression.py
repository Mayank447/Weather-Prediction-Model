#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as pl
import numpy as np
import pandas as pd


# In[3]:


irisdb = load(iris_db)


# In[1]:


from sklearn import datasets
irisdb = datasets.load_iris()
print(irisdb)


# In[2]:


X=irisdb.data[:,:2]
Y=irisdb.target
X
Y


# In[3]:


print(X,Y)


# In[6]:


from mpl_toolkits.mplot3d import Axes3D


# In[9]:


#Linear Regression Algorithm
xl=X[:,[0]]
print(xl)


# In[13]:


pl.scatter(xl,Y)


# In[14]:


a=np.array([1,3,6,5,8,4,6,51,84,0,53,14,98,4,12,1,68,4])
b=np.array([98,46,51,24,68,74,54,31,45,74,41,17,54,11,94,26,51,78])
print(np.size(a),np.size(b))


# In[15]:


pl.plot(a,b)


# In[16]:


pl.scatter(a,b)


# In[17]:


am=np.mean(a)
bm=np.mean(b)
print(am,bm)


# In[23]:


sum_m,sum_m2=0,0
''''def m():
    for i in range(0,len(a),1):
        temp=((a[i]-am)(b[i]-bm))
        temp2=((a[i]-am)**2)
        sum_m2=sum_m2+temp2
        sum_m=sum_m+temp
    return sum_m/sum_m2'''
for i in range(0,len(a),1):
        temp=((a[i]-am)*(b[i]-bm))
        temp2=((a[i]-am)**2)
        sum_m2=sum_m2+temp2
        sum_m=sum_m+temp
result = sum_m/sum_m2


# In[21]:


print(a[5])


# In[22]:


print(a-am)


# In[29]:


=(a-am)*(b-bm)
print(r)


# In[25]:


(a[0]-am)*(b[0]-bm)


# In[26]:


print(result)


# In[30]:


n=np.sum(r)


# In[31]:


d=np.sum((a-am)**2)
print(n,d,n/d)


# In[32]:


m=n/d


# In[33]:


#y=mx+c
#c=y-mx
c=bm-(m*am)
print(c)


# In[35]:


x=np.linspace(1,100,500)
y=m*x + c


# In[36]:


pl.plot(x,y)
pl.scatter(a,b)


# In[37]:


#Function for finding slope for Linear Regression
def slope(x,y):
    for i in range(0,len(x),1):
        temp=((x[i]-np.mean(x))(y[i]-np.mean(y)))
        temp2=((x[i]-np.mean(x))**2)
        sum_m2=sum_m2+temp2
        sum_m=sum_m+temp
    m=sum_m/sum_m2
    return m


# In[40]:


pl.scatter(X,Y)


# In[47]:


#Function for finding R-square Error for Linear Regression approximations.
yp=m*a + c
def LR_error(yp,b):
    N=D=0
    for i in range(0,len(yp),1):
        n=(yp[i]-np.mean(b))**2
        N=n+N # N refers to sum of all numerators
        d=(b[i]-np.mean(b))**2
        D=d+D # D refers to sum of all denominators 
    E=N/D # E refers to the total error
    return E


# In[48]:


print(LR_error(yp,b))


# In[77]:


#Final Linear Regression Model

def Linear_Regression():
    
    
    def List_input(L):
        #A function to take elements in a list
        n=int(input("Enter the length of the array: "))
        for i in range(0,n,1):
            L.append(int(input("Enter the element: ")))
            
    
    #Function to input arrays for Linear Regression
    def array_input():
        L1=[]
        L2=[]
        List_input(L1)
        List_input(L2)
        A1=np.array(L1)
        A2=np.array(L2)
        return A1,A2

    x,y=array_input()    


    #Function for finding slope for Linear Regression
    def slope(x,y):
        sum_m2=sum_m=0
        for i in range(0,len(x),1):
            temp=((x[i]-np.mean(x))*(y[i]-np.mean(y)))
            temp2=((x[i]-np.mean(x))**2)
            sum_m2=sum_m2+temp2
            sum_m=sum_m+temp
        m=sum_m/sum_m2
        return m

    m=slope(x,y)

    #Function for finding constant(Y-intercept) of the line
    def const(x,y,m):
        #y=mx+c
        #c=y-mx
        c=np.mean(y)-(m*np.mean(x))
        return c

    c=const(x,y,m)
    print(x,y,m,c)

    #Equation of Predicted Linear Function
    def Pred_eq(m,c):
        print("The Predicted Equation of the line is y = ",m,'x + ',c,sep='')

    Pred_eq(m,c)
    
    #Function for plotting both the points of arrays taken as input and of Regression Line
    def Graph_plot(m,c):
        x_=np.linspace(1,100,500)
        yp=m*x_ + c
        pl.plot(x_,yp)
        pl.scatter(x,y)
        return yp

    yp=Graph_plot(m,c)
 


    #Function for finding R-square Error for Linear Regression approximations.
    def LR_error(yp,y):
        N=D=0
        for i in range(0,len(yp),1):
            n=(yp[i]-np.mean(y))**2
            N=n+N # N refers to sum of all numerators
            d=(y[i]-np.mean(y))**2
            D=d+D # D refers to sum of all denominators 
        E=N/D # E refers to the total error
        print("The R-square value is",E)
        return E


# In[68]:


Linear_Regression()


# In[54]:





# In[55]:





# In[57]:


print(np.array([88544,5,18,12,51,8,1]))


# In[78]:


Linear_Regression()


# In[ ]:


Linear_Regression()


# In[ ]:


Linear_Regression()


# In[ ]:




