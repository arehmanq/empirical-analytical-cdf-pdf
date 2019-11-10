import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.stats as stats

print('Uniform Random Variable Program')
a=float(input('Please input the value of a'))
b=float(input('Please input the value of b'))

x=np.linspace(a-3,b+3 ,100)

#=======================================Histogram
data = uniform.rvs(a, b-a, size=10000)
z=plt.hist(data, bins=40, density=True, alpha=0.6, color='g', edgecolor='black')
plt.title('Histogram')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

#===================Empricial(Obtain via Histogram) vs Analytical CDF
data = uniform.rvs(a, b-a, size=10000)
plt.hist(data, bins=40, density=True,histtype='step',cumulative=True, label='Empirical')


y=uniform.cdf(x,a,b-a)
plt.plot(x,y)
plt.title('CDF:Uniform Random Variable')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

#===================Empricial(Obtain via Histogram) vs Analytical PDF
data = uniform.rvs(a, b-a, size=100000)
density = stats.gaussian_kde(data)
n, e, _ = plt.hist(data, bins=100, histtype='step', density=True)  
plt.close()
plt.figure()
plt.plot(e, density(e))

y=uniform.pdf(x,a,b-a)
plt.plot(x,y)
plt.title('PDF:Uniform Random Variable')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
import scipy.stats as stats
print('Exponential Random Variable Program')
a=float(input('Please input the value of Mean'))
lambd=1/a                    #lambda=1/mean


x=np.arange(0,15,0.1)

#اHistogram
data=expon.rvs(scale=a, size=10000)
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')

plt.title('Histogram')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

# Empirical(obtain via Histogram) vs Analytical cdf
data=expon.rvs(scale=a,loc=0, size=1000000)
plt.hist(data, bins=50, density=True,histtype='step',
                           cumulative=True, label='Empirical')

y=1-np.exp(-lambd*x)
plt.plot(x,y)
plt.title('Exponential RV CDF: Mean=%.2f' %a)
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

#Empirical(obtain via Histogram) vs  Analytical PDF   #for smooth line on pdf uncomment 1,2,3,4
data=expon.rvs(scale=a, size=1000000)
#density = stats.gaussian_kde(data)=========================================1
n, e, _ = plt.hist(data, bins=400, histtype='step', density=True)  
#plt.close()================================================================2

#plt.figure()===============================================================3
#plt.plot(e, density(e))====================================================4
y=lambd*np.exp(-lambd*x)
plt.plot(x,y)

plt.title('Exponential RV PDF: Mean=%.2f' %a)
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import math
from IPython.display import Math, Latex
# for displaying images
from IPython.core.display import Image
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
import scipy.stats as stats


print('Guassian Random Variable Program')
m = float(input('Please Enter the value of mean'))
var = float(input('Please enter the value of variance'))
std = math.sqrt(var)      #calculating std. deviation
x = np.linspace(m - 3*std, m + 3*std, 100)

#==============================================obtain histogram
data = norm.rvs(m, std, size=100000)
z=plt.hist(data, bins=60, density=True, alpha=0.6, color='g', edgecolor='black')
plt.title('Histogram')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()
#====================================Empricial(Obtain via Histogram) vs Analytical PDF
data = np.random.normal(m, std, size=100000)
density = stats.gaussian_kde(data)
n, e, _ = plt.hist(data, bins=60, histtype='step', density=True)  
plt.close()
plt.figure()
plt.plot(e,density(e))
y=norm.pdf(x, m, std)
plt.plot(x,y)
plt.title('Guassian RV PDF')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()
#=====================================Empricial(Obtain via Histogram) vs Analytical CDF
data = np.random.normal(m, std, size=100000)
plt.hist(data, bins=70, density=True,histtype='step',cumulative=True)

y2=norm.cdf(x, m, std)
plt.plot(x,y2)
plt.title('Guassian RV CDF')
plt.xlabel('x')
plt.ylabel('Probability')
plt.show()
