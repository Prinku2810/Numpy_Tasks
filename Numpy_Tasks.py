
# Numpy
"""



"""#### 1. Import the numpy package under the name `np` (★☆☆) 
(**hint**: import … as …)
"""

import numpy as np

"""#### 2. Print the numpy version and the configuration (★☆☆) 
(**hint**: np.\_\_version\_\_, np.show\_config)
"""

print(np.__version__)
print(np.show_config())

"""#### 3. Create a null vector of size 10 (★☆☆) 
(**hint**: np.zeros)
"""

n=np.zeros(10)
print(n)

"""#### 4.  How to find the memory size of any array (★☆☆) 
(**hint**: size, itemsize)
"""

x=np.array([1,2,3,4,5,6,7,8])
print(x.size)
print(x.itemsize)

"""#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 
(**hint**: np.info)
"""

print(np.info("add"))

"""#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
(**hint**: array\[4\])
"""

y=np.zeros(10)
print(y)
y[5]=1
print(y)

"""#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
(**hint**: np.arange)
"""

x=np.arange(10,49)
print(x)

"""#### 8.  Reverse a vector (first element becomes last) (★☆☆) 
(**hint**: array\[::-1\])
"""

x=np.array([1,2,3,4,5,6])
print(x[::-1])

"""#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
(**hint**: reshape)
"""

x=np.arange(0,9).reshape(3,3)
print(x)

"""#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
(**hint**: np.nonzero)
"""

x=[1,2,0,0,4,0]
print(np.nonzero(x))

"""#### 11. Create a 3x3 identity matrix (★☆☆) 
(**hint**: np.eye)
"""

x=np.eye(3,dtype=int)
print(x)

"""#### 12. Create a 3x3x3 array with random values (★☆☆) 
(**hint**: np.random.random)
"""

x=np.random.random((3,3,3))
print(x)

"""#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
(**hint**: min, max)
"""

x=np.random.random((10,10))
print(np.min(x))
print(np.max(x))

"""#### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
(**hint**: mean)
"""

y=np.random.random((10))
print(np.mean(y))

"""#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
(**hint**: array\[1:-1, 1:-1\])
"""

x=np.ones((10,10))
x[1:-1,1:-1]=0
print(x)

"""#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
(**hint**: np.pad)
"""

x=np.ones((3,3))
x=np.pad(x,pad_width=1,mode="constant",constant_values=0)
print(x)

"""#### 17. What is the result of the following expression? (★☆☆) 
(**hint**: NaN = not a number, inf = infinity)

```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```
"""

print(0*np.nan)
print(np.nan==np.nan)
print(np.inf>np.nan)
print(np.nan-np.nan)
print(0.3==(3*0.1))

"""#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
(**hint**: np.diag)
"""

x=np.diag(1+np.arange(4),k=-1)
print(x)

"""#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
(**hint**: array\[::2\])
"""

x=np.zeros((8,8),dtype=int)
x[1::2,::2]=1
x[::2,1::2]=1
print(x)

"""#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
(**hint**: np.unravel_index)
"""

x=np.unravel_index(100,(6,7,8))
print(x)

"""#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 
(**hint**: np.tile)
"""

x=np.array([[0,1],[1,0]])
x=np.tile(x,(4,4))
print(x)

"""#### 22. Normalize a 5x5 random matrix (★☆☆) 
(**hint**: (x - min) / (max - min))
"""

x=np.random.random((5,5))
mi=x.min()
ma=x.max()
x=(x-mi)/(ma-mi)
print(x)

"""#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 
(**hint**: np.dtype)
"""

x=np.dtype([('r',np.ubyte,1),
            ("g",np.ubyte,1),
            ("b",np.ubyte,1),
            ("a",np.ubyte,1)])
print(x)

"""#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
(**hint**: np.dot | @)
"""

x=np.random.random((5,3))
y=np.random.random((3,5))
z=np.dot(x,y)
print(z)

"""#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 
(**hint**: >, <=)
"""

a=np.array([1,2,3,4,5,6,7,8,9,10,11])
n=(3<a)&(a<=8)
a[n]*=-1
print(a)

"""#### 26. What is the output of the following script? (★☆☆) 
(**hint**: np.sum)

```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
"""

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

"""#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
"""

Z=np.ones(3)
print(Z**Z)
print(Z <- Z)
print(1j*Z)
print(Z/1/1)

"""#### 28. What are the result of the following expressions?

```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
"""

print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

"""#### 29. How to round away from zero a float array ? (★☆☆) 
(**hint**: np.uniform, np.copysign, np.ceil, np.abs)
"""

x=np.random.uniform(-1,-20,10)
z=np.copysign(np.ceil(np.abs(x)),x)
print(z)

"""#### 30. How to find common values between two arrays? (★☆☆) 
(**hint**: np.intersect1d)
"""

x=np.array([1,2,6,32,12])
y=np.array([2,56,23,89,1])
z=np.intersect1d(x,y)
print(z)

"""#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 
(**hint**: np.seterr, np.errstate)
"""

default=np.seterr(all="warn")
z=np.ones(1)/0

with np.errstate(divide="warn"):
  x=np.ones(3)/0

"""#### 32. Is the following expressions true? (★☆☆) 
(**hint**: imaginary number)

```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
"""

print(np.sqrt(-1) == np.emath.sqrt(-1))

"""#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 
(**hint**: np.datetime64, np.timedelta64)
"""

print(np.datetime64('today')-np.timedelta64(1,"D"))
print(np.datetime64('today'))
print(np.datetime64('today')+np.timedelta64(1,"D"))

"""#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 
(**hint**: np.arange(dtype=datetime64\['D'\]))
"""

print(np.arange('2016-07','2016-08',dtype='datetime64[D]'))

"""#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 
(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))
"""

A=np.array([1,3,4,5])
B=np.array([4,5,6,7])
C=np.array([1,2,3,4])
np.add(A,B,out=B)
np.divide(A,2,out=A,casting='unsafe')
np.negative(A,out=A)
print(np.multiply(A,B,out=A))

"""#### 36. Extract the integer part of a random array using 5 different methods (★★☆) 
(**hint**: %, np.floor, np.ceil, astype, np.trunc)
"""

x=np.random.uniform(0,100,5)
print(x)
print(np.floor(x))
print(np.ceil(x))
print(x.astype(int))
print(np.trunc(x))
print(x-x%1)

"""#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
(**hint**: np.arange)
"""

x=np.zeros((5,5))
x+=np.arange(0,5)
print(x)

"""#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 
(**hint**: np.fromiter)
"""

def arr():
  for i in range(30):
     yield i

x=np.fromiter(arr(),dtype=float,count=-1)
print(x)

"""#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
(**hint**: np.linspace)
"""

x=np.linspace(0,1,10,endpoint='false')[1:]
print(x)

"""#### 40. Create a random vector of size 10 and sort it (★★☆) 
(**hint**: sort)
"""

x=np.random.random(10)
print(np.sort(x))

"""#### 41. How to sum a small array faster than np.sum? (★★☆) 
(**hint**: np.add.reduce)
"""

x=np.arange(20)
print(np.add.reduce(x))

"""#### 42. Consider two random array A and B, check if they are equal (★★☆) 
(**hint**: np.allclose, np.array\_equal)
"""

x=np.random.randint(3,7,size=10)
y=np.random.randint(2,5,size=10)

print(np.allclose(x,y))
print(np.array_equal(x,y))

"""#### 43. Make an array immutable (read-only) (★★☆) 
(**hint**: flags.writeable)
"""

x=np.array([1,2,3,4,5,6])
x.flags.writeable=False
x[5]=10
print(x)

"""#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 
(**hint**: np.sqrt, np.arctan2)
"""

x=np.random.random((10,2))
X,Y=x[:,0],x[:,1]
s=np.sqrt(X**2+Y**2)
n=np.arctan2(Y,X)
print(s)
print(n)

"""#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
(**hint**: argmax)
"""

x=np.random.random((10))

x[x.argmax()]=0
print(x)

"""#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 
(**hint**: np.meshgrid)
"""

s = np.zeros((5,5), [('x',float),('y',float)])
s['x'], s['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(s)

"""####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 
(**hint**: np.subtract.outer)
"""

X = np.arange(10)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

"""#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 
(**hint**: np.iinfo, np.finfo, eps)
"""

for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

"""#### 49. How to print all the values of an array? (★★☆) 
(**hint**: np.set\_printoptions)
"""

np.set_printoptions(threshold=float("inf"))
x=np.arange(20)
print(x)

"""#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 
(**hint**: argmin)
"""

x=np.arange(10)
c=np.random.uniform(0,10)
v=(np.abs(x-c)).argmin()
print(x[v])

"""#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 
(**hint**: dtype)
"""

x=np.zeros(10,[("position",[("x",float,1),("y",float,1)]),
             ("color",[("r",float,1),("g",float,1),("b",float,1)])
             
])
print(x)

"""#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 
(**hint**: np.atleast\_2d, T, np.sqrt)
"""



"""#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 
(**hint**: astype(copy=False))
"""

x=(np.random.rand(1)*1000).astype(np.float32)
y=x.view(np.int32)
print(y)

"""#### 54. How to read the following file? (★★☆) 
(**hint**: np.genfromtxt)

```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
"""

from io import StringIO
f = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9,10,11
''')
x = np.genfromtxt(f, delimiter=",", dtype=np.int)
print(x)

"""#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 
(**hint**: np.ndenumerate, np.ndindex)
"""

x=np.array([[2,3,4],[5,6,7]])
for ind,y in np.ndenumerate(x):
  print(ind,y)
print("\n")
for ind in np.ndindex(x.shape):
  print(ind,x[ind])

"""#### 56. Generate a generic 2D Gaussian-like array (★★☆) 
(**hint**: np.meshgrid, np.exp)
"""

x, y = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
dst = np.sqrt(x*x+y*y)
sigma=1
muu=0.000
gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
print(gauss)

"""#### 57. How to randomly place p elements in a 2D array? (★★☆) 
(**hint**: np.put, np.random.choice)
"""

x = np.zeros((10,5))
np.put(x, np.random.choice(range(5), 2, replace=False),1)
print(x)

"""#### 58. Subtract the mean of each row of a matrix (★★☆) 
(**hint**: mean(axis=,keepdims=))
"""

x=np.random.rand(5,2)
y=x-x.mean(axis=1,keepdims=True)
print(y)

"""#### 59. How to sort an array by the nth column? (★★☆) 
(**hint**: argsort)
"""

a=np.random.rand(5,2)
print(a)
print(a.argsort(axis=1))

"""#### 60. How to tell if a given 2D array has null columns? (★★☆) 
(**hint**: any, ~)
"""

x = np.random.randint(0,5,(3,5))
print(x)
print(~x.any(axis=1))

"""#### 61. Find the nearest value from a given value in an array (★★☆) 
(**hint**: np.abs, argmin, flat)
"""

x=np.arange(10)
print(x)
c=np.random.uniform(0,10)
print(c)
v=x.flat[(np.abs(x-c)).argmin()]
print(v)

"""#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 
(**hint**: np.nditer)
"""

x=np.arange(3).reshape(1,3)
y=np.arange(3).reshape(3,1)
print(x)
print(y)
t=np.nditer([x,y,None])
for a,b,c in t:
  c[...]=a+b
print(t.operands[2])



"""#### 63. Create an array class that has a name attribute (★★☆) 
(**hint**: class method)
"""

class Narray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

x = Narray(np.arange(10), "range_10")
print (x.name)

"""#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 
(**hint**: np.bincount | np.add.at)
"""

x = np.ones(10)
print(x)
u = np.random.randint(0,len(x),20)
print(u)
x += np.bincount(u, minlength=len(x))
print(x)

np.add.at(x, u, 1)
print(x)

"""#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 
(**hint**: np.bincount)
"""

x = [1,2,3,4,5,6]
y = [1,2,3,4,5,6]
F = np.bincount(x,y)
print(F)

"""#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 
(**hint**: np.unique)
"""

w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
print(I)
colors = np.unique(I.reshape(-1, 3), axis=0)
n = len(colors)
print(n)

"""#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 
(**hint**: sum(axis=(-2,-1)))
"""

x=np.random.randint(0,10,(3,4,3,4))
s=x.sum(axis=(-2,-1))
print(s)

"""#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 
(**hint**: np.bincount)
"""

import sys
x= np.random.randint(0,10,10)
y = np.random.randint(0,10,10)
print(x)
print(y)
sums = np.bincount(x, weights=y)
counts = np.bincount(x)
means = sums / counts
print(means)

"""#### 69. How to get the diagonal of a dot product? (★★★) 
(**hint**: np.diag)
"""

a=np.random.uniform(0,1,(4,4))
b=np.random.uniform(0,10,(4,4))
s=np.diag(np.dot(a,b))
print(s)

"""#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
(**hint**: array\[::4\])
"""

x = np.array([1,2,3,4,5])
n = 3
y = np.zeros(len(x) + (len(x)-1)*(n))
print(y)
y[::n+1] = x
print(y)

"""#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
(**hint**: array\[:, :, None\])
"""

a=np.ones((5,5,3))
b=np.ones((5,5))
x=(a*b[:,:,None])
print(x)

"""#### 72. How to swap two rows of an array? (★★★) 
(**hint**: array\[\[\]\] = array\[\[\]\])
"""

a=np.arange(25).reshape(5,5)
print(a)
a[[1,2]]=a[[2,1]]
print(a)

"""#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 
(**hint**: repeat, np.roll, np.sort, view, np.unique)
"""

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)

"""#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 
(**hint**: np.repeat)
"""

C = np.bincount([1,2,4,4,5,2,6,7])
print(C)
A = np.repeat(np.arange(len(C)), C)
print(A)

"""#### 75. How to compute averages using a sliding window over an array? (★★★) 
(**hint**: np.cumsum)
"""

x=np.arange(10)
print(x)
z=np.lib.stride_tricks.sliding_window_view(x,3)
print(z)
print(np.cumsum(z))

"""#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 
(**hint**: from numpy.lib import stride_tricks)
"""

from numpy.lib import stride_tricks
x=np.arange(10)
print(stride_tricks.sliding_window_view(x,window_shape=3))

"""#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 
(**hint**: np.logical_not, np.negative)
"""

x=np.random.randint(0,10,10)
print(x)
print(np.logical_not(x))
print(np.negative(x))



"""#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)"""

p0 = np.random.uniform(-5,5,(10,2))
p1 = np.random.uniform(-5,5,(10,2))
p  = np.random.uniform(-5,5,( 1,2))

print(p)
s=p0-p1
print(s)
sums=(s**2).sum(axis=1)
print(sums)
d= -((p0[:,0]-p[...,0])*s[:,0] + (p0[:,1]-p[...,1])*s[:,1]) / sums
print(d)
d = d.reshape(len(d),1)
di = p0 + d*s - p
print(np.sqrt((di**2).sum(axis=1)))

"""#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)"""



"""#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 
(**hint**: minimum, maximum)
"""

x = np.arange(1,15)
y = stride_tricks.as_strided(x,(11,4),(4,4))
print(y)

"""#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 
(**hint**: stride\_tricks.as\_strided)
"""

x = np.arange(1,15,dtype=np.uint32)
y = np.lib.stride_tricks.as_strided(x,(11,4),(4,4))
print(y)

"""#### 82. Compute a matrix rank (★★★) 
(**hint**: np.linalg.svd) (suggestion: np.linalg.svd)
"""

x=np.arange(15).reshape(3,5)
print(x)
u,v,h=np.linalg.svd(x)
print(v)
print(np.sum(v>1e-10))

"""#### 83. How to find the most frequent value in an array? 
(**hint**: np.bincount, argmax)
"""

x = np.random.randint(0,5,20)
print(x)
print(np.bincount(x).argmax())

"""#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 
(**hint**: stride\_tricks.as\_strided)
"""

x=np.random.randint(0,10,(10,10))
print(x)
a=1+(x.shape[0]-3)
b=1+(x.shape[1]-3)
#print(x.strides)
z=np.lib.stride_tricks.as_strided(x,shape=(a,b,3,3),strides=x.strides+x.strides)
print(z)
print(z.strides)

"""#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 
(**hint**: class method)
"""

class arr(np.ndarray):
  def __setitem__(self,index,value):
    i,j=index
    super(arr, self).__setitem__((i,j), value)
    super(arr, self).__setitem__((j,i), value)
def sys(x):
  return np.asarray(x-x.T).view(arr)
x=sys(np.random.randint(0,5,(5,5)))
x[3,2]=12
print(x)

"""#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 
(**hint**: np.tensordot)
"""

x=np.ones((10,10,10))
y=np.ones((10,10,1))
c=np.tensordot(x,y,axes=[[0,2],[0,1]] )
print(c)

"""#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 
(**hint**: np.add.reduceat)
"""

x = np.ones((16,16))

y = np.add.reduceat(np.add.reduceat(x, np.arange(0, x.shape[0], 4), axis=0),
                                       np.arange(0, x.shape[1], 4), axis=1)
print(y)

"""#### 88. How to implement the Game of Life using numpy arrays? (★★★)"""



"""#### 89. How to get the n largest values of an array (★★★) 
(**hint**: np.argsort | np.argpartition)
"""

x = np.arange(25)
np.random.shuffle(x)
a=[np.argsort(x)[-10:]]
b=[np.argpartition(-x,5)[:10]]
print(a)
print(b)

"""#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 
(**hint**: np.indices)
"""



"""#### 91. How to create a record array from a regular array? (★★★) 
(**hint**: np.core.records.fromarrays)
"""

x = np.arange(10).reshape(5,2)
y = np.core.records.fromarrays(x.T,
                               names='col1, col2',
                               formats = 'i8,i8')
print(y)

"""#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 
(**hint**: np.power, \*, np.einsum)
"""

x=np.random.randint(1000000)
print(x)
print(np.power(x,3))
print(x**3)

"""#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 
(**hint**: np.where)
"""

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)

"""#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)"""

Z = np.random.randint(0,5,(2,4))
print(Z)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)

"""#### 95. Convert a vector of ints into a matrix binary representation (★★★) 
(**hint**: np.unpackbits)
"""

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))

"""#### 96. Given a two dimensional array, how to extract unique rows? (★★★) 
(**hint**: np.ascontiguousarray)
"""

x = np.random.randint(0,2,(6,6))
print(x)
y=np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
_,d = np.unique(y, return_index=True)
Z = x[d]
print(Z)

"""#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 
(**hint**: np.einsum)
"""

x = np.random.uniform(0,10,5)
y = np.random.uniform(0,10,5)
print(x)
print(y)
#sum
np.einsum('i->', x)  
#mul
np.einsum('i,i->i', x, y) 
#inner
np.einsum('i,i', x, y)  
#outer  
np.einsum('i,j->ij', x, y)

"""#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 
(**hint**: np.cumsum, np.interp)
"""



"""#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 
(**hint**: np.logical\_and.reduce, np.mod)
"""



"""#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 
(**hint**: np.percentile)
"""

