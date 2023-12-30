import numpy as np
import numpy.linalg as la
A=[[0.2161,0.1441],[1.2969,0.8648]]
A_array=np.array(A)

b=[0.1440,0.8642]

b_array=np.array(b)

print(np.linalg.solve(A_array,b_array))

print(np.linalg.det(A_array))


A1=[[2,1],[1,2]]
A1_array=np.array(A)

b1=[2,7]

b1_array=np.array(b)

print(np.linalg.solve(A1_array,b1_array))
print(la.det(A1_array))

A2=[[1,1],[1,1.001]]
b2=[2,2.001]

A2_array=np.array(A2)
b2_array=np.array(b2)

print(la.solve(A2_array,b2_array))
print(la.det(A2_array))


A3=[[0.2,0.101],[0.1,0.2]]
b3=[0.2,0.7]

A3_array=np.array(A3)
b3_array=np.array(b3)

print(la.solve(A3_array,b3_array))
print(la.det(A3_array))


# 4.2.3 in gezerlis


print(la.norm(A_array))
print(la.norm(A2_array))
print(la.norm(A1_array))


# A=np.array([[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2],[-2,-2,-2,-2,-2,-2,-2,-2]])
# A=np.triu(A,0)
# b=np.array([1,-1,1,-1,1,-1,1,-1])
# print(la.solve(A,b))
# print(la.norm(A))
# print(la.det(A))
# print(A)
# def create_upper_matrix(values, size):
#     upper = np.zeros((size, size))
#     upper[np.triu_indices(3, 0)] = values
#     return(upper)

# c = create_upper_matrix([-2, -2, -2, -2, -2, -2,-2,-2], 8)
# print(c)


#4.2.4 in gezerlis

#lets set up an eigenvalue problem

A=[[4,3,2,1.01],[3,3,2,1],[0,2,2,1],[0,0,1,1]]
a=np.array(A)

print(la.eig(a))       #the eigenvectors being printed are normalized (simplified)