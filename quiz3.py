import numpy as np
import numpy.linalg as la


A=np.array([[0.8647,0.5766],[0.4322,0.2822]])

b=np.array([0.2885,0.1442])

detA=la.det(A)
normA=la.norm(A)

sol=la.solve(A,b)
invA=la.inv(A)
condition_no=normA*la.norm(invA)
print('determinant of A:',detA)
print('norm of A:', normA)
print('sol:',sol)
print('condition no:',condition_no)

#condition no is norm of A multiplied by norm of A inversed

A1=np.array([[0.8647,0.5766],[0.4322001,0.2822]]) #perturbed matrix 1

sol_pert=la.solve(A1,b)

print('perturbed solution2:',sol_pert)

print('difference in original and perturbed solutions:',(sol-sol_pert))

A2=np.array([[0.8647001,0.5766],[0.4322,0.2822]]) #perturbed matrix 2

sol_pert1=la.solve(A2,b)

print('perturbed solution2:',sol_pert1)

print('difference in original and perturbed solutions:',(sol-sol_pert1))