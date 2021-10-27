import torch

#scalar1
scalar1 = torch.tensor([1.])
print(scalar1)

#scalar2
scalar2 = torch.tensor([3.])
print(scalar2)

#add_scalar
#method1
add_scalar = scalar1 + scalar2
print(add_scalar)

#method2
add_scalar2 = torch.add(scalar1,scalar2)
print(add_scalar2)

#vector1
vector1 = torch.tensor([1,2,3])
print(vector1)

#vector2
vector2 = torch.tensor([4.,5.,6.])
print(vector2)

#add_vector
add = vector1 + vector2
print(add)

