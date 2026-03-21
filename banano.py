import numpy as np

# a = np.array(list(range(1, 5)))
# print(type(a))
# print(a)
# print(a.shape)
# print(a[0], a[1], a[2], a[3])
# print(a[0])
#
# b = np.array([[1, 2, 3, 5, 6], [4, 5, 6, 7, 8]])
# print(b.shape)
# print(b)
# b[0, 0] = 1590
# print(b)
# print(b[0, 0], b[0, 1], b[1, 0])


# a = np.zeros((1, 2, 3, 4))
# print(a)
# print(a.shape)
# print("------------------")
#
# b = np.ones((2, 3))
# print(b)
# print(b.shape)
# print("------------------")


# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# print(a)
# print(a.shape)
# b = a[:2, 1:3]
# print(b)
#
# b[0, 0] = -11
# print(a)
# print(b)
# print(a[0,1])

# x = np.array([5, -4])
# print(x.dtype)
#
# x = np.array([1.0, 2.0])
# print(x.dtype)
#
# x = np.array([5, -4], dtype=np.int32)
# print(x.dtype)

# print(np.linspace(2, 3, num=10, endpoint=True, retstep=True))

# import matplotlib.pyplot as plt
#
# # plt.plot([1, 2, 3, 4], [1,4,2,3])
# # plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.linspace(0, 2, 50)
# #print(x)
#
# # Aún con el OO-style, usamos ".pyplot.figure" para crear la figura.
# fig, ax = plt.subplots()   # Crea la figura y los ejes.
#
# ax.plot(x, x, label="linear")       # Dibuja algunos datos en los ejes.
# ax.plot(x, x**2, label="quadratic") # Dibuja más datos en los ejes.
# ax.plot(x, x**3, label="cubic")     # ... y algunos más.
#
# ax.set_xlabel("x label")   # Agrega un x-label a los ejes.
# ax.set_ylabel("y label")   # Agrega un y-label a los ejes.
# ax.set_title("Simple Plot") # Agrega título a los ejes.
#
# ax.legend()   # Agrega una leyenda.
# plt.show()

import matplotlib.pyplot as plt

names = ["group_a", "group_b", "group_c"]
values = [3.4, 50.3, 23]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)

plt.subplot(132)
plt.scatter(names, values)

plt.subplot(133)
plt.plot(names, values)

plt.suptitle("Categorical Plotting")
plt.show()