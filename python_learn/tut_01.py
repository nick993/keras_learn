# List comprehension
l1 = [1,3,4,6]
double_list = [2 * x for x in l1]
print(double_list)

l2 = range(10)
squares = list(map(lambda x : x ** 2, l2))
print(squares)

l3 = [(x, y) for x in [1,3,4] for y in [1, 3, 4] if x != y]
print(l3)