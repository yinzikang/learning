# number
print('####number####')
number_1 = 1
number_2 = 2
# 只有赋值才创建

# string
print('\n####string####')
string_1 = 'abc'
string_2 = 'def'
string_3 = str()
string_4 = ''
print(string_3.__eq__(string_4))  # True
print('d' in string_2)  # True

# list
print('\n####list####')
list_1 = [[10, 20], ['mm', 'b']]
list_2 = [number_1, string_1]
list_3 = list()
list_4 = []
# 以下均为将list作为元素append了，即出现了list的嵌套
print(list_1.append(list_4))
print(list_2.append(list_4))
print(list_3.append(list_4))  # list_3 = [[]]
print(list_4.append(list_2))  # list_3 = [[[1, 'abc']]], list_4 = [[1, 'abc']]

# tuple
print('\n####tuple####')
tuple_1 = ((100, 200), (300, 400))
tuple_2 = (500,)
tuple_3 = tuple()
tuple_4 = ()
tuple_5 = (list_3,)
print((200) in tuple_1)  # False
print((100, 200) in tuple_1)  # True
print(list_3 in tuple_5)  # True

# dictionary
print('\n####dictionary####')
dict_1 = {number_1: string_1, number_2: string_2}
dict_2 = {string_2: number_2, string_1: number_2}
dict_3 = dict()
dict_4 = {}
print(dict_1[number_1])
print(dict_2[string_2])

# set
print('\n####set####')

while True:
    pass
