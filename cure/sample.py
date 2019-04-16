import random

total = 150
num = 0.4 * total
s = []

while len(s) < num:
    x = random.randint(0, total)
    if x not in s:
        s.append(x)
print len(s)

file = open('samples1.txt','w')
file.write(str(s));
file.close()
