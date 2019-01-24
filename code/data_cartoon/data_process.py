cnt = 0
for line in open('trainBB.txt','r'):
    #print(line[0:-3])
    cnt = cnt+1
tmp = 0
for line in open('trainBB.txt','r'):
    tmp = tmp+1
    if tmp >= 0.9 * cnt:
        print(line[0:-1])
