def bubblesort(l):
    
    for i in range(n):
        for j in range(n-1):
            if l[j]>l[j+1]:
                l[j],l[j+1]=l[j+1],l[j]



def selectionsort(l):
    for i in range(n):
        min=i
        for j in range(i+1, n):
            if l[min]>l[j]:
                min=j
    l[i],l[min]=l[min],l[i]

                    
l = [5, 6, 9, 10, 0, -1, -50, -360, 4]
n=len(l)
print("Enter your choice : \n"
      "1. Sort by bubble sort\n"
      "2. Sort by selection sort\n")
ch = input()
if ch=='1':
    bubblesort(l)
    
elif ch=='2':
    selectionsort(l)
    
print("The sorted array is : ")
for i in range(n):
    print(l[i])





