mat = []
for i in range(13):
	mat.append([0] * 13)

 
truck =       [0, 10, 9, 8, 7, 3, 5, 6, 0]
blue_drone =  [(0,11,9), (3,1,5)]
black_drone = [(10,12,9), (7,4,3), (3,2,5)]

for (x,y) in zip(truck, truck[1:]):
	mat[x][y] = 1

for (x,y,z) in blue_drone:
	mat[x][y] = 2
	mat[y][z] = 2

for (x,y,z) in black_drone:
	mat[x][y] = 3
	mat[y][z] = 3

print(0,list(range(13)))
[print(i,j) for i,j in enumerate(mat)]