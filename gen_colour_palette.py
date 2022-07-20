import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


pattern_len = 250
palette_len = 15000

# <divisionPoint position='0' color='1;1;1'/>
#    <divisionPoint position='0.15' color='1;0.8;0'/>
#    <divisionPoint position='0.33' color='0.53;0.12;0.075'/>
#    <divisionPoint position='0.67' color='0;0;0.6'/>
#    <divisionPoint position='0.85' color='0;0.4;1'/>
#    <divisionPoint position='1' color='1;1;1'/>

color_palettes = {}
color_palettes["seashore"] = [[0,[0.7909,0.9961,0.763]], [0.1667,[0.8974,0.8953,0.6565]], [0.3333 ,[0.9465, 0.3161, 0.1267]], [0.5 ,[0.5184, 0.1109, 0.0917]], [0.6667 ,[0.0198, 0.4563, 0.6839]], [0.8333 ,[0.5385, 0.8259, 0.8177]], [1 ,[0.7909, 0.9961, 0.763]]]
color_palettes["hot_and_cold"] = [[0,[1,1,1]], [0.16,[0,0.4,1]], [0.5 ,[0.2, 0.2, 0.2]], [0.84 ,[1, 0, 0.8]], [1 ,[1, 1, 1]]]


stop_points = color_palettes["seashore"]

for i,point in enumerate(stop_points):
  stop_points[i][0] = int(point[0]*pattern_len)

print("stop_points : ", stop_points)
pattern = []
cum_gen = 0
for i in range(1, len(stop_points)):
  num_points = stop_points[i][0] - stop_points[i-1][0]

  if(i==(len(stop_points)-1)):
    num_points = pattern_len - cum_gen

  cum_gen += num_points

  for j in range(num_points):
    prev_r = stop_points[i-1][1][0]
    next_r = stop_points[i][1][0]

    curr_r = prev_r + (next_r - prev_r)*j/num_points
    r = curr_r

    prev_g = stop_points[i-1][1][1]
    next_g = stop_points[i][1][1] 
    curr_g = prev_g + (next_g - prev_g)*j/num_points
    g = curr_g

    prev_b = stop_points[i-1][1][2]
    next_b = stop_points[i][1][2] 
    curr_b = prev_b + (next_b - prev_b)*j/num_points
    b = curr_b

    pattern.append(np.array([r,g,b]))

print("pattern size",len(pattern))
def plot_colormap(pattern):
  colormap = ListedColormap(pattern)
  a = np.array([[0,1]])
  plt.figure(figsize=(9, 1.5))
  img = plt.imshow(a, cmap=colormap)
  plt.gca().set_visible(False)
  cax = plt.axes([0.1, 0.2, 0.8, 0.6])
  plt.colorbar(cax=cax)
  plt.show()

def dump_palette(pattern):
  offset = 0 #int(0.3*pattern_len)
  with open("color_palette.h", 'w') as fp:
    fp.write("#define GRADIENTLENGTH %d\nunsigned char colors[GRADIENTLENGTH][3] ={\n"%(palette_len))
    i = 0
    pattern2 = [0]*len(pattern)
    for idx in range(offset, len(pattern)):
      pattern2[i] = pattern[idx]
      i+=1
    for idx in range(offset):
      pattern2[i] = pattern[idx]
      i+=1

    for idx in range(len(pattern)):
      pattern[idx] = pattern2[idx]

    i = 0
    while(1):
      print("i: ",i,", palette_len: ",palette_len)
      for colour in pattern:
        fp.write("{"+hex(int(colour[0]*255))+", ")
        fp.write(hex(int(colour[1]*255))+", ")
        fp.write(hex(int(colour[2]*255))+"}")
        i+=1
        if (i>=palette_len):
          break
        fp.write(",\n")
      if(i>=palette_len):
        break
    fp.write("};")

dump_palette(pattern)
# plot_colormap(pattern)