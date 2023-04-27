import os
data_dir = "/Users/sbhar/Riju/TorchDemoCode/TrinaCode/bitmojis"
count = 0
index = 189
for i,path in enumerate(os.listdir(data_dir)):
    if os.path.isfile(os.path.join(data_dir,path)):
        if i == 189:
            print(os.path.join(data_dir,path))
        count = count + 1

print(count)