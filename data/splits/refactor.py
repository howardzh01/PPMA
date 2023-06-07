
      
def refactor_file(f, x, y):
    l = f.readlines()
    
    # acts as a counter to know the
    # index of the element to be replaced
    c = 0
    for i in l:
        print(l)
        if x in i:
            # Replacement carries the value
            # of the text to be replaced
            Replacement = i.replace(x, y)
    
            # changes are made in the list
            l = Replacement
        c += 1
    
    # The pre existing text in the file is erased
    # f.truncate(0)
    
    # the modified list is written into
    # the file thereby replacing the old text
    # f.writelines(l)
    # f.close()
    print("Text successfully replaced")

 
import os
for root, dirs, files in os.walk(".", topdown=False):
   for name in files:
      print(os.path.join(root, name))
      if '.csv' in name:
          x = '/gpfs/u/home/DPLD/DPLDpndr/scratch-shared/datasets/'
          y=''
          with open(os.path.join(root, name), 'r+') as f:
            refactor_file(f, x, y)
        #   x='/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/'
        #   y=''
        #   with open(os.path.join(root, name), 'r+') as f:
        #     refactor_file(f, x, y)
            raise Exception