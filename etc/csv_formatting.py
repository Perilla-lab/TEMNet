import csv, re, copy, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.preprocessing.image import img_to_array, load_img


def parse_region_data(csvname):
  print("input_pypeline: parsing data from file", csvname)
  idx = []
  x   = []
  y   = []
  w   = []
  h   = []
  lab = []
  with open(csvname, newline='') as labels:
    fields = ['#filename',
              'file_size',
              'file_attributes',
              'region_count',
              'region_id',
              'region_shape_attributes',
              'region_attributes']
    reader = csv.DictReader(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    for row in reader:
      lt = row['region_attributes']
      lt = str(re.sub(r'([{}"])','',lt))
      lt = lt.split(':')
      if len(lt) == 2:
        lab.append(lt[1])
      idx.append(row['region_id'])
      tmp = row['region_shape_attributes']
      tmp = str(re.sub(r'([{}"])','',tmp))
      tmp = tmp.split(',')
      for t in tmp:
        tt = t.split(':')
        if len(tt) == 1:
          continue
        else:
          if tt[0] == 'x':
            x.append(int(tt[1]))
          elif tt[0] == 'y':
            y.append(int(tt[1]))
          elif tt[0] == 'height':
            h.append(int(tt[1]))
          elif tt[0] == 'width':
            w.append(int(tt[1]))
  idx.remove('region_id')
  print("input_pypeling: parsing region data with length (number of GT boxes) ", len(x))
  return idx, lab, x, y, w, h


def write_region_data(csvname, idx, lab, x, y, w, h, max_height, max_width):
  #Set all boxes outside our range to zero
  local_w = copy.deepcopy(w)
  local_h = copy.deepcopy(h)
  print(f"x:\n {x}")
  local_w[x+w>max_width]=0
  local_w[x<0]=0
  print(f"w:\n {w}")
  print(f"y:\n {y}")
  local_h[y+h>max_height]=0
  local_h[y<0]=0
  print(f"h:\n {h}")
  print("input_pypeline: writing data to file", csvname)
  with open(csvname, mode='w', newline='') as labels:
    fields = ['#filename',
              'file_size',
              'file_attributes',
              'region_count',
              'region_id',
              'region_shape_attributes',
              'region_attributes']
    writer = csv.DictWriter(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    
    writer.writeheader()
    n_region=0
    for i in range(len(x)):
      if (local_h[i]!=0 and local_w[i]!=0):
        n_region+=1
        writer.writerow({'#filename': csvname,
                       'file_size': 'irrelevant', 
                       'file_attributes': '{}',
                       'region_count': len(x),
                       'region_id': idx[i],
                       'region_shape_attributes': '{"name":"rect","x":'+str(x[i])+',"y":'+str(y[i])+',"width":'+str(w[i])+',"height":'+str(h[i])+'}',
                       'region_attributes':'{"particle_class":'+lab[i]+'}'
                       })
  print(f"input_pypeling: done writing region data with length (number of GT boxes) {n_region} to {csvname}")

def read_region_data(csvname):
  print("input_pypeline: parsing data from file", csvname)
  idx = []
  x   = []
  y   = []
  w   = []
  h   = []
  lab = []
  with open(csvname, newline='') as labels:
    fields = ['',
              'Label',
              'Area',
              'BX',
              'BY',
              'Width',
              'Height']
    reader = csv.DictReader(labels,
      fieldnames=fields,
      dialect='excel',
      quoting=csv.QUOTE_MINIMAL)
    for i, row in enumerate(reader):
      if i == 0: # Skip the header line
        continue
      x.append(int(row['BX']))
      y.append(int(row['BY']))
      w.append(int(row['Width']))
      h.append(int(row['Height']))
      lab.append(row['Label'])
      idx.append(i)
  print("input_pypeling: parsing region data with length (number of GT boxes) ", len(x))
  return idx, lab, x, y, w, h

def visualize_parsed_boxes(val_path): 
    image_ids_val = next(os.walk(val_path))[1] 

    for i in range(len(image_ids_val)):
        _id, lab, x, y, w, h = parse_region_data(val_path +'/'+ image_ids_val[i] + '/region_data_' + image_ids_val[i] + '.csv')
        fig, axes = plt.subplots(dpi = 100, figsize=(20, 13))
        plt.axis('off')
        imgData = np.uint8(img_to_array(load_img(val_path +'/'+ image_ids_val[i] + '/' + image_ids_val[i] + '.png')))
        axes.imshow(imgData)
        boxcount = 0
        for n, h, z, t, caption in zip(x, y, w, h, lab):
            axes.text(n, h, '{}'.format(caption), fontsize=15, color='w', backgroundcolor='none')
            if caption=='mature':
                color = 'r'
            elif caption=='immature':
                color='y'
            else:
                color ='g'
            rect = patches.Rectangle((n, h), z, t, linewidth=3, alpha=0.7, linestyle='dashed',edgecolor=color,facecolor='none')
            axes.add_patch(rect)
            boxcount += 1
        fig.savefig(val_path+'/'+image_ids_val[i]+"/BBOXES_VIS_" + image_ids_val[i] + ".png", bbox_inches = 'tight', pad_inches = 0.5)
        plt.close(fig)

if __name__ == '__main__' :
    # IMGS_PATH = '/home/gorzy/Downloads/new_dataset_formated'
    IMGS_PATH = '/scratch/07655/jsreyl/imgs/new_dataset_formated_light'
    img_ids = next(os.walk(IMGS_PATH))[1]#All folder names in TRAIN_PATH
    csv_paths = [os.path.join(IMGS_PATH, img_name, 'Results.csv') for img_name in img_ids]
    for csv_path, img_name in zip(csv_paths, img_ids):
        idx, lab, x, y, w, h = read_region_data(csv_path)
        write_path = os.path.join(IMGS_PATH, img_name, 'region_data_'+img_name+'.csv')
        write_region_data(write_path, np.array(idx), lab, np.array(x), np.array(y), np.array(w), np.array(h), 2620, 4000)
        #viz
    visualize_parsed_boxes(IMGS_PATH)
