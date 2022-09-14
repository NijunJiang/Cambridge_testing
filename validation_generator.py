import rasterio.mask
import geopandas as gpd
import csv
import pandas as pd
import sys
import os

place = 'hart'

shape_file_path = 'data2020/' + place + '_pop_vectorised.shp'
tiff_file_path = 'data2020/' + place + '_validation_0.00009.tif'
validation_file_path = 'validation_raster_' + place + '/'
csv1_path = 'pop_label_validation_' + place + '.csv'
csv2_path = 'pop_label_validation2_' + place + '.csv'
csv3_path = 'pop_label_validation3_' + place + '.csv'  # translation table
folder_exist = os.path.exists(validation_file_path)
if not folder_exist:
    os.makedirs(validation_file_path)
    print("The new validation file path created")


print('loaded1')
shape = gpd.read_file(shape_file_path)

print('loaded2')
shape.crs = "epsg:4326"
shape = shape.to_crs(epsg = 4326)

print('loaded3')
src = rasterio.open(tiff_file_path)

print('loaded4')
out_meta = src.meta

file_list = []


for i in range(len(shape.index)):
    shape_trial = shape.loc[i, 'geometry']
    out_image, out_transform = rasterio.mask.mask(src, [shape_trial], crop=True)
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    raster_id = str(i)
    name = place
    name = name + raster_id
    filename = "%s.tif" % name
    file_list.append(filename)
    print(raster_id)
    with rasterio.open(validation_file_path + filename, "w", **out_meta) as dest:
        dest.write(out_image)

with open(csv1_path, 'w+', newline='') as f:
    f.truncate()
    writer = csv.writer(f)
    rows = zip(file_list, list(shape['value']))
    for row in rows:
        writer.writerow(row)

df = pd.read_csv(csv1_path, header=None)
array_pop = df[df.columns[1]]
binned_pop = pd.cut(array_pop, bins=[0,5,10,50,100,500,1000,2000,3000,15000], right=False)
binned_pop_wLabel = pd.cut(array_pop, bins=[0,5,10,50,100,500,1000,2000,3000,15000], right=False, labels=[*range(0, 9, 1)])

binned_pop_list_lower = []
binned_pop_list_upper = []
for i in binned_pop:
    binned_pop_list_lower.append(i.left)

for i in binned_pop:
    binned_pop_list_upper.append(i.right)

df[1] = binned_pop_list_lower
df[2] = binned_pop_list_upper
df['true'] = array_pop

with open(csv2_path, 'w+', newline='') as f:
    f.truncate()
    writer = csv.writer(f)
    rows = zip(list(df[0]), binned_pop_wLabel, list(df[1]), list(df[2]), list(shape['geometry']), df['true'])
    for row in rows:
        writer.writerow(row)
df = pd.read_csv(csv2_path, header=None)


decoding_list = []
class_list = []
for i in range(9):
    count = 0

    decoding_one = []
    for j, k in zip(df[1], df[5]):
        class_list.append(j)
        if i == j:
            decoding_one.append(k)
            count += 1
    if count != 0:
        decoding_list.append(sum(decoding_one)/count)

print(decoding_list)

with open(csv3_path, 'w+', newline='') as f:
    f.truncate()
    writer = csv.writer(f)
    rows = zip(list(set(class_list)), decoding_list)
    for row in rows:
        writer.writerow(row)