Run the spatial_analysis_manchester.ipynb for spatial analysis. The test.shp file contains data needed for conducting spatial analysis of Greater Manchester results.<br /><br />
Due to limited upload volume, satellite images of Manchester is provided through Googld drive. You can contact me to obtain more datasets for testing.
<br /><br />
Download link of the ResNet (Sentinel-2) model's fine-tuned weigts: https://drive.google.com/file/d/1AcU4FpGnUKy5YDxMJ4jcUVHalrws9Eaq/view?usp=sharing
<br /><br />
Download link of the ResNet (ImageNet) model fine-tuned weigts: https://drive.google.com/file/d/1elh71RtkyQuNVULAeJ8NMx41wbonHDtV/view?usp=sharing
<br /><br />
Download link of the VGG-A (ImageNet) mdoel fine-tuned weights:https://drive.google.com/file/d/11KTOqol4LdilRdsxO1ZD7MWG9TK6n_vm/view?usp=sharing
<br /><br />
Download link of the VGG-D (ImageNet) mdoel fine-tuned weights:https://drive.google.com/file/d/18dgHxwo_A-HhCoT75AzJm_mbGNYJt68b/view?usp=sharing
<br /><br />
<br /><br />
<br /><br />
Testing procedures <br />
1: Download a weight file.<br />
2: Download Manchester images into a folder called validation_raster_Manchester<br />
3: Download csv file pop_label_validation2_manchester.csv that contains image paths and labels which are used after prediction is made to find the accuracy<br />
4 Download csv file pop_label_a3_testing.csv that contains definition of each class
5: Download dataset_generator.py into the working directory<br />
6: Run the python script infer.py to load images and the model
