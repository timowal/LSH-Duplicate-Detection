# Duplicate Product Detection

This code implements minhashing, LSH and agglomerative single linkage clustering to determine which products are duplicates of each other without knowing the true model IDs of the products.

To get started run the "main.py" script. In the main class to change the 'bootstrapAmount' (default = 20) and 'shingleLength' (default = 3) variables if you wish to use different settings.

For the creation of model word representations of products an extension is added to also include the brand name of products alongside their titles for the construction of model word representations. If you wish to use the normal method where only the title is used, change "includeBrand = True" to "includeBrand = False" on line 52 of the code.

The used data can be downloaded from https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip. Then create a folder 'Data' in your python project and put the downloaded JSON file in this folder.