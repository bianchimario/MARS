# Multivariate Asynchronous Random Shapelets (MARS)

Shapelets are time series subsequences that are maximally representative of a class [1]. <br>

MARS' shapelets are:
+ Multivariate: shapelets have the same number of dimensions of the time series provided. The distance between a shapelet and a time series is calculated as the sum of the minimum distance for each dimension.
+ Asynchronous: (by default) the different dimension of the shapelets can be extracted from different timestamps and they are compared with each timestamp of the time series.
+ Random: shapelets are extracted randomly for the sake of computation time.

![alt text](img/crash_fp_shp_0.png)
![alt text](img/crash_fp_shp_1.png)
![alt text](img/crash_fp_shp_2.png)
![alt text](img/crash_tp_shp_0.png)
![alt text](img/crash_tp_shp_1.png)
![alt text](img/crash_tp_shp_2.png)  


## How to install
```
pip install git+https://github.com/bianchimario/MARS
```

## Requirements
+ numpy
+ scipy
+ random
+ awkward

## References
[1] Ye, Lexiang, and Eamonn Keogh. ‘Time Series Shapelets: A Novel Technique That Allows Accurate, Interpretable and Fast Classification’. Data Mining and Knowledge Discovery 22, no. 1–2 (January 2011): 149–82. https://doi.org/10.1007/s10618-010-0179-5. <br>
[2] [Awkward library](https://awkward-array.org/doc/main/)

