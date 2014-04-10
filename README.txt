history
kaori-secode-calker:    optimizied for bow model, support cross-validation for gamma value of Chi-square kernel
kaori-secode-calker-v2: tbu
kaori-secode-calker-v3: optimized for fisher vector, with linear svm. there is
a bug when loading test data (contains NaN). And another bug is mu value
calculated when testing.
kaori-secode-calker-v4:  fix bug NaN
kaori-secode-calker-v5:  fix bug mu value calculated when testing
kaori-secode-calker-v6:  designed for MED 2013
kaori-secode-calker-v7:  designed for MED 2013, support different test sets
(to store kernels and scores for different test sets)
kaori-secode-calker-v7.1:  designed for MED 2013, load test data seperately
(not loading all), to support cal test kernel more efficiently
kaori-secode-calker-v7.2:   v7.2 removes short semgent (<10s) from trainning
< -- v7.1 modified calker_cal_map function at segment-level. Instead of using arrayfun, which is a fullly vectorized solution, using a for loop in this case can be faster....
< c.f. http://stackoverflow.com/questions/12522888/arrayfun-can-be-significantly-slower-than-an-explicit-loop-in-matlab-why
kaori-secode-calker-v7.3:  optimized for ad-hoc event 2013
kaori-secode-calker-v7.4:  working with near miss videos
kaori-secode-calker-v7.5:  working with attribute features
kaori-secode-calker-v8.0:  copied from v7.2, start testing different pooling techniques

File create database: only use for video-annotation