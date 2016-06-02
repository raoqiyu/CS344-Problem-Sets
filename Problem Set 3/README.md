### Problem Set 3
####Prefix Sum

----

##### Hillis Steele
![](http://http.developer.nvidia.com/GPUGems3/elementLinks/39fig02.jpg)  
```
for d = 1 to log2 n do 
	for all k in parallel do 
		if k >= 2^d  then 
			x[out][k] = x[in][k – 2^(d-1)] + x[in][k]
		else 
			x[out][k] = x[in][k]
```
----

##### Blelloch
Blelloch, Guy E. "Prefix sums and their applications." (1990).  
Reduce & Down-Sweep  
![](http://http.developer.nvidia.com/GPUGems3/elementLinks/39fig03.jpg)  
 An Illustration of the Reduce.  
 ```
for d = 0 to log2 n – 1 do 
	for all k = 0 to n – 1 by 2^(d+1) in parallel do 
		x[k +  2^(d+1) – 1] = x[k +  2^d  – 1] + x[k +  2^(d+1) – 1]
 ```
![](http://http.developer.nvidia.com/GPUGems3/elementLinks/39fig04.jpg)  
 An Illustration of the Down-Sweep.  
 ```
x[n – 1] := 0
for d = log2n – 1 down to 0 do 
	for all k = 0 to n – 1 by 2 d +1 in parallel do 
		t = x[k +  2 d  – 1]
		x[k +  2^d  – 1] = x[k +  2^(d+1) – 1]
		x[k +  2^(d+1) – 1] = t +  x[k +  2^(d+1) – 1]
 ```
 
 Figs and pseudocode  above are from [GPU Gems 3: Chapter 39. Parallel Prefix Sum (Scan) with CUDA](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html)
 ----
 
##### Reuce and Post Reduction Reverse
Heterogeneous Parallel Programming : Lecture-4-6-work-efficient-scan-kernel  
![](https://upload.wikimedia.org/wikipedia/commons/8/81/Prefix_sum_16.svg)
Fig from [Prefix sum](https://en.wikipedia.org/wiki/Prefix_sum)
