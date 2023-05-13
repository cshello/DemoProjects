



## 编译
```sh  
mkdir build && cd build
cmake ..
cmake --build . 
./main 0.jpg


```


## 测试结果
```sh
root@c demo02/build# ./main 9.jpg
[W TensorImpl.h:1156] Warning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (function operator())
out: -1.3237 -7.3724  0.3255 -6.0024 -3.0841 -6.0802 -4.0039 -2.0330 -0.2698  8.0463
[ CPUFloatType{1,10} ]
result: 9
```
