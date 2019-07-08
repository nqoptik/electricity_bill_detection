# Electricity bill detection
Find the electricity bill in the given image and warp it to a rectangle.

## Build project
Build project with cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Copy test data to build folder:
```
cp -r ../../../computer_vision_basics_data/electricity_bill/build/* .
```

Create output folders:
```
mkdir output_images_0
mkdir output_images_1
mkdir output_images_2
mkdir output_images_3
mkdir output_images_4
```

Run electricity bill warper:
```
./electricity_bill
```
