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
Clone and copy test data to build folder:
```
cd ../../
git clone https://github.com/nqoptik/computer_vision_data.git
cd electricity_bill_detection/build/
cp -r ../../computer_vision_data/electricity_bill_detection/build/* .
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
./electricity_bill_detection
```
