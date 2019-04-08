# Human-Detector
Binary Classifier for Human Detection in an Image

### Environment:
  1. Python 3
  2. Numpy
  3. OS: Linux ( Ubuntu 18 )
  4. Keras
  
### Approach:
  1. Used INRIA dataset
  2. Used CNN model
  
### CNN Architecture
  1. 3x3 Conv2D,32
  2. Maxpooling
  3. 3x3 Conv2D,32
  4. Maxpooling
  5. 3x3 SeparableConv2D,32
  6. Maxpooling
  7. Flatten
  6. Dense,128
  7. Dense,1
