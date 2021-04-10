<p align="right">
<img src="https://img.shields.io/github/license/farhanfuadabir/Sensor-Data-Glove">
</p>
     
# Hand-Gesture-Recognition

The system can classify `14` static and `3` dynamic hand gestures in real-time. 

<a href="https://github.com/farhanfuadabir/Hand-Gesture-Recognition/blob/main/Presentation_Report/4-2_report.pdf">Detailed Project Report</a>

### Developer

- Farhan Fuad Abir
- Md. Ahasan Atick Faisal

### Datasets

- <a href="Datasets/Static Gestures/">Static Gestures</a>
- <a href="Datasets/Dynamic Gestures/">Dynamic Gestures</a>
- <a href="Datasets/ASL Alphabet/">ASL Alphabet</a> [Under development]
- <a href="Datasets/ASL Words/">ASL Words</a> [Under development]

### Dependencies

#### Python 3.5
- Numpy
- Pandas
- Pyserial
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

#### ESP-32
- <a href="ContinuousDataAcquisitionESP32/Dependencies/I2Cdevlib-Core_ID11/">I2Cdevlib-Core_ID11</a>
- <a href="ContinuousDataAcquisitionESP32/Dependencies/I2Cdevlib-MPU6050_ID107/">I2Cdevlib-MPU6050_ID107</a> 

## Gestures

<p align="center">
<br>
<img src="Figures/Gestures.png" width="550">
<br>
</p>

## Dataglove

It contains 5 flex sensors and 1 `MPU6050` IMU sensor. A `ESP32`-based module is used as the processing unit.

<p align="center">
<br>
<img src="Figures/dataglove.png" width="450">
<br>
</p>

## Desktop Interface

It is developed using `Processing`. 

<p align="center">
<br>
<img src="Figures/software_interface.png" width="800">
<br>
</p>

The graphs show the sensor data in real-time. During `Train` mode, the software records sensor data in a `csv` file.

## Video Demostration

<p align="center">
  <a href="https://www.youtube.com/watch?v=JfePNcJKkbE">
    <img src="Figures/thumbnail.png" width="450"/>
  </a>
</p>

## License

This work is licensed under [MIT License](Hand-Gesture-Recognition/LICENSE).
