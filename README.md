# KalmanFilter
This is my implementation of a Kalman Filter in python from Paul Newman (our prof. at Oxford's) MATLAB code.

I've used python's object oriented features to ease the storage and management of the several variables.
The task is to estimate the position and velocity of a rocket while it descends in mars' atmosphere.

Notes:
The plant model (F) is of constant velocity (noise in acceleration).

The state consists of position and velocity, however, there is no explicit calculation of velocity done anywhere. The observation
model (H) only relates the position with the measurement from the sensor (which is time of flight). Thus, velocity is being
inferred from the Covariance between the 2 states. 

The filter tuning involves changing the Process noise matrix (Q) which is the uncertainty in our prediction model. It dilutes
the covariance matrix P in each prediction step but reducing it too much will mean we have overconfidence in our predictions 
and the update step will hold less importance. This is a trade-off and we need to tune this matrix.
