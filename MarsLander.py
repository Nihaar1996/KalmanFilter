import numpy as np
from numpy.random import randn
import numpy.linalg as npl
import matplotlib.pyplot as plt

#draw pictures....
#DoMarsGraphics(k);
class MarsLander:
	def __init__(self):
		self.Params={}
		self.XTrue = np.array([[],[]],np.float32)
		self.VehicleStatus={}
		self.Store={}
		self.z = np.zeros(6000)

		# %------ user configurable parameters -------------
		self.Params["StopTime"] = 600 #run for how many seconds (maximum)?
		self.Params["dT"] = 0.1; #% Run navigation at 10Hz
		self.Params["c_light"] = 2.998e8;
		self.Params["EntryDrag"] = 5;                       #% linear drag constant
		self.Params["ChuteDrag"] = 2.5*self.Params["EntryDrag"];      #% linear drag constant with chute open
		self.Params["g"] = 9.8/3;  #% assumed gravity on mars
		self.Params["m"] = 50;     #% mass of vehcile
		self.Params["RocketBurnHeight"] = 1000;  #% when to turn on brakes
		self.Params["OpenChuteHeight"] = 4000;   #%when to open chute
		self.Params["X0"] = 10000; #% true entry height 
		self.Params["V0"] = 0;     #% true entry velocity
		self.Params["InitialHeightError"] = 0; #% error on  entry height 
		self.Params["InitialVelocityError"] = 0; #% error on entry velocity
		self.Params["InitialHeightStd"] = 100;  #%uncertainty in initial conditions
		self.Params["InitialVelocityStd"] = 20; #%uncertainty in initial conditions
		self.Params["BurnTime"] = np.nan;
		self.Params["ChuteTime"] = np.nan;
		self.Params["LandingTime"] = np.nan;

		# initial vehicle condition at entry into atmosphere...
		self.VehicleStatus["ChuteOpen"] = 0
		self.VehicleStatus["RocketsOn"] = 0
		self.VehicleStatus["Drag"] = self.Params["EntryDrag"]
		self.VehicleStatus["Thrust"] = 0
		self.VehicleStatus["Landed"] = 0

		# process plant model (constant velcoity with noise in acceleration)
		self.Params["F"] = np.array([[1,self.Params["dT"]],[0, 1]],np.float32)

		# process noise model (maps acceleration noise to other states)
		self.Params["G"] = np.array([[(self.Params["dT"]**2)/2], [self.Params["dT"]]],np.float32)

		# actual process noise truely occuring - atmospher entry is a bumpy business
		# note this noise strength - not the deacceleration of the vehicle....
		self.Params["SigmaQ"] = 0.2 #ms^{-2}

		# process noise strength how much acceleration (varinace) in one tick
		# we expect (used to 'explain' inaccuracies in our model)
		# the 3 is scale factor (set it to 1 and real and modelled noises will
		# be equal
		self.Params["Q"] = (1.1*self.Params["SigmaQ"])**2 #(ms^2 std)

		# observation model (explains observations in terms of state to be estimated)
		self.Params["H"] = np.array([2/self.Params["c_light"], 0],np.float32)[np.newaxis]

		# observation noise strength (RTrue) is how noisey the sensor really is
		self.Params["SigmaR"] = 1.3e-7 #%(seconds) 3.0e-7 corresponds to around 50m error....

		# observation expected noise strength (we never know this parameter exactly)
		# set the scale factor to 1 to make model and reallity match
		self.Params["R"] = (1.1*self.Params["SigmaR"])**2

		# initial conditions of (true) world:
		self.XTrue = np.array([[self.Params["X0"]],[self.Params["V0"]]],np.float32)
		


	def DoWorldSimulation(self,k):
	# World simulatuon will give us the 'True state' x so that we can use it to simulate the senor measurements using z = Hx

		oldvel = self.XTrue[1,k];
		oldpos = self.XTrue[0,k];
		dT = self.Params["dT"];

		# %friction is a function of height
		cxtau = 500; # spatial exponential factor for atmosphere density)
		AtmospherDensityScaleFactor = (1-np.exp(-(self.Params["X0"]-oldpos)/cxtau) );
		c = AtmospherDensityScaleFactor*self.VehicleStatus["Drag"];

		# %clamp between 0 and c for numerical safety
		c = min(max(c,0),self.VehicleStatus["Drag"]);

		# %simple Euler integration
		acc = (-c*oldvel- self.Params["m"]*self.Params["g"]+self.VehicleStatus["Thrust"])/self.Params["m"] + self.Params["SigmaQ"]*randn(1);
		newvel = oldvel+acc*dT;
		newpos = oldpos+oldvel*dT+0.5*acc*dT**2;
		XConc = np.concatenate((self.XTrue,np.array([newpos,newvel])),axis=1)
		self.XTrue = XConc
		

	def GetSensorData(self,k):
		XTrue = self.XTrue[:,k][np.newaxis]
		zk =  np.dot(XTrue,(self.Params["H"]).transpose()) + self.Params["SigmaR"]* randn(1);
		z = self.z
		z[k] = zk[0,0]
		self.z = z
		return zk[0,0]

	def DoEstimation(self,XEst,PEst,z):

		F = self.Params["F"]
		G = self.Params["G"]
		Q = self.Params["Q"]
		R = self.Params["R"]
		H = self.Params["H"]

		# %prediction...
		XPred = F*XEst;
		PPred = F*PEst*F.transpose()+G*Q*G.transpose();

		# % prepare for update...
		Innovation = z-H*XPred;
		S = H*PPred*H.transpose()+R
		W = PPred*H.transpose()*npl.inv(S)

		# % do update....
		XEst = XPred+W*Innovation;
		PEst = PPred-W*S*W.transpose();
		return XEst,PEst,S,Innovation


	def DoControl(self,k,XEst):

		if (XEst[0,0]<self.Params["OpenChuteHeight"]) and (self.VehicleStatus["ChuteOpen"]==0):
		    # %open parachute:
		    self.VehicleStatus["ChuteOpen"] = 1;
		    self.VehicleStatus["Drag"] = self.Params["ChuteDrag"];
		    print('Opening Chute at time %f\n',k*self.Params["dT"]);
		    self.Params["ChuteTime"] = k*self.Params["dT"];


		if(XEst[0,0]<self.Params["RocketBurnHeight"]):
			if(~self.VehicleStatus["RocketsOn"]):
				print('Releasing Chute at time %f\n',k*self.Params["dT"])
				print('Firing Rockets at time %f\n',k*self.Params["dT"])
				self.Params["BurnTime"] = k*self.Params["dT"];
				#turn on thrusters    
				self.VehicleStatus["RocketsOn"] = 1
			    #drop chute..
				self.VehicleStatus["Drag"] = 0
			    #simple littel controller here (from v^2 = u^2+2as) and +mg for weight of vehicle
			    # sets/controls the thrust based on current height and vel estimates (XEst)
				self.VehicleStatus["Thrust"] = (self.Params["m"]*XEst[1]**2-1)/(2*XEst[0])+0.99*self.Params["m"]*self.Params["g"];

		if(XEst[0,0]<1):
		    # %stop when we hit the ground...
		    print('Landed at time %f\n',k*self.Params["dT"]);
		    self.VehicleStatus["Landed"] = 1;
		    self.Params["LandingTime"] = k*self.Params["dT"];
		    

	# def DoStore(self,k,XEst,PEst,Innovation,S,z):
	# if(k==1):
	#     self.Store.XEst  = XEst;
	#     self.Store.PEst  = diag(PEst);
	#     self.Store["Innovation"] = Innovation;
	#     self.Store["S"] = S;
	#     self.Store["z"] = z;
	# else:
	#     self.Store["XEst"] = [self.Store["XEst"] XEst];
	#     self.Store["PEst"]  = [self.Store["PEst"] np.diag(PEst)];
	#     self.Store["Innovation"] = [self.Store["Innovation"] Innovation];
	#     self.Store["S"] = [Store.S diag(S)];
	#     self.Store["z"] = [Store.z z];


def SimulateMarsLander():
	# Set up the parameters
	mldr = MarsLander() 

	# Initial conditions of the estimator (note: InitialHeightError etc. is a key value in parameters dict)
	XEst = [mldr.Params["X0"] + mldr.Params["InitialHeightError"], mldr.Params["V0"] + mldr.Params["InitialVelocityError"]]
	PEst = np.diag([mldr.Params["InitialHeightStd"]**2, mldr.Params["InitialVelocityStd"]**2])

	# # Store initial conditions (TODO)
	# DoStore(1,XEst, PEst, [0],[0],np.nan)

	k =0
	while(mldr.VehicleStatus["Landed"] == 0 and k < mldr.Params["StopTime"]/mldr.Params["dT"]):
            
	    # simulate the world
	    XTrue = mldr.DoWorldSimulation(k)
	                
	    # read from sensor
	    zk = mldr.GetSensorData(k)
	        
	    # estimate the state of the vehicle
	    XEst,PEst,S,Innovation = mldr.DoEstimation(XEst,PEst,zk)
	        
	    # make decisions based on our esimated state
	    mldr.DoControl(k,XEst)
	                  
	    # # store results 
	    # DoStore(k,XEst,PEst,Innovation,S,z[k])
	    plt.plot(k,Innovation[0,0],'bo')    
	    # tick...
	    k = k+1
	# x = np.arange(0,6000)
	# plt.plot(x,XEst)
	plt.show()


SimulateMarsLander()

