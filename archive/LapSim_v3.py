import numpy as np
import matplotlib.pyplot as plt
from car import Car

class LapSim:
    '''
    A lap time simulator (point-mass) for 2D tracks
    Forward integration until losing traction
    Backward integration from next apex to find brake point

    energy consumption (endurance)
        split between gas and battery
    powertrain
        maximum acceleration/power output

    Change gears when rpm exceeds range

    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.g = 9.81                                       # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)                # number of discretized points
        
        self.pts = kwargs.pop('pts',0)                  # input track data
        self.pts_interp = kwargs.pop('pts_interp',0)    # interpolated track data
        self.track_len = kwargs.pop('track_len',0)      # total track length

        self.ds = kwargs.pop('ds',0)                    # differential arc length
        self.r = kwargs.pop('r',0)                      # radius of curvature
        self.apex = kwargs.pop('apex',0)                # apex points location
        self.brake = kwargs.pop('brake',0)              # brake points location
        self.v = kwargs.pop('v',0)                      # velocity at each discretized point on track

        self.time = kwargs.pop('time',0)                # total lap time

        self.car = kwargs.pop('car',0)                  # Car class


    @classmethod
    def init_ellipse(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.pop('resolution',10)               # resolution of initial track data

        # input track data
        s = np.linspace(0,2*np.pi,res,endpoint=False)
        pts = np.vstack((300*np.cos(s),200*np.sin(s)))

        return cls(pts=pts, **kwargs)


    @classmethod
    def init_data(cls, **kwargs):
        '''
        Reat track data from file

        '''
        track_data = kwargs.pop('track_data',0)         # track data file name

        import pandas as pd

        df1 = pd.read_excel(track_data)                 # read track data file
        X = df1['X'].values
        Y = df1['Y'].values
        # Z = df1['Z'].values

        # pts = np.vstack((X,Y,Z))
        pts = np.vstack((X,Y))
        
        return cls(pts=pts, **kwargs)


    def lap_time(self):
        '''
        Calculates lap time
        '''

        # interpolate equidistant points on the track
        self.pts_interp, self.ds, self.track_len = self.discretize()

        # calculate radius of curvature
        self.dpds, self.d2pds2, self.r = self.roc()

        # find apex locations
        self.apex = self.find_apex()

        # calculate traction-limited velocity at each point
        self.v, self.gear, self.energy, self.time_list = self.get_velocity_list()

        # find brake points
        self.brake = self.find_brake_pts()

        self.plot_discretized_points(apex=0, brake=0, elevation=0)            # check apex location

        # calculate lap time
        self.time = np.sum(self.time_list)

        self.plot_velocity(apex=0)

        return 1   


    def discretize(self):
        
        # Parametrize track by variable s. Assume that the input points are ordered and arbitrarily spaced.
        # Parametrization with respect to normalized arc length
        diff = np.roll(self.pts,-1,axis=1)-self.pts
        arclen = np.linalg.norm(diff,axis=0)   # length of displacement vector
        track_len = np.sum(arclen)
        s = np.cumsum(arclen)/track_len
        self.pts = self.pts/track_len*1000      # normalize track data to 1km total track length

        # periodic boundary condition (x(0) == x(1), y(0) == y(1))
        s = np.append(0,s)
        self.s = s
        snew = np.linspace(0,1,num=self.steps, endpoint=False)
        dim = len(self.pts)
        pts_interp = np.zeros((dim, self.steps))

        from scipy.interpolate import interp1d

        for i in np.arange(dim):
            x = np.append(self.pts[i],self.pts[i,0])
            fx = interp1d(s,x,kind='cubic', fill_value='extrapolate')
            xnew = fx(snew)
            pts_interp[i] = xnew
        
        if dim == 3:

            xdiff = pts_interp[0] - np.roll(pts_interp[0],1,axis=0)
            ydiff = pts_interp[1] - np.roll(pts_interp[1],1,axis=0)
            zdiff = pts_interp[2] - np.roll(pts_interp[2],1,axis=0)
            self.elevation = np.arctan2(zdiff,np.sqrt(xdiff**2+ydiff**2))
        else:
            self.elevation = np.zeros(len(pts_interp[0]))

        ds = 1000/self.steps

        return pts_interp, ds, 1000

    
    def roc(self):
        '''
        Calculates radius of curvature at each point
        r(s) = |(dxds^2+dyds^2)^(3/2)/(dxds*dyds2-dyds*dxds2)|
        '''

        diff = ((self.pts_interp - np.roll(self.pts_interp,1,axis=1))+(np.roll(self.pts_interp,-1,axis=1)-self.pts_interp))/2
        dpds = diff[:2]/self.ds

        diff2 = ((dpds - np.roll(dpds,1,axis=1))+(np.roll(dpds,-1,axis=1)-dpds))/2
        d2pds2 = diff2[:2]/self.ds

        num = np.linalg.norm(dpds,axis=0)**3
        den = np.absolute(np.cross(dpds,d2pds2,axis=0))
        r = num/den

        r[np.isnan(r)] = np.inf

        r = (r+np.roll(r, 1, axis=0) + np.roll(r, -1, axis=0))/3
        
        return dpds, d2pds2, r

    
    def find_apex(self):
        '''
        finds cornering apex list: look for sign change in dr
        shift the arrays such that the discretization starts with an apex

        '''

        dr = self.r - np.roll(self.r,1,axis=0)
        sign = np.sign(dr)
        sign_flip = sign - np.roll(sign,-1,axis=0)

        s = np.arange(self.steps)
        apex = np.where((sign_flip == np.min(sign_flip)))
        # apex = np.where(self.r[apex]<np.median(self.r))

        # apex_min = np.argmin(self.r)
        idx_0 = self.r.shape[0]-apex[0][0]
        idx = np.arange(self.r.shape[0]) - idx_0

        # re-indexing to make apex the first point
        apex = apex-apex[0][0]
        self.r = self.r[idx]
        self.pts_interp = self.pts_interp[:,idx]

        return apex


    def get_velocity_list(self):
        '''
        Calculates traction-limited velocity at each point
        m*ap = mv^2/r
        a = sqrt(ap^2+at^2)
        a = mu * g
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for increasing roc
        Repeat calculation until losing traction, then jump to the next apex and integrate backwards to find the brake point.
        
        Get speed -> gear map for every powertrain
            ignore shift time
        '''

        self.car.alim = self.g * self.car.mu                            # might want to split lateral/longitudinal traction limit
        energy_list = np.zeros((self.steps,2))
        v, gear = self.v_apex()                                         # velocity and gear at apex
        time = np.zeros(self.steps)

        i = 0
        apex_idx = 0
        state = 'f'
        gear[0] = 1

        # get velocity list
        while i<self.steps:
            if state == 'f':                                                        # forward
                if v[np.remainder(i+1, self.steps)]==0:
                    ap = v[i]**2/self.r[np.remainder(i+1, self.steps)]#*np.cos(self.elevation[i])-self.car.m*self.g*np.sin(self.elevation[i])
                    if self.car.alim>ap:                                                # below traction limit
                        i1 = np.remainder(i+1, self.steps)                           # step forward
                        v[i1], gear[i1], energy_list[i1],time[i1]= self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]),roc=self.r[np.remainder(i+1, self.steps)])
                        i = i1
                    else:                                                           # traction is lost
                        state = 'b'
                        apex_idx= np.remainder(apex_idx+1, len(self.apex[0]))
                        print('losing traction, jumping to apex '+str(apex_idx+1), ' at i=',self.apex[0][apex_idx], ', current i=',i)
                        i = self.apex[0][apex_idx]
                else:                                                               # check if velocity at next apex can be achieved with the current gear
                    if np.min(v)==0:                                                  # reaching an apex without braking
                        i = np.remainder(i+1, self.steps)
                        apex_idx = np.remainder(apex_idx+1, len(self.apex[0]))
                    else:
                        print('reached end of track')
                        break
            elif state == 'b':                                                  # backward
                ap = v[i]**2/self.r[i-1]#*np.cos(self.elevation[i])-self.car.m*self.g*np.sin(self.elevation[i])
                if v[i-1]==0:                                                   # if velocity is not yet calculated
                    v[i-1], gear[i-1], energy_list[i-1], time[i-1] = self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]),roc=self.r[np.remainder(i-1, self.steps)])
                    i-=1
                else:                                                           # if velocity is calculated from forward integration
                    if self.car.alim<ap:                                        # if the previous point is an apex; loosing traction
                        print('losing traction (back), start integrating forward from apex '+str(apex_idx+1))
                        state = 'f'
                        i = self.apex[0][apex_idx]
                    else:                                                       # if still can accelerate
                        vb, gearb, energyb, timeb = self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]),roc=self.r[np.remainder(i-1, self.steps)])
                        if vb < v[i-1]:                                          # continue backward integration
                            v[i-1] = vb
                            energy_list[i-1] = energyb
                            time[i-1] = timeb
                            gear[i-1] = gearb
                            i-=1
                        else:                                                       # found brake point 
                            print('reached break point, start integrating forward from apex '+str(apex_idx+1))
                            state = 'f'
                            i = self.apex[0][apex_idx]

        energy_list, time = self.e_apex(v, gear, energy_list, time)
        dec = np.where(np. sign(v-np.roll(v,-1))==1)                 # decelerating points
        energy_list[dec[0]] = [0,0]

        return v, gear, energy_list, time

    def v_apex(self):

        v = np.zeros(self.steps)
        gear = np.zeros(self.steps)
        
        for i in self.apex[0]:
            v_trac = np.sqrt(self.car.mu * self.g * self.r[i])
            rpm0 = v_trac/(self.car.wheel_radius*0.0254*2*np.pi)*60
            if self.car.hybrid == 1:
                _, maxrpm, gear[i] = self.v_lim_hybrid(v_trac, 0, rpm0)
            else:
                _, maxrpm = self.v_lim_electric(v_trac, rpm0)
                gear[i] = 1
            v_rpm = maxrpm/60*(self.car.wheel_radius*0.0254*2*np.pi)
            v[i] = np.min([v_trac,v_rpm])
            
        return v, gear

    def e_apex(self, v=0, gear=0, energy_list=0, time=0):
        
        for i in self.apex[0]:
            a = (v[i+1]**2-v[i]**2)/(2*self.ds)
            time[i] = (v[i+1]-v[i])/a
            p_ICE = self.car.m * a * v[i] * (self.car.hybrid*self.car.power_split)
            p_EM = self.car.m * a * v[i] - p_ICE

            e_ICE = self.calc_fuel(gear[i], v[i], p_ICE, time[i])
            e_EM = p_EM*time[i]/(self.car.motor.eta/100)

            energy_list[i] = [e_ICE, e_EM]

        return energy_list, time


    def calc_velocity (self,vin=0,ap=0,gear=1, roc=0):

        # calculate rpm and check for shifting conditions
        rpm0 = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels at current velocity
        
        if self.car.hybrid == 1:
            a_tor, maxrpm, gear_new = self.v_lim_hybrid(vin, gear, rpm0)
        else:
            a_tor, maxrpm = self.v_lim_electric(vin, rpm0)
            gear_new = 1

        # torque-limited velocity [m/s]
        v_tor = np.sqrt(2*a_tor*self.ds+vin**2)    # v^2-vi^2 = 2a*ds
        t_tor = (v_tor-vin)/a_tor

        # traction-limited velocity [m/s]                   
        a_trac = np.sqrt(self.car.alim**2-ap**2)
        v_trac = np.sqrt(2*a_trac*self.ds+vin**2)  
        t_trac = (v_trac-vin)/a_trac                                   

        # lateral traction-limited velocity [m/s]
        v_trac_l = np.sqrt(self.car.alim*roc)
        a_trac_l = (v_trac_l**2-vin**2)/(2*self.ds)
        t_trac_l = (v_trac_l-vin)/a_trac_l 

        # rpm-limited velocity [m/s]
        v_rpm = maxrpm/60*(self.car.wheel_radius*0.0254*2*np.pi)
        a_rpm = (v_rpm**2-vin**2)/(2*self.ds)
        t_rpm = (v_rpm-vin)/a_rpm

        v = np.min([v_trac,v_tor,v_rpm,v_trac_l])
        if v == v_tor:
            t = t_tor
            p_ICE = self.car.m * a_tor * v * (self.car.hybrid*self.car.power_split)
            p_EM = self.car.m * a_tor * v - p_ICE
            print('Power limited. Velocity =',v, ', ICE Power [hp] =', str('{0:.2f}'.format(p_ICE)), 'EM Power [hp] =', str('{0:.2f}'.format(p_EM/745.7)))
        elif v == v_trac:
            t = t_trac
            p_ICE = self.car.m * a_trac * v * (self.car.hybrid*self.car.power_split)
            p_EM = self.car.m * a_trac * v - p_ICE
            print('Traction limited. Velocity =',v, ', ICE Power [hp] =', str('{0:.2f}'.format(p_ICE)), 'EM Power [hp] =', str('{0:.2f}'.format(p_EM/745.7)))
        elif v == v_trac_l:
            t = t_trac_l
            p_ICE = self.car.m * a_trac_l * v * (self.car.hybrid*self.car.power_split)
            p_EM = self.car.m * a_trac_l * v - p_ICE
            print('Lateral traction limited. Velocity =',v, ', ICE Power [hp] =', str('{0:.2f}'.format(p_ICE)), 'EM Power [hp] =', str('{0:.2f}'.format(p_EM/745.7)))
        elif v == v_rpm:
            t = t_rpm
            p_ICE = self.car.m * a_rpm * v * (self.car.hybrid*self.car.power_split)
            p_EM = self.car.m * a_rpm * v - p_ICE
            print('RPM limited. Velocity =',v, ', RPM at EM =', str('{0:.0f}'.format(maxrpm*self.car.motor.trans)), 'max EM RPM =', self.car.motor.maxrpm)

        e_ICE = self.calc_fuel(gear_new, v, p_ICE, t)
        e_EM = p_EM*t/(self.car.motor.eta/100)
        
        return v, gear_new, [e_ICE, e_EM], t
    

    def v_lim_hybrid(self, vin=0, gear=1, rpm0=0):
        '''
        Calculates velocity at the next discretized step (hybrid vehicles only)
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        ICE+EM
        '''
        
        # calculate rpm and check for shifting conditions
        r = 0.95                                             # set the max rpm
        rpm_list = rpm0*self.car.engine.trans[2:]*self.car.engine.trans[0]*self.car.engine.trans[1]   # rpm at all gears

        # calculate Power output
        if (gear == 1 and rpm_list[0]<self.car.engine.minrpm):
            rpm_at_gear_new = rpm_list[0]                                                  
            gear_new = gear
            p_ICE = self.car.engine.power(self.car.engine.minrpm)                                       # use constant extrapolation for v near 0        
        else:
            rpm_idx = np.where((self.car.engine.maxrpm*r>rpm_list) & (self.car.engine.minrpm<rpm_list))       # index of possible rpm
            if len(rpm_idx[0]) == 0:
                rpm_at_gear_new = self.car.engine.maxrpm
                gear_new = gear
            else:
                gear_new = rpm_idx[0][0]+1                                                     # gear chosen for next step
                rpm_at_gear_new = rpm_list[rpm_idx[0][0]]
            p_ICE = self.car.engine.power(rpm_at_gear_new)                                          # ICE power output after shifting                              

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_ICE = (rpm_at_gear_new/60)*(2*np.pi)                                           # angular velocity [rad/s] revolution per minute / 60s * 2pi
        if omega_ICE != 0:
            torque_ICE_at_wheel = (p_ICE*745.7/omega_ICE)*self.car.engine.trans[gear_new+1]  # always use maximum torque during acceleration
        else:
            torque_ICE_at_wheel = 0

        # torque limited acceleration
        torque_EM_at_wheel = self.car.motor.torque_max*1.356*self.car.motor.trans
        a_tor = (torque_EM_at_wheel+torque_ICE_at_wheel)/(self.car.wheel_radius*0.0254*self.car.m)
        
        # maxrpm determined by transmission
        wheel_maxrpm_ICE = self.car.engine.maxrpm/(self.car.engine.trans[gear_new+1]*self.car.engine.trans[0]*self.car.engine.trans[1])     
        wheel_maxrpm_EM = self.car.motor.maxrpm/self.car.motor.trans      
        maxrpm = np.min([wheel_maxrpm_EM,wheel_maxrpm_ICE])
        
        if gear != gear_new:
            print('Shifting...... Current gear:', gear_new)
        
        return a_tor, maxrpm, gear_new


    def v_lim_electric(self, vin=0, rpm0=0):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        EM only
        '''

        rpm = rpm0*self.car.motor.trans                         # rpm at motor
        omega = (rpm/60)*(2*np.pi)                              # angular velocity [rad/s]                       # angular velocity [rad/s] revolution per minute / 60s * 2pi
        
        # torque-limited velocity [m/s]
        torque_EM_at_wheel = self.car.motor.power_max*1.356*self.car.motor.trans
        a_tor = torque_EM_at_wheel/(self.car.wheel_radius*0.0254*self.car.m)               # torque-limited acceleration
        
        # rpm-limited velocity [m/s]
        maxrpm = self.car.motor.maxrpm/self.car.motor.trans

        return a_tor, maxrpm


    def calc_fuel(self, gear, v, Power, t):
        '''
        ONLY FOR HYBRID VEHICLES
        Calculates the total energy consumed at a discrete step
        ICE efficiency 2D interpolation of the fuel efficiency chart
        EM efficiency is assumed to be a constant
        '''

        if Power == 0:
            return 0

        rpm = v/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.engine.trans[int(gear)+1]*self.car.engine.trans[0]*self.car.engine.trans[1]   # rpm at current gear
        
        # calculate energy consumed from fuel efficiency
        x = rpm/60*2*np.pi                  # ICE angular velocity [rad/s]
        if x<self.car.engine.eta[0,0]:
            x = self.car.engine.eta[0,0]                        # for low v, use constant interpolation for fuel efficiency
        y = Power/x                                       # torque [Nm]
        if y<self.car.engine.eta[0,1]:
            y = self.car.engine.eta[0,1]                        # for low v, use constant interpolation for fuel efficiency
        from scipy.interpolate import griddata
        intmethod = 'cubic'
        eta = griddata(self.car.engine.eta[:,:2], self.car.engine.eta[:,2], (x,y), method=intmethod)

        e_ICE = Power*100/eta*t                 # energy consumed by ICE [J]

        if np.isnan(eta):
            print('WARNING: ICE speed and/or torque are outside of the interpolation range.')

        return e_ICE


    def find_brake_pts(self):
        '''
        Find brake points from velocity list
        '''

        v_diff = np.sign(self.v - np.roll(self.v, 1, axis=0))
        sign_flip = v_diff - np.roll(v_diff,-1,axis=0)
        brake = np.where(sign_flip == np.max(sign_flip))

        return brake


    def plot_discretized_points(self, apex=0, brake=0, elevation=0, index=0):
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cmx
        import matplotlib.colors

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        if elevation == 1:
            plt.subplots_adjust(right=0.85)
            cm = plt.get_cmap('plasma')
            cNorm = matplotlib.colors.Normalize(vmin = min(self.pts_interp[2]),vmax = max(self.pts_interp[2]))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            ax.scatter(self.pts_interp[0], self.pts_interp[1],c=scalarMap.to_rgba(self.pts_interp[2]),s=10, label='interpolation')
        else:
            plt.scatter(self.pts_interp[0], self.pts_interp[1],s=10,label='Interpolation')
        plt.scatter(self.pts[0],self.pts[1],s=2,label='Input')
        if index == 1:
            for i in range(len(self.pts[0])):
                ax.annotate(str(i)+'('+'{0:.3g}'.format(self.s[i]*100)+')',xy=(self.pts[0,i],self.pts[1,i]), xycoords='data')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='g',marker='^',label='apex')
        if brake==1:
            plt.scatter(self.pts_interp[0,self.brake],self.pts_interp[1,self.brake],c='r',marker='x',label='brake')
        plt.title('Discretized Track points (equidistant track length sampling)')
        plt.legend()
        if elevation == 1:
            cbaxes = fig.add_axes([0.87, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(scalarMap,cax=cbaxes)
            cbar.ax.set_ylabel('Elevation [m]')
        plt.draw()
        
        return 1

    
    def plot_velocity(self, apex=0):

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cmx
        import matplotlib.colors

        v = self.v*2.237                        # convert to [mph]
        fig2 = plt.figure(figsize=(8,6))
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(right=0.85)
        ax.set_aspect('equal')
        cm = plt.get_cmap('viridis')
        cNorm = matplotlib.colors.Normalize(vmin = min(v),vmax = max(v))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(self.pts_interp[0], self.pts_interp[1],c=scalarMap.to_rgba(v),s=10)
        # ax.scatter(self.pts[0],self.pts[1],c='k',s=2,label='Input')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='r',marker='x',label='apex')
            plt.legend(fontsize=10)
        plt.xlabel('X [m]', fontsize=10)
        plt.ylabel('Y [m]', fontsize=10)
        plt.title('Average speed:'+str('{0:.2f}'.format(np.mean(self.v)*2.23))+'mph'+\
            '\nTotal energy consumption:'+str('{0:.2f}'.format(np.sum(self.energy)/1000))+'kJ', fontsize=12)
        cbaxes = fig2.add_axes([0.87, 0.2, 0.02, 0.6])
        cbar = fig2.colorbar(scalarMap,cax=cbaxes)
        cbar.ax.set_ylabel('velocity [mph]')
        plt.draw()

        return 1


    def plot_derivatives(self):
        '''
        check derivative vectors for curvature calculation
        '''

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        plt.scatter(self.pts_interp[0],self.pts_interp[1],label='Discretized points')
        plt.quiver(self.pts_interp[0],self.pts_interp[1],self.dpds[0],self.dpds[1],linewidth=0.5,label='dpds')
        plt.quiver(self.pts_interp[0],self.pts_interp[1],self.d2pds2[0],self.d2pds2[1],label='d2pds2')
        plt.title('Curvature')
        plt.legend()
        plt.draw()

        return 1