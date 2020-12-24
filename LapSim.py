import numpy as np
import matplotlib.pyplot as plt

class LapSim:
    '''
    A lap time simulator (point-mass)
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.m = kwargs.pop('m',300)                    # mass of car [kg]
        self.mu = kwargs.pop('mu',0.5)                    # tyre frictional coefficient
        self.g = 9.81                                   # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)            # number of discretized points

        self.pts = kwargs.pop('pts',0)                  # input track data
        self.pts_interp = kwargs.pop('pts_interp',0)    # interpolated track data
        self.track_len = kwargs.pop('track_len',0)      # total track length

        self.ds = kwargs.pop('ds',0)                    # differential arc length
        self.r = kwargs.pop('r',0)                      # radius of curvature
        self.apex = kwargs.pop('apex',0)                # apex points location
        self.brake = kwargs.pop('brake',0)              # brake points location
        self.v = kwargs.pop('v',0)                      # velocity at each discretized point on track

        self.time = kwargs.pop('time',0)                # total lap time

    @classmethod
    def init_ellipse(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.pop('resolution',10)               # resolution of initial track data
        # input track data
        s = np.linspace(0,2*np.pi,res,endpoint=False)
        pts = np.vstack((3*np.cos(s),2*np.sin(s)))
        
        return cls(pts=pts, **kwargs)


    def lap_time(self):
        '''
        Calculates lap time
        '''
        
        # interpolate equidistant points on the track
        self.pts_interp, self.ds, self.track_len = self.discretize()

        # calculate radius of curvature
        self.r = self.roc()

        # find apex locations
        self.apex = self.find_apex()

        # calculate traction-limited velocity at each point
        self.v, self.brake = self.get_velocity()
        self.plot_discretized_points(apex=1, brake=1)            # check apex location

        # calculate lap time
        self.time = np.sum(1/(self.v/self.ds))

        return 1

    def discretize(self):
        
        # Parametrize track by variable s. Assume that the input points are ordered and arbitrarily spaced.
        # Parametrization with respect to normalized arc length
        diff = self.pts - np.roll(self.pts,1,axis=1)
        arclen = np.linalg.norm(diff,axis=0)
        track_len = np.sum(arclen)
        s = np.cumsum(arclen)/track_len

        # periodic boundary condition (x(0) == x(1), y(0) == y(1))
        s = np.append(0,s)
        x = np.append(self.pts[0],self.pts[0,0])
        y = np.append(self.pts[1],self.pts[1,0])

        # Discretize track by equidistant points in s
        from scipy.interpolate import interp1d
        snew = np.linspace(0,1,num=self.steps, endpoint=False)
        fx = interp1d(s,x, kind='cubic')
        fy = interp1d(s,y, kind='cubic')

        xnew = fx(snew)
        ynew = fy(snew)
        pts_interp = np.vstack((xnew,ynew))

        ds = track_len/self.steps

        return pts_interp, ds, track_len

    
    def roc(self):
        '''
        Calculates radius of curvature at each point
        r(s) = |(dxds^2+dyds^2)^(3/2)/(dxds*dyds2-dyds*dxds2)|
        '''

        diff = ((self.pts_interp - np.roll(self.pts_interp,1,axis=1))+(np.roll(self.pts_interp,-1,axis=1)-self.pts_interp))/2
        drds = diff/self.ds

        diff2 = ((drds - np.roll(drds,1,axis=1))+(np.roll(drds,-1,axis=1)-drds))/2
        d2rds2 = diff2/self.ds

        num = np.linalg.norm(drds,axis=0)**3
        den = np.cross(drds,d2rds2,axis=0)
        r = num/den
        
        return r

    
    def find_apex(self):
        '''
        finds cornering apex list: look for sign change in dr
        shift the arrays such that the discretization starts with an apex

        '''

        dr = self.r - np.roll(self.r,1,axis=0)
        sign = np.sign(dr)
        sign_flip = sign - np.roll(sign,-1,axis=0)

        apex = np.where(sign_flip == np.min(sign_flip))
        idx_0 = self.r.shape[0]-apex[0][0]
        idx = np.arange(self.r.shape[0]) - idx_0

        # re-indexing to make apex the first point
        apex = apex-apex[0][0]
        self.r = self.r[idx]
        self.pts_interp = self.pts_interp[:,idx]

        return apex


    def get_velocity(self):
        '''
        Calculates traction-limited velocity at each point
        m*ap = mv^2/r
        a = sqrt(ap^2+at^2)
        a = mu * N
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for forward integration
        v_{i-1} = -ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for backward integration
        '''

        a = self.g * self.mu
        vf = np.zeros(self.steps)
        vf[self.apex] = np.sqrt(self.mu * self.g * self.r[self.apex])     # velocity at apex
        
        # get velocity list (forward)
        for i in np.arange(self.steps-1):
            if vf[i+1] == 0:
                ap = vf[i]**2/self.r[i+1]
                if a>ap:
                    at = np.sqrt(a**2 - ap**2)
                else: 
                    at = 0
                vf[i+1] = vf[i]+at*np.abs(1/vf[i])*self.ds

        vb = np.zeros(self.steps)
        vb[self.apex] = np.sqrt(self.mu * self.g * self.r[self.apex])     # velocity at apex

        # get velocity list (backward)
        for i in np.arange(self.steps-1):
            if vb[-(i+1)] == 0:
                ap = vb[-i]**2/self.r[-(i+1)]
                if a>ap:
                    at = np.sqrt(a**2 - ap**2)
                else: 
                    at = 0
                vb[-(i+1)] = vb[-i]+at*np.abs(1/vb[-i])*self.ds

        # get braking locations (forward and backward vlists intersect)
        diff = np.sign(vb-vf)
        signflip = diff - np.roll(diff,-1,axis=0)
        brakepts = np.where(signflip == np.max(signflip))

        # get final velocity list
        v = np.min([vb,vf],axis=0)

        return v, brakepts


    def plot_discretized_points(self, apex=0, brake=0):
        
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        plt.scatter(self.pts_interp[0], self.pts_interp[1],label='Interpolation')
        plt.scatter(self.pts[0],self.pts[1],s=2,label='Input')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='r',label='apex')
        if brake==1:
            plt.scatter(self.pts_interp[0,self.brake],self.pts_interp[1,self.brake],c='k',label='brake')
        plt.title('Discretized Track points (equidistant in s)')
        plt.legend()
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
        plt.quiver(self.pts_interp[0],self.pts_interp[1],drds[0],drds[1],linewidth=0.5,label='drds')
        plt.quiver(self.pts_interp[0],self.pts_interp[1],d2rds2[0],d2rds2[1],label='d2rds2')
        plt.title('Curvature')
        plt.legend()
        plt.draw()

        return 1