from enum import IntEnum
from typing import List, Tuple, Optional
import logging
import numpy as np

logging.basicConfig(filename='path_generator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class PathGenerator():
    class Stage(IntEnum):
        BeforeStart=0

        IncreasingJerk = 1

        ConstantJerk = 2

        DecreasingJerk = 3
        
        ConstantAcceleration = 4

        DecreasingJerk2 = 5

        ConstantJerk2 = 6

        IncreasingJerk2 = 7

        ConstantVelocity = 8 

        DecreasingJerk3 = 9

        ConstantJerk3 = 10

        IncreasingJerk3 = 11
        
        ConstantAcceleration2 = 12

        IncreasingJerk4 = 13
        
        ConstantJerk4 = 14

        DecreasingJerk4 = 15

        AfterEnd = 16

    def __init__(self) -> None:
        # private attributes
        
        self._stroke_time: float = 0.0
        """The time available to make the stroke"""
        self._max_v: float = 0.0
        """the max speed"""
        self._max_a: float = 0.0
        """the max acceleration"""
        self._max_j: float = 0.0
        """the max jerk"""
        self._max_s: float = 0.0
        """the max snap"""
        self._min_x: float = 0.0
        self._max_x: float = 0.0
        """"if both max_x and min_x are zero, then
        there is actually no constraint for position"""
        self._stroke_sign: float = 1.0
        """the sign of the motion profile +/-1"""
        self._stroke: float = 0.0
        self._motion_time: List[float] = ([0.0] *
                                            len(PathGenerator.Stage))
        """holds the times at the end
        of each part (there are 15 parts)"""
        self._position: List[float] = ([0.0] *
                                        len(PathGenerator.Stage))
        """holds the positions at the end
        of each part"""
        self._velocity: List[float] = [0.0] * len(PathGenerator.Stage)
        """holds velocities at the end
        of each part (there are 15 parts)"""
        self._acceleration: List[float] = [0.0] * len(PathGenerator.Stage)
        """holds accelerations  at the end
        of each part (there are 15 parts)"""
        self._jerk: List[float] = [0.0] * len(PathGenerator.Stage)
        """holds the jerks at the end
        of each part (there are 15 parts)"""
        self._snap: List[float] = [0.0] * len(PathGenerator.Stage)
        """holds the snap at the end
        of each part (there are 15 parts)"""    

    @property
    def stroke_time(self) -> float:
        """The time available to make te stroke"""
        return self._stroke_time

    @stroke_time.setter
    def stroke_time(self, new_stroke_time: float) -> None:
        self._stroke_time = abs(new_stroke_time)
    
    @property
    def max_v(self) -> float:
        """The maximum absolute velocity (sign is always positive)"""
        return self._max_v

    @max_v.setter
    def max_v(self,new_max_v: float) -> None:
        self._max_v = abs(new_max_v)

    @property
    def max_a(self) -> float:
        """The maximum absolute acceleration (sign is always positive).
        """
        return self._max_a

    @max_a.setter
    def max_a(self, new_max_a: float) -> None:
        self._max_a = abs(new_max_a)

    @property
    def max_j(self) -> float:
        """The maximum absolute acceleration (sign is always positive).
        """
        return self._max_j

    @max_j.setter
    def max_j(self, new_max_j: float) -> None:
        self._max_j = abs(new_max_j)

    @property
    def max_s(self) -> float:
        return self._max_s

    @max_s.setter
    def max_s(self, new_max_s: float) -> None:
        self._max_s = abs(new_max_s)


    @property
    def max_x(self) -> float:
        """The upper limit for the position."""
        return self._max_x

    @max_x.setter
    def max_x(self, new_max_x: float) -> None:
        if new_max_x > self.min_x:
            self._max_x = new_max_x
        else:
            logging.info('max_x setter ignored, as max_x is less than min_x')

    @property
    def min_x(self) -> float:
        """The lower limit for the position."""
        return self._min_x

    @min_x.setter
    def min_x(self, new_min_x: float) -> None:
        if self.max_x > new_min_x:
            self._min_x = new_min_x
        else:
            logging.info('min_x setter ignored, as max_x is less than min_x')

    def calculate_path(self, startpos: float = 0.0,
                       stroke: float = 0.0, stroke_time: float = 0.0 ,starttime: float = 0.0, fraction_s: float = 1.0,
                       fraction_j: float = 1.0, fraction_a: float = 1.0,
                       fraction_v: float = 1.0) -> None:
        """Method for parametrizing the path to be generated later.

        Args:
            startpos: the position from which the path starts.
            stroke: the stroke of the path
            starttime: the value for the motion time at which
            the movement is to start.
            fraction_j: the maximum jerk
            (in terms of a fraction of the acceleration constraint)
            that should be used (sign doesn't matter)
            fraction_a: the maximum acceleration
            (in terms of a fraction of the acceleration constraint)
            that should be used (sign doesn't matter)
            fraction_v: the maximum velocity
            (in terms of a fraction of the velocity constraint)
            that should be used (sign doesn't matter).
        """
        # check the startposition and the stroke to respect position limits

        if (self.min_x == 0.0 and self.max_x == 0.0):
                checked_startpos = startpos
                checked_stroke = stroke
        else:
            if ((startpos <= self.max_x) and (startpos >= self.min_x)):
                checked_startpos = startpos
                if ((startpos + stroke) <= self.max_x):
                    if ((startpos + stroke) >= self.min_x):
                        checked_stroke = stroke
                    else:
                        logging.warning(""" The stroke exceeds the mimimum
                                        position, modification applied""")
                        checked_stroke = self.min_x - startpos
                else:
                    logging.warning(""" The stroke exceeds the maximum
                                    position, modification applied""")
                    checked_stroke = self.max_x - startpos
            else:
                # definitely something wrong; do not move
                logging.warning(""" The start position does not lie within the
                                position limits""")
                checked_startpos = startpos
                checked_stroke = 0.0

        self._stroke_sign = 1.0
        self._stroke = abs(checked_stroke)
        self._stroke_time = stroke_time

        self._clear_members_parts()

        stime = 0.0
        jtime = 0.0
        atime = 0.0
        vtime = 0.0
        order = 0
        
        s_max = abs(fraction_s) * self.max_s

        if s_max > 0.0:
            order = 4
        j_max = abs(fraction_j) * self.max_j
        if s_max > 0.0:
            order = 3
        a_max = abs(fraction_a) * self.max_a
        if a_max > 0.0 and order == 0:
            order = 2
        v_max = abs(fraction_v) * self.max_v
        if v_max > 0.0 and order == 0:
            order = 1

        if self._stroke > 0.0:
            if order > 0:
                stime, jtime, atime, vtime = self._calculate_real_path(s_max,
                                                                j_max,
                                                                a_max,
                                                                v_max)
            self._stroke_sign = checked_stroke / self._stroke

        self._motion_time[0] = starttime
        self._position[0] = checked_startpos
        self._fill_members_parts(stime, jtime, atime, vtime)
    
    def get_motion_stage(self, the_time: float) -> Stage:
        the_stage = PathGenerator.Stage.BeforeStart
        # find out in which part of the motion we are
        while ((the_stage < PathGenerator.Stage.AfterEnd) and
              (the_time > self._motion_time[the_stage])):
            the_stage = PathGenerator.Stage(the_stage + 1)
        return the_stage

    def generate_x(self, the_time: float,
                   the_stage: Optional[Stage] = None) -> float:
        """Method for generating the position
        at a certain moment in time.

        Args:
            the_time: the moment in time.
            the_stage: Current Stage

        Returns:
            the position value
        """
        stge = the_stage or self.get_motion_stage(the_time)

        if(stge == PathGenerator.Stage.BeforeStart or
           stge == PathGenerator.Stage.AfterEnd):
            return self._position[stge]

        '''lower stge by one such that it holds what we have
        completed already'''
        stge -= 1
        delta_t = the_time-self._motion_time[stge]
        return (self._position[stge] + self._velocity[stge]*delta_t +
                (1.0/2.0)*self._acceleration[stge]*(delta_t)**2 +
                (1.0/6.0)*self._jerk[stge]*(delta_t)**3 + 
                (1.0/24.0)*self._snap[stge+1]*(delta_t)**4)


    def generate_v(self, the_time: float,
                   the_stage: Optional[Stage] = None) -> float:
        """Method for generating the velocity
        at a certain moment in time.

        Args:
            the_time: the moment in time.
            the_stage: Current Stage

        Returns:
            the velocity value
        """
        stge = the_stage or self.get_motion_stage(the_time)

        if(stge == PathGenerator.Stage.BeforeStart or
           stge == PathGenerator.Stage.AfterEnd):
            return self._velocity[stge]

        '''lower stge by one such that it holds what we have
        completed already'''
        stge -= 1
        delta_t = the_time-self._motion_time[stge]
        return (self._velocity[stge] +
                self._acceleration[stge]*delta_t +
                (1.0/2.0)*self._jerk[stge]*(delta_t)**2 +
                (1.0/6.0)*self._snap[stge+1]*(delta_t)**3)

    def generate_a(self, the_time: float,
                   the_stage: Optional[Stage] = None) -> float:
        """Method for generating the acceleration
        at a certain moment in time.

        Args:
            the_time: the moment in time.
            the_stage: Current Stage

        Returns:
            Acceleration value
        """
        stge = the_stage or self.get_motion_stage(the_time)

        if(stge == PathGenerator.Stage.BeforeStart or
           stge == PathGenerator.Stage.AfterEnd):
            return self._acceleration[stge]

        '''lower stge by one such that it holds what we have
        completed already'''
        stge -= 1
        delta_t = the_time-self._motion_time[stge]
        return  ( self._acceleration[stge] +
                self._jerk[stge] * delta_t + 
                (1.0/2.0)*self._snap[stge+1] * (delta_t)**2)


    def generate_j(self, the_time: float,
                   the_stage: Optional[Stage] = None) -> float:
        """Method for generating the jerk
        at a certain moment in time.

        Args:
            the_time: the moment in time.
            the_stage: Current Stage

        Returns:
            Jerk Value
        """
        stge = the_stage or self.get_motion_stage(the_time)

        if(stge == PathGenerator.Stage.BeforeStart or
           stge == PathGenerator.Stage.AfterEnd):
            return self._jerk[stge]

        '''lower stge by one such that it holds what we have
        completed already'''
        stge -= 1
        delta_t = the_time-self._motion_time[stge]
        return (self._jerk[stge] + 
                self._snap[stge+1] * delta_t)

    def generate_s(self, the_time: float,
                   the_stage: Optional[Stage] = None) -> float:
        """Method for generating the jerk
        at a certain moment in time.

        Args:
            the_time: the moment in time.
            the_stage: Current Stage

        Returns:
            Jerk Value
        """
        stge = the_stage or self.get_motion_stage(the_time)

        if(stge == PathGenerator.Stage.BeforeStart or
           stge == PathGenerator.Stage.AfterEnd):
            return self._snap[stge]

        '''lower stge by one such that it holds what we have
        completed already'''
        stge -= 1
        return self._snap[stge+1]


    def _calculate_real_path(self, s_max: float, j_max: float, a_max: float,
                                v_max: float) -> Tuple[float, float, float]:
        """Method for calculating the duration of constant jerk / acceleration
        / velocity for the given stroke and constraints.

        Stroke, j_max, a_max and v_max all positive (sign is dealt with later)
        Startpos equal to 0.0 (is dealt with later)

        Returns:
            (jtime, atime, vtime)
        """
        stime = 0.0
        jtime = 0.0
        atime = 0.0
        vtime = 0.0

        self._snap[1] = s_max

        # The minimum order that can be created with a defined stroke and time is a 1st order
        # path. 

        if (v_max > 0.0):
            if (a_max > 0.0):
                if (j_max > 0.0):
                    if (s_max > 0.0):
                        # 4th Order
                        # Find the smallest jerk that can be used to obtain the stroke in the given time
                        coef = [  -2 / (16 * s_max) * self._stroke_time**2 , 
                                2/64 * self._stroke_time**3,  -self._stroke]
                        roots = np.roots(coef)
                        # Find the minimal positive real solution.
                        realroots = []
                        for i in range(len(roots)):
                            if roots[i].real > 0 and roots[i].imag == 0:
                                realroots.append(roots[i].real)
                        # If the solution exists use the found jerk -> j_used. Otherwise maximize
                        # the stroke with j_max
                        if realroots == []:
                            print("Snap Limited")
                            j_used = j_max
                            stime = self.stroke_time/8
                        else:
                            j_used = np.min(realroots)
                            stime = j_used/ s_max

                        # The jerk is minimal for the entire stroke iff the jerk is constant for
                        # the longest duration possible. This is the entire time where the snap
                        # is 0     
                        jtime = (self._stroke_time - 8 * stime)/4
                        self._fill_members_parts(stime, jtime, atime, vtime)

                        # check if stime is too long to respect j_max
                        if ((j_max > 0.0) and (self._jerk[1] > j_max)):
                            stime = j_max/s_max
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            print("Jerk Limited")
                        
                        # check if jtime is too long to respect a_max
                        if ((a_max > 0.0) and (self._acceleration[3] > a_max)):
                            print("Acceleration limited")
                            temp_var = a_max - 2 * self._acceleration[1]
                            jtime = 0.5* temp_var / self._jerk[1]
                            self._fill_members_parts(stime, jtime, atime, vtime) 
                            # check if stime is too long to respect a_max                         
                            if ((a_max > 0.0) and (self._acceleration[3] > a_max)):
                                stime = (a_max/(s_max))**0.5
                            self._fill_members_parts(stime, jtime, atime, vtime)

                        # check if jtime is too long to respect v_max
                        if ((v_max > 0.0) and (self._velocity[7]) > v_max):
                            print("Velocity limited")
                            
                            a= self._snap[1] * self._motion_time[1]
                            b= (3 * self._snap[1]  * (self._motion_time[1] **2))
                            c= 2 * self._snap[1]  * (self._motion_time[1] **3) - v_max
                            jtime = (-b + (b**2 - 4 * a * c)**0.5) / (2 * a)
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            # check if stime is to long to respect v_max
                            if ((v_max > 0.0) and (self._velocity[7]) > v_max):
                                stime = (v_max/(2*s_max))**(1.0/3.00)
                                self._fill_members_parts(stime, jtime, atime, vtime)
                        
                        
                        # Snap and Jerk are found that minimize the Jerk
                        # In the case that Jerk and Snap are limited a constant
                        # acceleration can be used to extend the stroke within the given time
                        if (self._position[15] < self._stroke):  
                            atime = (self._stroke_time - 8*stime - 4*jtime)/2
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            if ((v_max > 0.0) and (self._velocity[7] > v_max)):
                                atime = (v_max - 2*s_max*stime**3 - 3 *s_max*stime**2*jtime - 
                                    s_max * stime * jtime**2) / (s_max * stime**2 + s_max * stime * jtime)
                                self._fill_members_parts(stime, jtime, atime, vtime)
                        # In the situation where the constant acceleration is limited by the
                        # velocity. A constant velocity can be added to maximize the distance
                        # within the given time.                    
                        if (self._position[15] < self._stroke):
                            temp_var0 = 2.0*(0.5*self._stroke - self._position[7])
                            vtime = temp_var0/self._velocity[7]
                            if vtime > self._stroke_time - 8*stime - 4*jtime - 2*atime:
                                vtime = self._stroke_time - 8*stime - 4*jtime - 2*atime
                            self._fill_members_parts(stime, jtime, atime, vtime)
                        

                    else:
                        # 3th Order
                        # The jerk is mimimal over the entire stroke if there is no constant
                        # velocity or constant acceleration phase. So use all time as jtime
                        jtime = self._stroke_time/4
                        j_used = self._stroke / (2 * (jtime)**3)
                        if j_used > j_max:
                            j_used = j_max
                            print("Jerk limited")
                        self._jerk[1] = j_used
                        self._jerk[5] = -j_used
                        self._jerk[9] = -j_used
                        self._jerk[13] = j_used
                        
                        self._fill_members_parts(stime, jtime, atime, vtime)

                        # check if jtime is too long to respect a_max
                        if ((a_max > 0.0) and (self._acceleration[3] > a_max)):
                            print("Acceleration limited")
                            jtime = a_max/j_used
                            self._fill_members_parts(stime, jtime, atime, vtime)
                        # check if jtime is too long to respect v_max
                        if ((v_max > 0.0) and (self._velocity[7] > v_max)):
                            jtime = (v_max/j_used)**0.5
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            print("Velocity limited")
                        
                        #If the stroke was limited add constant acceleration phase
                        if (self._position[15] < self._stroke):  
                            atime = (self._stroke_time - 8*stime - 4*jtime)/2
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            # check if a time is too long to respect v_max
                            if ((v_max > 0.0) and (self._velocity[7] > v_max)):
                                atime = (v_max - self._jerk[1] * jtime**2) / (self._jerk[1] * jtime)
                                self._fill_members_parts(stime, jtime, atime, vtime)
                        #If the stroke was limited, if so add constant velocity phase
                        if (self._position[15] < self._stroke):
                            temp_var0 = 2.0*(0.5*self._stroke - self._position[7])
                            vtime = temp_var0/self._velocity[7]
                            #check if vtime remains in stoke time.
                            if vtime > self._stroke_time - 8*stime - 4*jtime - 2*atime:
                                vtime = self._stroke_time - 8*stime - 4*jtime - 2*atime
                            self._fill_members_parts(stime, jtime, atime, vtime)    

                else:
                    # 2nd Order
                    atime = self._stroke_time/2
                    a_used = self._stroke / (atime**2)
                    if a_used > a_max:
                        a_used = a_max
                        print("Acceleration limited")
                    self._acceleration[3] = a_used
                    self._acceleration[11] = -a_used

                    self._fill_members_parts(stime, jtime, atime, vtime)
        
                    # check if atime is too long to respect v_max
                    if ((v_max > 0.0) and (self._velocity[7] > v_max)):
                            atime = v_max/a_max
                            self._fill_members_parts(stime, jtime, atime, vtime)
                            print("Velocity limited")

                    #check if the troke was limited, if so add constant velocity        
                    if (self._position[15] < self._stroke):
                            temp_var0 = 2.0*(0.5*self._stroke - self._position[7])
                            vtime = temp_var0/self._velocity[7]
                            #check if vtime remains in stoke time.
                            if vtime > self._stroke_time - 8*stime - 4*jtime - 2*atime:
                                vtime = self._stroke_time - 8*stime - 4*jtime - 2*atime
                            self._fill_members_parts(stime, jtime, atime, vtime)

            else:
                # 1st Order
                vtime = self._stroke_time
                v_used = self._stroke/self._stroke_time
                if v_used > v_max:
                    v_used = v_max
                    print("Velocity limited")
                self._velocity[7] = v_used

        self._fill_members_parts(stime, jtime, atime, vtime)

        return (stime, jtime, atime, vtime)    

                





    def _fill_members_parts(self, stime: float ,jtime: float, atime: float, vtime: float) -> None:
        """ Fill Members Method

        Args:
            stime
            jtime
            atime
            vtime

        Returns:
            None
        """
        self._motion_time[1] = self._motion_time[0]+stime
        self._snap[1] = self._stroke_sign*self._snap[1]

        self._motion_time[2] = self._motion_time[1]+jtime
        self._snap[2] = 0.0

        self._motion_time[3] = self._motion_time[2]+stime
        self._snap[3] = -self._snap[1]

        self._motion_time[4] = self._motion_time[3]+atime
        self._snap[4] = 0.0

        self._motion_time[5] = self._motion_time[4]+stime
        self._snap[5] = -self._snap[1]

        self._motion_time[6] = self._motion_time[5]+jtime
        self._snap[6] = 0.0

        self._motion_time[7] = self._motion_time[6]+stime
        self._snap[7] = self._snap[1]

        self._motion_time[8] = self._motion_time[7]+vtime
        self._snap[8] = 0.0

        self._motion_time[9] = self._motion_time[8]+stime
        self._snap[9] = -self._snap[1]

        self._motion_time[10] = self._motion_time[9]+jtime
        self._snap[10] = 0.0

        self._motion_time[11] = self._motion_time[10]+stime
        self._snap[11] = self._snap[1]

        self._motion_time[12] = self._motion_time[11]+atime
        self._snap[12] = 0.0

        self._motion_time[13] = self._motion_time[12]+stime
        self._snap[13] = self._snap[1]

        self._motion_time[14] = self._motion_time[13]+jtime
        self._snap[14] = 0.0

        self._motion_time[15] = self._motion_time[14]+stime
        self._snap[15] = -self._snap[1]

        for i in range(1, 16):
            delta_t = self._motion_time[i]-self._motion_time[i-1]
            if delta_t > 0.0:
                self._jerk[i] = (self._jerk[i-1] + 
                                self._snap[i] * delta_t)

                self._acceleration[i] = (self._acceleration[i-1] +
                                        self._jerk[i-1] * delta_t +
                                        (1.0/2.0)*self._snap[i]*delta_t**2)
                                    
                self._velocity[i] = (self._velocity[i-1] +
                                    self._acceleration[i-1]*delta_t +
                                    (1.0/2.0)*self._jerk[i-1]*(delta_t**2) + 
                                    (1.0/6.0)*self._snap[i]*(delta_t**3))

                self._position[i] = (self._position[i-1] +
                                    self._velocity[i-1]*delta_t +
                                    (1.0/2.0)*self._acceleration[i-1] *
                                    (delta_t**2) +
                                    (1.0/6.0)*self._jerk[i-1]*(delta_t**3) + 
                                    (1.0/24.0)*self._snap[i]*(delta_t**4))
            else:
                if i in [1]:
                    # Jerk is discontinuous
                    # take care of the sign
                    self._jerk[1] = (self._stroke_sign* self._jerk[1])
                    self._jerk[5] = (self._stroke_sign* self._jerk[5])
                    self._jerk[9] = (self._stroke_sign* self._jerk[9])
                    self._jerk[13] = (self._stroke_sign* self._jerk[13])

                if i in [3,5,7,9,11,13,15]:
                    # Jerk is discontinuous
                    # Position is continuous
                    self._position[i] = self._position[i-1]
                    
                    # Acceleration may or may not be continuous;
                    if (atime == 0 and jtime == 0 and stime == 0):
                        self._velocity[7] = (self._stroke_sign * self._velocity[7])
                    elif (jtime == 0 and stime == 0):
                        self._acceleration[3] = (self._stroke_sign * self._acceleration[3])
                        self._acceleration[11] = (self._stroke_sign * self._acceleration[11])
                        self._velocity[i] = self._velocity[i-1]
                    else:
                        self._acceleration[i] = self._acceleration[i-1]
                        self._velocity[i] = self._velocity[i-1]
                        

                if i in [2, 6, 10, 14]:
                    # No constant Jerk Phase
                    self._jerk[i] = self._jerk[i-1]
                    self._acceleration[i] = self._acceleration[i-1]
                    self._velocity[i] = self._velocity[i-1]
                    self._position[i] = self._position[i-1]

                if i in [4, 12]:
                    # No constant Acceleration Phase
                    self._acceleration[i] = self._acceleration[i-1]
                    self._velocity[i] = self._velocity[i-1]
                    self._position[i] = self._position[i-1]

                if i in [8]:
                    # No constant velocity phase
                    self._velocity[i] = self._velocity[i-1]
                    self._position[i] = self._position[i-1]
                
        self._position[16] = self._position[15]

    def _clear_members_parts(self) -> None:
        """ Fill Members with 0.0
        """
        for i in range(0, 16):
            self._motion_time[i] = 0.0
            self._snap[i] = 0.0
            self._jerk[i] = 0.0
            self._acceleration[i] = 0.0
            self._velocity[i] = 0.0
            self._position[i] = 0.0