import numpy as np
from numpy.polynomial import polynomial as P

class CT_System:
    def __init__(self, 
                Controller:str, 
                DO:bool, 
                m :float, 
                wc:float, 
                Bd:float = 10, 
                Bi:float = 2, 
                Bi_DO:float = 2, 
                a1:float = 0, 
                dr1:float = 0.01, 
                y:float = 0.01, 
                a2:float = 0, 
                dr2:float = 0.01 , 
                r:float = 0.01 ,
                dc:float = 0.8):
        self.Controller = Controller
        self.DO         = DO
        self.m          = m
        self.wc         = wc
        self.Bi_DO      = Bi_DO
        self.a1         = a1
        self.y          = y
        self.a2         = a2
        self.r          = r
        self.dc         = dc

        self.wr1        = self.wc / self.y
        self.wr2        = self.wc / (self.y * self.r)
        self.d1         = dr1 * 2 * self.wr1
        self.d2         = dr2 * 2 * self.wr2

        self.tz = np.sqrt(Bd)/wc
        self.ti = (Bi * self.tz)
        self.tp = 1 / (wc * np.sqrt(Bd))
        self.kp = m * wc**2 / (np.sqrt(Bd))

    def Update(self, 
               Bi_DO:float = 2, 
               a1:float = 0, 
               dr1:float = 0.01, 
               y:float = 0.01, 
               a2:float = 0, 
               dr2:float = 0.01 , 
               r:float = 0.01 ,
               dc:float = 0.8):
        self.Bi_DO      = Bi_DO
        self.a1         = a1
        self.y          = y
        self.a2         = a2
        self.r          = r   
        self.dc         = dc

        self.wr1        = self.wc / self.y
        self.wr2        = self.wc / (self.y * self.r)
        self.d1         = dr1 * 2 * self.wr1
        self.d2         = dr2 * 2 * self.wr2

    def PD(self): # --> (1,4) , (1,4)
        b0 = self.kp*self.tz
        b1 = self.kp
        a0 = self.tp
        a1 = 1
        return [0,0,b0,b1],[0,0,a0,a1]

    def PDF(self): # --> (1,4) , (1,4)
        b0 = 0
        b1 = self.kp*self.tz 
        b2 = self.kp
        a0 = self.tp**2
        a1 = 2 * self.dc * self.tp
        a2 = 1
        return [0,b0,b1,b2],[0,a0,a1,a2]

    def PID(self): # --> (1,4) , (1,4)
        b0 = self.kp*self.ti*self.tz
        b1 = self.kp*(self.ti+self.tz)
        b2 = self.kp
        a0 = self.tp*self.ti
        a1 = self.ti
        a2 = 0
        return [0,b0,b1,b2],[0,a0,a1,a2]

    def PIDF(self): # --> (1,4) , (1,4)
        b0 = self.kp*self.ti*self.tz
        b1 = self.kp*(self.ti+self.tz)
        b2 = self.kp
        a0 = self.tp**2*self.ti
        a1 = 2*self.dc*self.tp*self.ti
        a2 = self.ti
        a3 = 0
        return [0,b0,b1,b2],[a0,a1,a2,a3]

    def G(self): # --> (1,5) , (1,7)
        b0 = self.a1 + self.a2 + 1
        b1 = (self.a2 + 1) * self.d1 + (self.a1 + 1)*self.d2
        b2 = self.d1 * self.d2 + (self.a2 + 1)*self.wr1**2 + (self.a1 + 1)*self.wr2**2
        b3 = self.d2*self.wr1**2 + self.d1*self.wr2**2
        b4 = self.wr1**2 * self.wr2**2

        a0 = self.m
        a1 = self.m*(self.d1 + self.d2)
        a2 = self.m*(self.d1*self.d2 + self.wr1**2 + self.wr2**2)
        a3 = self.m*(self.d1*self.wr2**2 + self.d2*self.wr1**2)
        a4 = self.m*(self.wr1**2 * self.wr2**2)
        a5 = 0
        a6 = 0
        return [b0,b1,b2,b3,b4] , [a0,a1,a2,a3,a4,a5,a6]

    def Gn(self): # --> (1,1) , (1,3)
        return [1] , [self.m , 0, 0]

    def M(self): #M = (Gn^-1 G - I) --> (1,7) , (1,7)
        b , a = self.G()
        d , c = self.Gn()

        Num = [b[0]*c[0]                                                 - d[0]*a[0],
               b[0]*c[1] + b[1]*c[0]                                     - d[0]*a[1],
               b[0]*c[2] + b[1]*c[1] + b[2]*c[0]                         - d[0]*a[2],
                         + b[1]*c[2] + b[2]*c[1] + b[3]*c[0]             - d[0]*a[3],
                                     + b[2]*c[2] + b[3]*c[1] + b[4]*c[0] - d[0]*a[4],
                                                 + b[3]*c[2] + b[4]*c[1] - d[0]*a[5],
                                                             + b[4]*c[2] - d[0]*a[6]]
        
        Den = [d[0]*a[0],
               d[0]*a[1],
               d[0]*a[2],
               d[0]*a[3],
               d[0]*a[4],
               d[0]*a[5],
               d[0]*a[6]]

        return Num , Den

    def Q(self): # --> (1,1) , (1,3)
        ti = 1 / (self.Bi_DO * self.tz)
        return [1], [1 / (ti**2), np.sqrt(2)*(1/ti) , 1]

    def IQM(self): #(I + GM)^-1 --> (1,9) , (1,9)
        b , a = self.M()
        d , c = self.Q()


        Num = [c[0]*a[0],
               c[0]*a[1] + c[1]*a[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0] + d[0]*b[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + d[0]*b[1],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + d[0]*b[2],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + d[0]*b[3],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + d[0]*b[4],
                         + c[1]*a[6] + c[2]*a[5] + d[0]*b[5],
                                     + c[2]*a[6] + d[0]*b[6],]

        Den = [c[0]*a[0],
               c[0]*a[1] + c[1]*a[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4],
                         + c[1]*a[6] + c[2]*a[5],
                                     + c[2]*a[6]]

        return Den, Num         #Inverse, so Den and Num switched

    def H(self): #G(I + QM)^-1 == G()*IGM() --> (1,13) , (1,15)
        b , a = self.G() #(1,5) (1,7)
        d , c = self.IQM() #(1,9) (1,9)

        Num = [d[0]*b[0],
               d[0]*b[1] + d[1]*b[0],
               d[0]*b[2] + d[1]*b[1] + d[2]*b[0],
               d[0]*b[3] + d[1]*b[2] + d[2]*b[1] + d[3]*b[0],
               d[0]*b[4] + d[1]*b[3] + d[2]*b[2] + d[3]*b[1] + d[4]*b[0],
                         + d[1]*b[4] + d[2]*b[3] + d[3]*b[2] + d[4]*b[1] + d[5]*b[0],
                                     + d[2]*b[4] + d[3]*b[3] + d[4]*b[2] + d[5]*b[1] + d[6]*b[0],
                                                 + d[3]*b[4] + d[4]*b[3] + d[5]*b[2] + d[6]*b[1] + d[7]*b[0],
                                                             + d[4]*b[4] + d[5]*b[3] + d[6]*b[2] + d[7]*b[1] + d[8]*b[0],
                                                                         + d[5]*b[4] + d[6]*b[3] + d[7]*b[2] + d[8]*b[1],
                                                                                     + d[6]*b[4] + d[7]*b[3] + d[8]*b[2],
                                                                                                 + d[7]*b[4] + d[8]*b[3],
                                                                                                             + d[8]*b[4]]
        Den = [c[0]*a[0],
               c[0]*a[1] + c[1]*a[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + c[3]*a[0],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + c[3]*a[1] + c[4]*a[0],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + c[3]*a[2] + c[4]*a[1] + c[5]*a[0],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + c[3]*a[3] + c[4]*a[2] + c[5]*a[1] + c[6]*a[0],
                         + c[1]*a[6] + c[2]*a[5] + c[3]*a[4] + c[4]*a[3] + c[5]*a[2] + c[6]*a[1] + c[7]*a[0],
                                     + c[2]*a[6] + c[3]*a[5] + c[4]*a[4] + c[5]*a[3] + c[6]*a[2] + c[7]*a[1] + c[8]*a[0],
                                                 + c[3]*a[6] + c[4]*a[5] + c[5]*a[4] + c[6]*a[3] + c[7]*a[2] + c[8]*a[1],
                                                             + c[4]*a[6] + c[5]*a[5] + c[6]*a[4] + c[7]*a[3] + c[8]*a[2],
                                                                         + c[5]*a[6] + c[6]*a[5] + c[7]*a[4] + c[8]*a[3],
                                                                                     + c[6]*a[6] + c[7]*a[5] + c[8]*a[4],
                                                                                                 + c[7]*a[6] + c[8]*a[5],
                                                                                                             + c[8]*a[6]]
        return Num , Den

    def f_Controller(self, x):
        return {
            'PD': self.PD(),
            'PDF': self.PDF(),
            'PID': self.PID(),
            'PIDF': self.PIDF(),
        }[x]
    
    def f_DO(self, x):
        return {
            False: [np.append(np.zeros((1,8)),self.G()[0]) , np.append(np.zeros((1,8)),self.G()[1])],
            True: self.H(),
        }[x]

    def CL(self): #--> Num, Den
        d , c = self.f_Controller(self.Controller) # f() --> N(1,4) , D(1,4)
        b , a = self.f_DO(self.DO)                 # f() --> N(1,13) , D(1,15) 

        Den = [                                              + c[0]*a[0],
                                                             + c[0]*a[1]  + c[1]*a[0],
               d[0]*b[0]                                     + c[0]*a[2]  + c[1]*a[1] + c[2]*a[0],
               d[0]*b[1] + d[1]*b[0]                         + c[0]*a[3]  + c[1]*a[2] + c[2]*a[1] + c[3]*a[0],
               d[0]*b[2] + d[1]*b[1] + d[2]*b[0]             + c[0]*a[4]  + c[1]*a[3] + c[2]*a[2] + c[3]*a[1],
               d[0]*b[3] + d[1]*b[2] + d[2]*b[1] + d[3]*b[0] + c[0]*a[5]  + c[1]*a[4] + c[2]*a[3] + c[3]*a[2],
               d[0]*b[4] + d[1]*b[3] + d[2]*b[2] + d[3]*b[1] + c[0]*a[6]  + c[1]*a[5] + c[2]*a[4] + c[3]*a[3],
               d[0]*b[5] + d[1]*b[4] + d[2]*b[3] + d[3]*b[2] + c[0]*a[7]  + c[1]*a[6] + c[2]*a[5] + c[3]*a[4],
               d[0]*b[6] + d[1]*b[5] + d[2]*b[4] + d[3]*b[3] + c[0]*a[8]  + c[1]*a[7] + c[2]*a[6] + c[3]*a[5],
               d[0]*b[7] + d[1]*b[6] + d[2]*b[5] + d[3]*b[4] + c[0]*a[9]  + c[1]*a[8] + c[2]*a[7] + c[3]*a[6],
               d[0]*b[8] + d[1]*b[7] + d[2]*b[6] + d[3]*b[5] + c[0]*a[10] + c[1]*a[9] + c[2]*a[8] + c[3]*a[7],
               d[0]*b[9] + d[1]*b[8] + d[2]*b[7] + d[3]*b[6] + c[0]*a[11] + c[1]*a[10]+ c[2]*a[9] + c[3]*a[8],
               d[0]*b[10]+ d[1]*b[9] + d[2]*b[8] + d[3]*b[7] + c[0]*a[12] + c[1]*a[11]+ c[2]*a[10]+ c[3]*a[9],
               d[0]*b[11]+ d[1]*b[10]+ d[2]*b[9] + d[3]*b[8] + c[0]*a[13] + c[1]*a[12]+ c[2]*a[11]+ c[3]*a[10],
               d[0]*b[12]+ d[1]*b[11]+ d[2]*b[10]+ d[3]*b[9] + c[0]*a[14] + c[1]*a[13]+ c[2]*a[12]+ c[3]*a[11],
                         + d[1]*b[12]+ d[2]*b[11]+ d[3]*b[10]             + c[1]*a[14]+ c[2]*a[13]+ c[3]*a[12],
                                     + d[2]*b[12]+ d[3]*b[11]                         + c[2]*a[14]+ c[3]*a[13],
                                                 + d[3]*b[12]                                     + c[3]*a[14]]
        Num = [d[0]*b[0],
               d[0]*b[1] + d[1]*b[0],
               d[0]*b[2] + d[1]*b[1] + d[2]*b[0],
               d[0]*b[3] + d[1]*b[2] + d[2]*b[1] + d[3]*b[0],
               d[0]*b[4] + d[1]*b[3] + d[2]*b[2] + d[3]*b[1],
               d[0]*b[5] + d[1]*b[4] + d[2]*b[3] + d[3]*b[2],
               d[0]*b[6] + d[1]*b[5] + d[2]*b[4] + d[3]*b[3],
               d[0]*b[7] + d[1]*b[6] + d[2]*b[5] + d[3]*b[4],
               d[0]*b[8] + d[1]*b[7] + d[2]*b[6] + d[3]*b[5],
               d[0]*b[9] + d[1]*b[8] + d[2]*b[7] + d[3]*b[6],
               d[0]*b[10]+ d[1]*b[9] + d[2]*b[8] + d[3]*b[7],
               d[0]*b[11]+ d[1]*b[10]+ d[2]*b[9] + d[3]*b[8],
               d[0]*b[12]+ d[1]*b[11]+ d[2]*b[10]+ d[3]*b[9],
                         + d[1]*b[12]+ d[2]*b[11]+ d[3]*b[10],
                                     + d[2]*b[12]+ d[3]*b[11],
                                                 + d[3]*b[12]]
        return Num, Den



class DT_System:
    def __init__(self, Controller:str, 
                DO:bool, 
                m :float, 
                Ts:float,
                Relation:float = 0.1,
                Bd:float = 10, 
                Bi:float = 2, 
                Bi_DO:float = 2, 
                a1:float = 0, 
                dr1:float = 0.01, 
                y:float = 0.01, 
                a2:float = 0, 
                dr2:float = 0.01 , 
                r:float = 0.01 ,
                dc:float = 0.8,
                PIDF_type:str = 'Series'):
        """
        Relation: wc /(2 pi /Ts)
        PIDF_type: {'Series','Parallel'} 

        """
        
        self.wc = (2*np.pi*1/Ts)*Relation
        self.Ts = Ts

        self.Controller = Controller
        self.DO         = DO
        self.m          = m
        self.Bi_DO      = Bi_DO
        self.a1         = a1
        self.y          = y
        self.a2         = a2
        self.r          = r
        self.dc         = dc

        self.wr1        = self.wc / self.y
        self.wr2        = self.wc / (self.y * self.r)
        self.d1         = dr1 * 2 * self.wr1
        self.d2         = dr2 * 2 * self.wr2

        self.tz = np.sqrt(Bd)/self.wc
        self.ti = (Bi * self.tz)
        self.tp = 1 / (self.wc * np.sqrt(Bd))
        self.kp = m * self.wc**2 / (np.sqrt(Bd))

        self.PIDF_type = PIDF_type
    def Update(self, 
               Ts:float,
               Relation:float = 0.1,
               Bi_DO:float = 2, 
               a1:float = 0, 
               dr1:float = 0.01, 
               y:float = 0.01, 
               a2:float = 0, 
               dr2:float = 0.01 , 
               r:float = 0.01 ,
               dc:float = 0.8):

        self.wc = (2*np.pi*1/Ts)*Relation
        self.Ts = Ts

        self.Bi_DO      = Bi_DO
        self.a1         = a1
        self.y          = y
        self.a2         = a2
        self.r          = r   
        self.dc         = dc

        self.wr1        = self.wc / self.y
        self.wr2        = self.wc / (self.y * self.r)
        self.d1         = dr1 * 2 * self.wr1
        self.d2         = dr2 * 2 * self.wr2

    def PID(self): #f() --> (1,4),(1,4)
        c0 = self.kp * (1 + self.ti *self.tz * 4/(self.Ts**2) + (self.ti + self.tz)*2/self.Ts)
        c1 = self.kp * (2 - 8 / self.Ts**2 * self.ti * self.tz)
        c2 = self.kp * (1 + self.ti*self.tz*4/(self.Ts**2) - (self.ti + self.tz)*2/self.Ts)
        d0 = 2 / self.Ts * self.ti + 4 / (self.Ts**2) *self.ti * self.tp
        d1 = -8/(self.Ts**2) * self.ti * self.tp
        d2 = 4/(self.Ts**2) * self.ti * self.tp - 2/self.Ts * self.ti
        return [0, c0, c1, c2], [0, d0, d1, d2] 

    def I(self): #f() --> (1,4),(1,4)
        b0 = (2 / self.Ts * self.ti + 1)
        b1 = (1 - 2/ self.Ts * self.ti)
        a0 = 2 / self.Ts * self.ti
        a1 = - 2 / self.Ts * self.ti
        return [0, 0, b0, b1], [0, 0, a0, a1]

    def PDF(self): #f() --> (1,4),(1,4)
        b0 = self.kp * (1 + 2/self.Ts * self.tz)
        b1 = self.kp * (2)
        b2 = self.kp * (1 - 2/self.Ts * self.tz)
        a0 = 1 + 4/self.Ts**2 * self.tp**2 + self.dc * self.tp * 4/self.Ts
        a1 = 2 - 8 / self.Ts**2 * self.tp**2
        a2 = 1 + 4/self.Ts**2 * self.tp**2 - self.dc * self.tp * 4/self.Ts 
        return [0, b0, b1, b2], [0, a0, a1, a2]

    def PD(self): #f() --> (1,4),(1,4)
        b0 = self.kp * (2/self.Ts * self.tz + 1)
        b1 = self.kp * (1 - 2/self.Ts * self.tz)
        a0 = 1 + 2/ self.Ts * self.tp
        a1 = 1 - 2/ self.Ts * self.tp
        return [0, 0, b0, b1], [0, 0, a0, a1]

    def PIDF(self): #f() --> (1,4),(1,4)
        b , a = self.PDF()
        d , c = self.I()
        # Series
        if self.PIDF_type == 'Series':
            f0= b[1]*d[2]
            f1= b[2]*d[2] + b[1]*d[3]
            f2= b[3]*d[2] + b[2]*d[3]
            f3=           + b[3]*d[3]
            e0= a[1]*c[2]
            e1= a[2]*c[2] + a[1]*c[3]
            e2= a[3]*c[2] + a[2]*c[3]
            e3=           + a[3]*c[3]
        elif self.PIDF_type == 'Parallel':
            d , c = 0, 0, 0 , self.kp / self.ti ,0, 0, 1 , 0
            f0= b[1]*c[2]             + a[1]*d[2]
            f1= b[2]*c[2] + b[1]*c[3] + a[2]*d[2] + a[1]*d[3]
            f2= b[3]*c[2] + b[2]*c[3] + a[3]*d[2] + a[2]*d[3]
            f3=           + b[3]*c[3]             + a[3]*d[3]
            e0= a[1]*c[2]
            e1= a[2]*c[2] + a[1]*c[2]
            e2= a[3]*c[2] + a[2]*c[2]
            e3=           + a[3]*c[2]
        else:
            exit()
        return  [f0, f1, f2, f3], [e0, e1, e2, e3]

    def HG(self): #f() --> (1,9),(1,9)
        b0 = self.a1 + self.a2 + 1
        b1 = (self.a2 + 1)* self.d1 + (self.a1 + 1)*self.d2
        b2 = self.d1*self.d2 + (self.a2 + 1)*self.wr1**2 + (self.a1 + 1)*self.wr2**2 
        b3 = self.d1 * self.wr2**2 + self.d2 * self.wr1**2 
        b4 = self.wr1**2 * self.wr2**2
        
        a0 = 1
        a1 = self.d1 + self.d2
        a2 = self.d1*self.d2 + self.wr1**2 + self.wr2**2
        a3 = self.d1*self.wr2**2 + self.d2*self.wr1**2
        a4 = self.wr1**2 * self.wr2**2

        d0 =        b0*16/self.Ts**4 +      b1*8/self.Ts**3 +      b2*4/self.Ts**2 +      b3*2/self.Ts + b4*1 
        d1 =    -2 *b0*16/self.Ts**4 + 0 *  b1*8/self.Ts**3 + 2 *  b2*4/self.Ts**2 + 4 *  b3*2/self.Ts + b4*6
        d2 =    -2 *b0*16/self.Ts**4 - 4 *  b1*8/self.Ts**3 - 2 *  b2*4/self.Ts**2 + 4 *  b3*2/self.Ts + b4*14
        d3 =     6 *b0*16/self.Ts**4 + 0 *  b1*8/self.Ts**3 - 6 *  b2*4/self.Ts**2 - 4 *  b3*2/self.Ts + b4*14
        d4 =     0 *b0*16/self.Ts**4 + 6 *  b1*8/self.Ts**3 + 0 *  b2*4/self.Ts**2 - 10*  b3*2/self.Ts 
        d5 =    -6 *b0*16/self.Ts**4 + 0 *  b1*8/self.Ts**3 + 6 *  b2*4/self.Ts**2 - 4 *  b3*2/self.Ts - b4*14
        d6 =     2 *b0*16/self.Ts**4 - 4 *  b1*8/self.Ts**3 + 2 *  b2*4/self.Ts**2 + 4 *  b3*2/self.Ts - b4*14
        d7 =     2 *b0*16/self.Ts**4 + 0 *  b1*8/self.Ts**3 - 2 *  b2*4/self.Ts**2 + 4 *  b3*2/self.Ts - b4*6
        d8 =     -  b0*16/self.Ts**4 +      b1*8/self.Ts**3 -      b2*4/self.Ts**2 +      b3*2/self.Ts - b4*1


        c0 = self.Ts *self.m * (       a0*128/self.Ts**7 +     a1*64/self.Ts**6 +     a2*32/self.Ts**5 +     a3*16/self.Ts**4 +     a4*8/self.Ts**3)
        c1 = self.Ts *self.m * (   -7 *a0*128/self.Ts**7 - 5 * a1*64/self.Ts**6 - 3 * a2*32/self.Ts**5 -     a3*16/self.Ts**4 +     a4*8/self.Ts**3)
        c2 = self.Ts *self.m * (   21 *a0*128/self.Ts**7 + 9 * a1*64/self.Ts**6 +     a2*32/self.Ts**5 - 3 * a3*16/self.Ts**4 - 3 * a4*8/self.Ts**3)
        c3 = self.Ts *self.m * (   -35*a0*128/self.Ts**7 - 5 * a1*64/self.Ts**6 + 5 * a2*32/self.Ts**5 + 3 * a3*16/self.Ts**4 - 3 * a4*8/self.Ts**3)
        c4 = self.Ts *self.m * (   35 *a0*128/self.Ts**7 - 5 * a1*64/self.Ts**6 - 5 * a2*32/self.Ts**5 + 3 * a3*16/self.Ts**4 + 3 * a4*8/self.Ts**3)
        c5 = self.Ts *self.m * (   -21*a0*128/self.Ts**7 + 9 * a1*64/self.Ts**6 -     a2*32/self.Ts**5 - 3 * a3*16/self.Ts**4 + 3 * a4*8/self.Ts**3)
        c6 = self.Ts *self.m * (   7  *a0*128/self.Ts**7 - 5 * a1*64/self.Ts**6 + 3 * a2*32/self.Ts**5 -     a3*16/self.Ts**4 -     a4*8/self.Ts**3)
        c7 = self.Ts *self.m * (   -   a0*128/self.Ts**7 +     a1*64/self.Ts**6 -     a2*32/self.Ts**5 +     a3*16/self.Ts**4 -     a4*8/self.Ts**3)
        c8 = self.Ts *self.m * (   0  )
        return [d0, d1, d2, d3, d4, d5, d6, d7, d8], [c0, c1, c2, c3, c4, c5, c6, c7, c8]

    def Gn(self):
        b0 = self.Ts**2 
        b1 = 2*self.Ts**2
        b2 = self.Ts**2
        a0 = self.m*4
        a1 = -self.m*8
        a2 = self.m*4
        return [b0,b1,b2],[a0,a1,a2]

    def M(self): #M = (Gn^-1 G - I) --> (1,11) , (1,11)
        b , a = self.HG()
        d , c = self.Gn()

        Num = [c[0]*b[0]                         - d[0]*a[0],
               c[0]*b[1] + c[1]*b[0]             - d[0]*a[1] - d[1]*a[0],
               c[0]*b[2] + c[1]*b[1] + c[2]*b[0] - d[0]*a[2] - d[1]*a[1] - d[2]*a[0],
               c[0]*b[3] + c[1]*b[2] + c[2]*b[1] - d[0]*a[3] - d[1]*a[2] - d[2]*a[1],
               c[0]*b[4] + c[1]*b[3] + c[2]*b[2] - d[0]*a[4] - d[1]*a[3] - d[2]*a[2],
               c[0]*b[5] + c[1]*b[4] + c[2]*b[3] - d[0]*a[5] - d[1]*a[4] - d[2]*a[3],
               c[0]*b[6] + c[1]*b[5] + c[2]*b[4] - d[0]*a[6] - d[1]*a[5] - d[2]*a[4],
               c[0]*b[7] + c[1]*b[6] + c[2]*b[5] - d[0]*a[7] - d[1]*a[6] - d[2]*a[5],
               c[0]*b[8] + c[1]*b[7] + c[2]*b[6] - d[0]*a[8] - d[1]*a[7] - d[2]*a[6],
                         + c[1]*b[8] + c[2]*b[7]             - d[1]*a[8] - d[2]*a[7],
                                     + c[2]*b[8]                         - d[2]*a[8]]
        Den = [d[0]*a[0],   
               d[0]*a[1] + d[1]*a[0],
               d[0]*a[2] + d[1]*a[1] + d[2]*a[0],
               d[0]*a[3] + d[1]*a[2] + d[2]*a[1],
               d[0]*a[4] + d[1]*a[3] + d[2]*a[2],
               d[0]*a[5] + d[1]*a[4] + d[2]*a[3],
               d[0]*a[6] + d[1]*a[5] + d[2]*a[4],
               d[0]*a[7] + d[1]*a[6] + d[2]*a[5],
               d[0]*a[8] + d[1]*a[7] + d[2]*a[6],
                         + d[1]*a[8] + d[2]*a[7],
                                     + d[2]*a[0]]
        return Num, Den

    def IQM(self): #(I + GM)^-1 --> (1,19) , (1,19)
        b , a = self.HG()
        d , c = self.M()

        Num = [c[0]*a[0]                                                                                                                          + d[0]*b[0],
               c[0]*a[1] + c[1]*a[0]                                                                                                              + d[0]*b[1] + d[1]*b[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0]                                                                                                  + d[0]*b[2] + d[1]*b[1] + d[2]*b[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + c[3]*a[0]                                                                                      + d[0]*b[3] + d[1]*b[2] + d[2]*b[1] + d[3]*b[0],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + c[3]*a[1] + c[4]*a[0]                                                                          + d[0]*b[4] + d[1]*b[3] + d[2]*b[2] + d[3]*b[1] + d[4]*b[0],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + c[3]*a[2] + c[4]*a[1] + c[5]*a[0]                                                              + d[0]*b[5] + d[1]*b[4] + d[2]*b[3] + d[3]*b[2] + d[4]*b[1] + d[5]*b[0],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + c[3]*a[3] + c[4]*a[2] + c[5]*a[1] + c[6]*a[0]                                                  + d[0]*b[6] + d[1]*b[5] + d[2]*b[4] + d[3]*b[3] + d[4]*b[2] + d[5]*b[1] + d[6]*b[0],
               c[0]*a[7] + c[1]*a[6] + c[2]*a[5] + c[3]*a[4] + c[4]*a[3] + c[5]*a[2] + c[6]*a[1] + c[7]*a[0]                                      + d[0]*b[7] + d[1]*b[6] + d[2]*b[5] + d[3]*b[4] + d[4]*b[3] + d[5]*b[2] + d[6]*b[1] + d[7]*b[0],
               c[0]*a[8] + c[1]*a[7] + c[2]*a[6] + c[3]*a[5] + c[4]*a[4] + c[5]*a[3] + c[6]*a[2] + c[7]*a[1] + c[8]*a[0]                          + d[0]*b[8] + d[1]*b[7] + d[2]*b[6] + d[3]*b[5] + d[4]*b[4] + d[5]*b[3] + d[6]*b[2] + d[7]*b[1] + d[8]*b[0],
                         + c[1]*a[8] + c[2]*a[7] + c[3]*a[6] + c[4]*a[5] + c[5]*a[4] + c[6]*a[3] + c[7]*a[2] + c[8]*a[1] + c[9]*a[0]                          + d[1]*b[8] + d[2]*b[7] + d[3]*b[6] + d[4]*b[5] + d[5]*b[4] + d[6]*b[3] + d[7]*b[2] + d[8]*b[1] + d[9]*b[0],
                                     + c[2]*a[8] + c[3]*a[7] + c[4]*a[6] + c[5]*a[5] + c[6]*a[4] + c[7]*a[3] + c[8]*a[2] + c[9]*a[1] + c[10]*a[0]                         + d[2]*b[8] + d[3]*b[7] + d[4]*b[6] + d[5]*b[5] + d[6]*b[4] + d[7]*b[3] + d[8]*b[2] + d[9]*b[1] + d[10]*b[0],
                                                 + c[3]*a[8] + c[4]*a[7] + c[5]*a[6] + c[6]*a[5] + c[7]*a[4] + c[8]*a[3] + c[9]*a[2] + c[10]*a[1]                                     + d[3]*b[8] + d[4]*b[7] + d[5]*b[6] + d[6]*b[5] + d[7]*b[4] + d[8]*b[3] + d[9]*b[2] + d[10]*b[1],
                                                             + c[4]*a[8] + c[5]*a[7] + c[6]*a[6] + c[7]*a[5] + c[8]*a[4] + c[9]*a[3] + c[10]*a[2]                                                 + d[4]*b[8] + d[5]*b[7] + d[6]*b[6] + d[7]*b[5] + d[8]*b[4] + d[9]*b[3] + d[10]*b[2],
                                                                         + c[5]*a[8] + c[6]*a[7] + c[7]*a[6] + c[8]*a[5] + c[9]*a[4] + c[10]*a[3]                                                             + d[5]*b[8] + d[6]*b[7] + d[7]*b[6] + d[8]*b[5] + d[9]*b[4] + d[10]*b[3],
                                                                                     + c[6]*a[8] + c[7]*a[7] + c[8]*a[6] + c[9]*a[5] + c[10]*a[4]                                                                         + d[6]*b[8] + d[7]*b[7] + d[8]*b[6] + d[9]*b[5] + d[10]*b[4],
                                                                                                 + c[7]*a[8] + c[8]*a[7] + c[9]*a[6] + c[10]*a[5]                                                                                     + d[7]*b[8] + d[8]*b[7] + d[9]*b[6] + d[10]*b[5],
                                                                                                             + c[8]*a[8] + c[9]*a[7] + c[10]*a[6]                                                                                                 + d[8]*b[8] + d[9]*b[7] + d[10]*b[6],
                                                                                                                         + c[9]*a[8] + c[10]*a[7]                                                                                                             + d[9]*b[8] + d[10]*b[7],
                                                                                                                                     + c[10]*a[8]                                                                                                                         + d[10]*b[8]]

        Den = [c[0]*a[0],
               c[0]*a[1] + c[1]*a[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + c[3]*a[0],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + c[3]*a[1] + c[4]*a[0],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + c[3]*a[2] + c[4]*a[1] + c[5]*a[0],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + c[3]*a[3] + c[4]*a[2] + c[5]*a[1] + c[6]*a[0],
               c[0]*a[7] + c[1]*a[6] + c[2]*a[5] + c[3]*a[4] + c[4]*a[3] + c[5]*a[2] + c[6]*a[1] + c[7]*a[0],
               c[0]*a[8] + c[1]*a[7] + c[2]*a[6] + c[3]*a[5] + c[4]*a[4] + c[5]*a[3] + c[6]*a[2] + c[7]*a[1] + c[8]*a[0],
                         + c[1]*a[8] + c[2]*a[7] + c[3]*a[6] + c[4]*a[5] + c[5]*a[4] + c[6]*a[3] + c[7]*a[2] + c[8]*a[1] + c[9]*a[0],
                                     + c[2]*a[8] + c[3]*a[7] + c[4]*a[6] + c[5]*a[5] + c[6]*a[4] + c[7]*a[3] + c[8]*a[2] + c[9]*a[1] + c[10]*a[0],
                                                 + c[3]*a[8] + c[4]*a[7] + c[5]*a[6] + c[6]*a[5] + c[7]*a[4] + c[8]*a[3] + c[9]*a[2] + c[10]*a[1],
                                                             + c[4]*a[8] + c[5]*a[7] + c[6]*a[6] + c[7]*a[5] + c[8]*a[4] + c[9]*a[3] + c[10]*a[2],
                                                                         + c[5]*a[8] + c[6]*a[7] + c[7]*a[6] + c[8]*a[5] + c[9]*a[4] + c[10]*a[3],
                                                                                     + c[6]*a[8] + c[7]*a[7] + c[8]*a[6] + c[9]*a[5] + c[10]*a[4],
                                                                                                 + c[7]*a[8] + c[8]*a[7] + c[9]*a[6] + c[10]*a[5],
                                                                                                             + c[8]*a[8] + c[9]*a[7] + c[10]*a[6],
                                                                                                                         + c[9]*a[8] + c[10]*a[7],
                                                                                                                                     + c[10]*a[8]]
        return Den, Num

    def H(self):
        b , a = self.IQM()
        d , c = self.HG()

        Num = P.polymul(b,d)
        Den = P.polymul(a,c)

        return Num, Den

    def f_Controller(self, x):
        return {
            'PD': self.PD(),
            'PDF': self.PDF(),
            'PID': self.PID(),
            'PIDF': self.PIDF(),
        }[x]

    def f_DO(self, x):
        return {
            False: [np.append(np.zeros((1,9)),self.HG()[0]) , np.append(np.zeros((1,9)),self.HG()[1])],
            True: self.H(),
        }[x]

    def CL(self):
        b , a = self.f_DO(self.DO)
        d , c = self.f_Controller(self.Controller)

        Num = [c[0]*a[0],
               c[0]*a[1] + c[1]*a[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + c[3]*a[0],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + c[3]*a[1],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + c[3]*a[2],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + c[3]*a[3],
               c[0]*a[7] + c[1]*a[6] + c[2]*a[5] + c[3]*a[4],
               c[0]*a[8] + c[1]*a[7] + c[2]*a[6] + c[3]*a[5],
               c[0]*a[8] + c[1]*a[8] + c[2]*a[7] + c[3]*a[6],
               c[0]*a[10]+ c[1]*a[9] + c[2]*a[8] + c[3]*a[7],
               c[0]*a[11]+ c[1]*a[10]+ c[2]*a[9] + c[3]*a[8],
               c[0]*a[12]+ c[1]*a[11]+ c[2]*a[10]+ c[3]*a[9],
               c[0]*a[13]+ c[1]*a[12]+ c[2]*a[11]+ c[3]*a[10],
               c[0]*a[14]+ c[1]*a[13]+ c[2]*a[12]+ c[3]*a[11],
               c[0]*a[15]+ c[1]*a[14]+ c[2]*a[13]+ c[3]*a[12],
               c[0]*a[16]+ c[1]*a[15]+ c[2]*a[14]+ c[3]*a[13],
               c[0]*a[17]+ c[1]*a[16]+ c[2]*a[15]+ c[3]*a[14],
               c[0]*a[18]+ c[1]*a[17]+ c[2]*a[16]+ c[3]*a[15],
                         + c[1]*a[18]+ c[2]*a[17]+ c[3]*a[16],
                                     + c[2]*a[18]+ c[3]*a[17],
                                                 + c[3]*a[18]]

        Den = [c[0]*a[0]                                     + d[0]*b[0],
               c[0]*a[1] + c[1]*a[0]                         + d[0]*b[1] + d[1]*b[0],
               c[0]*a[2] + c[1]*a[1] + c[2]*a[0]             + d[0]*b[2] + d[1]*b[1] + d[2]*b[0],
               c[0]*a[3] + c[1]*a[2] + c[2]*a[1] + c[3]*a[0] + d[0]*b[3] + d[1]*b[2] + d[2]*b[1] + d[3]*b[0],
               c[0]*a[4] + c[1]*a[3] + c[2]*a[2] + c[3]*a[1] + d[0]*b[4] + d[1]*b[3] + d[2]*b[2] + d[3]*b[1],
               c[0]*a[5] + c[1]*a[4] + c[2]*a[3] + c[3]*a[2] + d[0]*b[5] + d[1]*b[4] + d[2]*b[3] + d[3]*b[2],
               c[0]*a[6] + c[1]*a[5] + c[2]*a[4] + c[3]*a[3] + d[0]*b[6] + d[1]*b[5] + d[2]*b[4] + d[3]*b[3],
               c[0]*a[7] + c[1]*a[6] + c[2]*a[5] + c[3]*a[4] + d[0]*b[7] + d[1]*b[6] + d[2]*b[5] + d[3]*b[4],
               c[0]*a[8] + c[1]*a[7] + c[2]*a[6] + c[3]*a[5] + d[0]*b[8] + d[1]*b[7] + d[2]*b[6] + d[3]*b[5],
               c[0]*a[8] + c[1]*a[8] + c[2]*a[7] + c[3]*a[6] + d[0]*b[9] + d[1]*b[8] + d[2]*b[7] + d[3]*b[6],
               c[0]*a[10]+ c[1]*a[9] + c[2]*a[8] + c[3]*a[7] + d[0]*b[10]+ d[1]*b[9] + d[2]*b[8] + d[3]*b[7],
               c[0]*a[11]+ c[1]*a[10]+ c[2]*a[9] + c[3]*a[8] + d[0]*b[11]+ d[1]*b[10]+ d[2]*b[9] + d[3]*b[8],
               c[0]*a[12]+ c[1]*a[11]+ c[2]*a[10]+ c[3]*a[9] + d[0]*b[12]+ d[1]*b[11]+ d[2]*b[10]+ d[3]*b[9],
               c[0]*a[13]+ c[1]*a[12]+ c[2]*a[11]+ c[3]*a[10]+ d[0]*b[13]+ d[1]*b[12]+ d[2]*b[11]+ d[3]*b[10],
               c[0]*a[14]+ c[1]*a[13]+ c[2]*a[12]+ c[3]*a[11]+ d[0]*b[14]+ d[1]*b[13]+ d[2]*b[12]+ d[3]*b[11],
               c[0]*a[15]+ c[1]*a[14]+ c[2]*a[13]+ c[3]*a[12]+ d[0]*b[15]+ d[1]*b[14]+ d[2]*b[13]+ d[3]*b[12],
               c[0]*a[16]+ c[1]*a[15]+ c[2]*a[14]+ c[3]*a[13]+ d[0]*b[16]+ d[1]*b[15]+ d[2]*b[14]+ d[3]*b[13],
               c[0]*a[17]+ c[1]*a[16]+ c[2]*a[15]+ c[3]*a[14]+ d[0]*b[17]+ d[1]*b[16]+ d[2]*b[15]+ d[3]*b[14],
               c[0]*a[18]+ c[1]*a[17]+ c[2]*a[16]+ c[3]*a[15]+ d[0]*b[18]+ d[1]*b[17]+ d[2]*b[16]+ d[3]*b[15],
                         + c[1]*a[18]+ c[2]*a[17]+ c[3]*a[16]            + d[1]*b[18]+ d[2]*b[17]+ d[3]*b[16],
                                     + c[2]*a[18]+ c[3]*a[17]                        + d[2]*b[17]+ d[3]*b[17],
                                                 + c[3]*a[18]                                    + d[3]*b[18]]
        return Num, Den

