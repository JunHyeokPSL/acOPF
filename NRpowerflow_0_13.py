"""
Programmed on 29/Nov 2021

version 0.11
@author: JunHyeok.Kim
"""

import numpy as np
import pandas as pd
import math as m

def findBus(a, buses):  # return index from zero!!!!!!!
    for bus in buses:
        if bus.I == a:
            return buses.index(bus)
           
# def findSlackBus(buses):  # return index from zero!!!!!!!
#     for bus in buses:
#         if bus.IDE == 1:
#             return buses.index(bus)

class Busdata:
    def __init__(self, I, TYPE, VM, VA, VMAX, VMIN):
        self.I = I  # BUS NUMBER
        self.TYPE = TYPE  # BUS TYPE
        # BUS TYPE CODE 1-SLACK 2-GENERATOR(PV) 3-LOAD(PQ) 4-DISCONNECT

        self.VM = VM  # VOLTAGE MAGNITUDE
        self.VA = VA  # VOLTAGE PHASE IN DEGREE
        self.VMAX = VMAX #VOLTAGE MAX
        self.VMIN = VMIN #VOLTAGE MIN

class Loaddata:
    def __init__(self, I, PL, QL):
        self.I = I  # LOAD NUMBER
        self.PL = PL  # LOAD ACTIVE POWER IN MW
        self.QL = QL  # LOAD REACTIVE POWER IN MW
        

class Fixedshuntdata:
    def __init__(self, I, ID, STATUS, GL, BL):
        self.I = I  # BUS NUMBER
        self.ID = ID
        self.STATUS = STATUS
        self.GL = GL
        self.BL = BL


class Generatordata:
    def __init__(self, I, PG, QG):
        self.I = I
        self.PG = PG
        self.QG = QG

class Branchdata:
    def __init__(self, I, J, R, X, B):
        self.I = I  # FROM BUS NUMBER
        self.J = J  # TO BUS NUMBER
        self.R = R  # RESISTANCE
        self.X = X  # REACTANCE
        self.B = B  # SUSCEPTANCE

class DERdata:
    def __init__(self,I, TYPE, COMPANY, PG, QG):
        self.I = I
        self.TYPE = TYPE
        self.COMPANY = COMPANY
        self.PG = PG
        self.QG = QG

class LoadProfiledata:
    def __init__(self,TIME, FACTOR):
        self.TIME = TIME
        self.FACTOR = FACTOR
    
class PVProfiledata:
    def __init__(self,TIME, FACTOR):
        self.TIME = TIME
        self.FACTOR = FACTOR

#Read Data
# DATA I/O

def Preprocessing():
    
    DLdata = pd.read_excel('data/DLdata.xlsx',sheet_name=7,skiprows=2,usecols="B:O")
    DLdata.columns = ['ganzi','fromtype','frombusname','From','totype','tobusname','To','exDL','status','sectionload','linetype','vline','mline','length']
    
    LineImpeadance = pd.read_excel('data/DLdata.xlsx',sheet_name=8)
    LineImpeadance.columns = ['vltype','vline','mltype','mline','R','X1','X','R0','X0','Current']
    Bus = []
    busnum = 1
    k=1
    busdata1=[]
    busdata2=[]
    
    ## Bus rename
    for i in range(len(DLdata.From)):
        if DLdata.From[i] == DLdata.To[i]:
            newbus = '%s_new'
            
            if DLdata.From[i+2] == DLdata.From[i]:
                DLdata.iloc[i,6] = newbus%(DLdata.From[i])
                DLdata.iloc[i+1,3] = newbus%(DLdata.From[i])
                DLdata.iloc[i+2,3] = newbus%(DLdata.From[i])
            else:
                DLdata.iloc[i,6] = newbus%(DLdata.From[i])
                DLdata.iloc[i+1,3] = newbus%(DLdata.From[i])
    
    DLdata.iloc[31,6]= DLdata.iloc[32,3] = '11L1R5L14H1-1'
    DLdata.iloc[33,6]= DLdata.iloc[34,3] = '11L1R5L14H1-2'
    
    ## Bus numbering 
    
    for i in range(len(DLdata.From)):   
        if DLdata.To[i] == '고객':
            cust = '고객%s'
            Bus.append(cust%(k))
            DLdata.iloc[i,6]=cust%(k)
            k+=1
            busnum+=1
    
        if DLdata.From[i] in Bus:
            for j in range(len(Bus)):
                if Bus[j] == DLdata.From[i]:
                    busdata1.append(j+1)   
        else:
            Bus.append(DLdata.From[i])
            busdata1.append(busnum)
            busnum += 1
    
        if DLdata.To[i] in Bus:
            for j in range(len(Bus)):
                if Bus[j] == DLdata.To[i]:
                    busdata2.append(j+1)
        else:
            Bus.append(DLdata.To[i])
            busdata2.append(busnum)
            busnum += 1
                    
    
    busdat = pd.DataFrame([busdata1,busdata2],index=['From','to'])
    busdata = busdat.transpose()
    
    ##Calculate Reactance/Resistance
    
    Resistance=['']
    Reactance=['']
    for i in range(len(DLdata.From)):
        if DLdata.vline[i] == 400:
            Resistance.append(LineImpeadance.R[5]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[5]*DLdata.length[i]/100)
        elif DLdata.vline[i] == 325:
            Resistance.append(LineImpeadance.R[4]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[4]*DLdata.length[i]/100)       
        elif DLdata.vline[i] == 160:
            Resistance.append(LineImpeadance.R[3]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[3]*DLdata.length[i]/100)
        elif DLdata.vline[i] == 95:
            Resistance.append(LineImpeadance.R[2]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[2]*DLdata.length[i]/100)
        elif DLdata.vline[i] == 60:
            Resistance.append(LineImpeadance.R[1]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[1]*DLdata.length[i]/100)
        elif DLdata.vline[i] == 58:
            Resistance.append(LineImpeadance.R[0]*DLdata.length[i]/100)
            Reactance.append(LineImpeadance.X1[0]*DLdata.length[i]/100)
            
    del Resistance[0]
    del Reactance[0]
    
    Lindata1 = pd.DataFrame([DLdata.From, DLdata.To],index=['From','to'])
    Lindata2 = pd.DataFrame([Resistance, Reactance],index=['Line_Resistance','Line_Reactance'])
    Lindata = pd.concat([Lindata1,Lindata2])
    Lindata3 = Lindata2.transpose()
    line_data = pd.concat([busdata,Lindata3],axis=1)
    
    ## Calculate Load
    sectload=pd.DataFrame([DLdata.sectionload])
    Load=pd.concat([busdata,sectload.transpose()],axis=1)
    connect=[]
    Power=[float(0) for i in range(len(Bus))]
    for i in range(len(Bus)):
        m=0
        for j in range(len(Load.From)):
            if (Load.From[j] == i+1 or Load.to[j] == i+1):
                m+=1
        connect.append(m)
        
    try:
            
        for i in range(len(Bus)):
            if connect[i]==1:
                for j in range(len(Load.From)):
                    if (Load.From[j] == i+1 or Load.to[j] == i+1):
                        Power[i+1] = Load.sectionload[j] + Load.sectionload[j+1]/2
            
            else:
                for j in range(len(Load.From)):        
                    if (Load.From[j] == i+2 or Load.to[j] == i+2):
                       Power[i+1] = Power [i+1] + Load.sectionload[j]/2
    except:
        pass
    
    
    busindex=[]
    bustype=[]     
    PG=[]
    QG=[]
    PL=[]
    QL=[]
    V=[]
    theta=[]
    print(type(QL))
    # SL = PL + jQL (ASSUMED PF = 0.9)  
    for i in range(len(Power)):
        PL.append(0.9*Power[i]/10000)
        QL.append((np.sqrt(Power[i]**2 - (10000*PL[i])**2))/10000)     
    
    # MAPPING THE DATA TO LIST   
    for i in range (len(Bus)):
        busindex.append(i+1)
        V.append(1.039)
        theta.append(0)
        PG.append(0)
        QG.append(0)
        if i == 0:
            bustype.append(1)
        else:
            bustype.append(3)             
    
    a = pd.DataFrame([busindex,bustype,PG,QG,PL,QL,V,theta],index=['no','Bustype','PG','QG','PL','QL','Vpu','Vtheta'])          
    bus_data = a.transpose()
    bus_data=bus_data.astype({'Bustype':'int'})
    bus_data=bus_data.astype({'no':'int'})    
    
    return [bus_data, line_data]

def Read_File(fileURL):
    #systemMVAbase = 0
    URL = fileURL[0]
    DATA = fileURL[1]
    if DATA == 'KEPCO_BUS':
        if type(URL) == str:
            KEPCO_busdata = pd.read_excel(URL)
        else:
            KEPCO_busdata = URL
        buses = []
        loads = []
        generators = []
        for i in range(len(KEPCO_busdata)):    
            #Bus Data
            buses.append(
                Busdata(
                    int(KEPCO_busdata['no'][i]),                
                    int(KEPCO_busdata['Bustype'][i]),
                    float(KEPCO_busdata['Vpu'][i]),
                    float(KEPCO_busdata['Vtheta'][i])
                    )
                )
     
            # Generator Data
            generators.append(
                Generatordata(
                    int(KEPCO_busdata['no'][i]),
                    float(KEPCO_busdata['PG'][i]),
                    float(KEPCO_busdata['QG'][i])
                    )
                )
            
            # Load Data
            loads.append(
                Loaddata(
                    int(KEPCO_busdata['no'][i]),
                    float(KEPCO_busdata['PL'][i]),
                    float(KEPCO_busdata['QL'][i])
                    )
                )
                
        return [buses, generators, loads]
        
    elif DATA == 'KEPCO_LINE':
        if type(URL) == str:
            KEPCO_linedata = pd.read_excel(URL)
        else:
            KEPCO_linedata = URL
        branches = []
        for i in range(len(KEPCO_linedata)):
            branches.append(
                Branchdata(
                    int(KEPCO_linedata['From'][i]),   # From
                    int(KEPCO_linedata['to'][i]),   # To
                    float(KEPCO_linedata['Line_Resistance'][i]),  # R
                    float(KEPCO_linedata['Line_Reactance'][i]),  # X
                    float(0)  # B
                    )
                )
        return branches        
        
    else:      
        with open(URL, 'r') as rawdata:
            if(DATA == 'Bus'):
                lines = rawdata.readlines()
                buses = []
                loads = []
                generators = []
                for line in lines:
                    a = line.split('\t')
               
                    #Bus Data
                    buses.append(
                        Busdata(
                            int(a[0]),                
                            int(a[1]),
                            float(a[6]),
                            float(a[7]),
                            1.039,
                            0.912
                            )
                        )
             
                    # Generator Data
                    generators.append(
                        Generatordata(
                            int(a[0]),
                            float(a[2]),
                            float(a[3])
                            )
                        )
                    
                    # Load Data
                    loads.append(
                        Loaddata(
                            int(a[0]),
                            float(a[4]),
                            float(a[5])
                            )
                        )
                        
                return [buses, generators, loads]
                                   
            elif(DATA == 'Line'):
                lines = rawdata.readlines()
                branches = []
                for line in lines:
                    a = line.split('\t')
                    branches.append(
                        Branchdata(
                            int(a[0]),   # From
                            int(a[1]),   # To
                            float(a[2]),  # R
                            float(a[3]),  # X
                            float(a[4])  # B
                            )
                        )
                return branches
            
            elif(DATA == 'LoadProfile'):
                lines = rawdata.readlines()
                loadprofiles = []
                t = 0
                for line in lines:
                    a = line.split('\t')
                    loadprofiles.append(
                        LoadProfiledata(
                            int(t),
                            float(a[0])
                            )
                        )
                    t = t + 1
                return loadprofiles
            elif(DATA == 'PVProfile'):
                lines = rawdata.readlines()
                pvprofiles = []
                t = 0
                for line in lines:
                    a = line.split('\t')
                    pvprofiles.append(
                        LoadProfiledata(
                            int(t),
                            float(a[0])
                            )
                        )
                    t = t + 1
                return pvprofiles
        
        
def DERmapping(DER_List):
    ders = []
    for i in range(len(DER_List)):
        ders.append(
            DERdata(
                int(DER_List[i,0]),
                str(DER_List[i,1]),
                str(DER_List[i,2]),
                float(DER_List[i,3]),
                float(DER_List[i,4])
                )
            )
    return ders

def Generate_Busmatrix(buses, sets, factors, profiles): 
    
    loads = sets[0]
    generators = sets[1]
    ders = sets [2]
    
    generator_factor = factors [0]
    load_factor = factors[1]
    
    loadprofile = profiles[0]
    pvprofile = profiles[1]
    
    busnumber = [0] * len(buses)
    bustype = [0] * len(buses)
    PG = [0] * len(buses)  
    QG = [0] * len(buses)
    PL = [0] * len(buses)  
    QL = [0] * len(buses)
    VM = [1] * len(buses)
    VA = [0] * len(buses)
    
    for bus in buses:
        busnumber[findBus(bus.I,buses)] = bus.I
        bustype[findBus(bus.I,buses)] = bus.TYPE
        VM[findBus(bus.I,buses)] = bus.VM 
        VA[findBus(bus.I,buses)] =bus.VA
    
    for load in loads:

        PL[findBus(load.I, buses)] += load.PL*load_factor*loadprofile
        QL[findBus(load.I, buses)] += load.QL*load_factor*loadprofile
        
    # GENERATOR P/Q FOR EACH BUS
    for generator in generators:
        PG[findBus(generator.I, buses)] += generator.PG*generator_factor
        QG[findBus(generator.I, buses)] += generator.QG*generator_factor
    # DER P/Q FOR EACH BUS
    for der in ders:
        if der.TYPE == 'PV':
            PG[findBus(der.I, buses)] += der.PG*generator_factor*pvprofile
            QG[findBus(der.I, buses)] += der.QG*generator_factor*pvprofile
        else:
            PG[findBus(der.I, buses)] += der.PG*generator_factor
            QG[findBus(der.I, buses)] += der.QG*generator_factor   
    busmatrix = pd.DataFrame([busnumber,bustype,PG,QG,PL,QL,VM,VA], index=['BUS','TYPE','PG','QG','PL','QL','VM','VA']).transpose()
    busmatrix = busmatrix.astype({'BUS':'int', 'TYPE':'int'})  
      
    return busmatrix    

def Generate_DERmatrix(ders):
    busnumber = []
    dertype = [] 
    company = []
    PG = []
    QG = []
    
    for der in ders:
        busnumber.append(der.I)
        dertype.append(der.TYPE)
        company.append(der.COMPANY)
        PG.append(der.PG)
        QG.append(der.QG)
        
    dermatrix = pd.DataFrame([busnumber,dertype,company,PG,QG], index=['BUS','TYPE','COMPANY','PG','QG']).transpose()    
    return dermatrix 

def Rearrange_bus(buses):
    orderedbuses = []
    Num_slack = 0
    Num_PQ = 0
    Num_PV = 0
    
    for bus in buses:
        if bus.TYPE == 1:  # SLACK BUS
            orderedbuses.append(bus)
            Num_slack = Num_slack + 1
    
    for bus in buses:
        if bus.TYPE == 3:  # PQ BUS
            orderedbuses.append(bus)
            Num_PQ = Num_PQ + 1
    
    for bus in buses:
        if bus.TYPE == 2:  # PV BUS
            orderedbuses.append(bus)
            Num_PV = Num_PV + 1    

    NumBus = [Num_slack, Num_PQ, Num_PV]
    return [orderedbuses, NumBus]

def PQcalculation(orderedbuses,VthData,Ymat):
    P = np.zeros(len(orderedbuses))
    Q = np.zeros(len(orderedbuses))    
    V = VthData[0]
    Theta = VthData[1]
    for i in range(len(orderedbuses)):
        for j in range(len(orderedbuses)):
            indexfori = findBus(orderedbuses[i].I, orderedbuses)
            indexforj = findBus(orderedbuses[j].I, orderedbuses)
            # THESE VARIABLES FOR SEARCHING YMatrix
            G = Ymat[indexfori][indexforj].real
            B = Ymat[indexfori][indexforj].imag
            th = Theta[i] - Theta[j]
            P[i] = P[i] + V[i] * V[j] * (G * m.cos(th) + B * m.sin(th))
            Q[i] = Q[i] + V[i] * V[j] * (G * m.sin(th) - B * m.cos(th))
    return [P,Q]

def delPQcalculation(orderedbuses, PQbus, PQ, NumBus):

    [Pbus, Qbus] = PQbus    
    [P,Q] = PQ
    [Num_slack, Num_PQ, Num_PV] = NumBus
    
    delP = np.zeros(Num_PQ + Num_PV)  # LEN(BUS) - 1 (NO SLACK BUS)
    delQ = np.zeros(Num_PQ)    # NO SLACK AND PV BUS        
    for i in range(Num_slack, len(orderedbuses)):
        # TO SEARCH FROM BUSESAP ARRAY
        indexfori = findBus(orderedbuses[i].I, orderedbuses)
        delP[i - Num_slack] = Pbus[indexfori] - P[i]

    for i in range(Num_slack, Num_slack + Num_PQ):
        # TO SEARCH FROM BUSESRP ARRAY
        indexfori = findBus(orderedbuses[i].I, orderedbuses)
        delQ[i - Num_slack] = Qbus[indexfori] - Q[i]
        
    return [delP, delQ]

def GenerateYmatrix(buses, branches):
    
    Ymat = np.zeros((len(buses), len(buses))).astype(complex)
    for branch in branches:
        # non-diagonal is minus
        Ymat[findBus(branch.I, buses)][findBus(branch.J, buses)] += - \
            1 / complex(branch.R, branch.X)
        Ymat[findBus(branch.J, buses)][findBus(branch.I, buses)] += - \
            1 / complex(branch.R, branch.X)
        # diagonal is plus
        Ymat[findBus(branch.I, buses)][findBus(branch.I, buses)] += (1 /
             complex(branch.R, branch.X) + 0.5 * complex(0, branch.B))
        Ymat[findBus(branch.J, buses)][findBus(branch.J, buses)] += (1 /
             complex(branch.R, branch.X) + 0.5 * complex(0, branch.B))
    return Ymat
                          
def GenerateJacobian(orderedbuses, BusData, Ymat, NumBus):
    
    P = BusData[0]
    Q = BusData[1]
    V = BusData[2]
    Theta = BusData[3]

    Num_PQ = NumBus[1]
    Num_PV = NumBus[2]
    
    j11 = np.zeros((Num_PQ + Num_PV, Num_PQ + Num_PV))
    j12 = np.ones((Num_PQ + Num_PV, Num_PQ ))
    j21 = np.ones((Num_PQ , Num_PQ + Num_PV))
    j22 = np.zeros((Num_PQ , Num_PQ))   
    
# Constructing J11 matrix
    for i in range(1, Num_PQ + Num_PV + 1):
        for j in range(1, Num_PQ + Num_PV + 1):
            indexfori = findBus(orderedbuses[i].I, orderedbuses)
            indexforj = findBus(orderedbuses[j].I, orderedbuses)
            # THESE VARIABLES FOR SEARCHING YMTRX
            G = Ymat[indexfori][indexforj].real
            B = Ymat[indexfori][indexforj].imag
            th = Theta[i] - Theta[j]

            if i != j:
                j11[i - 1][j - 1] = - V[i] * V[j] * \
                    (G * m.sin(th) - B * m.cos(th))

            if i == j:
                j11[i - 1][j - 1] = V[i] * V[i] * B + Q[i]

# Constructing J12 matrix
    for i in range(1, Num_PQ + Num_PV + 1):
        for j in range(1, Num_PQ + 1):
            indexfori = findBus(orderedbuses[i].I, orderedbuses)
            indexforj = findBus(orderedbuses[j].I, orderedbuses)
            # THESE VARIABLES FOR SEARCHING YMTRX
            G = Ymat[indexfori][indexforj].real
            B = Ymat[indexfori][indexforj].imag
            th = Theta[i] - Theta[j]
            if i != j:
                j12[i - 1][j - 1] = - V[i] * \
                    (G * m.cos(th) + B * m.sin(th))
            if i == j:
                j12[i - 1][j - 1] = - V[i] * G - P[i]/V[i]

# Constructing J21 matrix
    for i in range(1, Num_PQ + 1):
        for j in range(1, Num_PQ + Num_PV + 1):
            indexfori = findBus(orderedbuses[i].I, orderedbuses)
            indexforj = findBus(orderedbuses[j].I, orderedbuses)
            # THESE VARIABLES FOR SEARCHING YMTRX
            G = Ymat[indexfori][indexforj].real
            B = Ymat[indexfori][indexforj].imag
            th = Theta[i] - Theta[j]
            if i != j:
                j21[i - 1][j - 1] = V[i] * V[j] * \
                    (G * m.cos(th) + B * m.sin(th))
            if i == j:
                j21[i - 1][j - 1] = V[i] * V[i] * G - P[i]

# Constructing J22 matrix
    for i in range(1, Num_PQ + 1):
        for j in range(1, Num_PQ + 1):
            indexfori = findBus(orderedbuses[i].I, orderedbuses)
            indexforj = findBus(orderedbuses[j].I, orderedbuses)
            # THESE VARIABLES FOR SEARCHING YMTRX
            G = Ymat[indexfori][indexforj].real
            B = Ymat[indexfori][indexforj].imag
            th = Theta[i] - Theta[j]
            if i != j:
                j22[i - 1][j - 1] = - V[i]  * \
                    (G * m.sin(th) - B * m.cos(th))
            if i == j:
                j22[i - 1][j - 1] = V[i] * B - Q[i]/V[i]

    return [[j11,j12],[j21,j22]]

def SolvePowerFlow(orderedbuses,PQbus,Ymat,NumBus):
    
    count = 0 
    count_max = 20
    Error = 1
    
    # HERE, WE WILL USE THE ORDEREDBUSES ARRAY
    V = np.ones(len(orderedbuses))
    Theta = np.zeros(len(orderedbuses))
    for i in range(len(orderedbuses)):
        if orderedbuses[i].TYPE == 1:  # SLACK BUS
            V[i] = orderedbuses[i].VM
            Theta[i] = np.deg2rad(orderedbuses[i].VA)
        if orderedbuses[i].TYPE == 2:  # PV BUS
            V[i] = orderedbuses[i].VM
    
    Num_slack = NumBus[0]
    Num_PQ = NumBus[1]
    Num_PV = NumBus[2]    
    
    while Error > 0.0001 :
        #print(iteration)
        
        # UPDATING P AND Q
        
        [P,Q] = PQcalculation(orderedbuses, [V,Theta], Ymat)
        
        # UPDATING DELP AND DELQ
        PQ = [P,Q]
        [delP, delQ] = delPQcalculation(orderedbuses,PQbus,PQ,NumBus)
        delPQ = np.r_[delP, delQ]
         
        # GENERATE JACOBIAN MATRIX
        # [j11,j12,j21,j22] = [Jpth,Jpv,Jqth,Jqv]
        BusData = [P,Q,V,Theta]
        [[j11,j12],[j21,j22]] = GenerateJacobian(orderedbuses, BusData, Ymat, NumBus)
        # CONSTRUCTING JACOBIAN
        jacobian = np.r_[np.c_[j11, j12], np.c_[j21, j22]]
        # CALCULATE INVERSE JACOBIAN MATRIX
        InvJ= np.linalg.inv(jacobian)
     
        # CALCULATE delTh, delV BY INVERSE JACOBIAN MATRIX
        result = - np.dot(InvJ, delPQ)
        delTh, delV = np.split(result, [Num_PQ + Num_PV, ])
     
        # UPDATE Theta, V
        for i in range(Num_slack, len(orderedbuses)):
            Theta[i] = Theta[i] + delTh[i - Num_slack]
        for i in range(Num_slack, Num_slack + Num_PQ):
            V[i] = V[i] + delV[i - Num_slack]
        
        # ERROR OF ACTIVE POWER AND REACTIVE POWER (P,Q) 
        Error = np.max(np.abs(delPQ))
        count = count + 1
        if count > count_max:
            print("Cannot solve NR powerflow")
            break    
    return [V, Theta, [[j11,j12],[j21,j22]]]

def ResultVtheta(busmatrix):
    I = np.zeros(len(busmatrix),dtype=int)
    Theta = np.zeros(len(busmatrix))
    V = np.ones(len(busmatrix))

    for i in range(len(busmatrix)):
        I[i] = busmatrix['BUS'][i]
        Theta[i] = busmatrix['VA'][i]
        V[i] = busmatrix['VM'][i]
    return [I, V, Theta]
    