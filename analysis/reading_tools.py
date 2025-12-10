import numpy as np

# Function reads a .dat file and returns a dictionary with the information from the file
def readDatFile(fileName):

    # Print introductory message
    
    print("#####################")
    print("---------------------")
    print("Running readDatFile.py...")
    print("\tInput:")
    print("\t\tfileName:\t\txlo = "+str(fileName))
    print("---------------------")


    # Open dat file, read and store data, close the file.
    f=open(fileName)
    readData=f.readlines()
    f.close()

    # Initialize data storage dictionary
    data={}

    # Initialize line counter
    counter=1

    # Initialize boolean keys
    booleanKeys={}
    booleanKeys["Header"]=False
    booleanKeys["Masses"]=False
    booleanKeys["Nonbond Coeffs"]=False
    booleanKeys["Pair Coeffs"]=False
    booleanKeys["Bond Coeffs"]=False
    booleanKeys["Angle Coeffs"]=False
    booleanKeys["Dihedral Coeffs"]=False
    booleanKeys["Improper Coeffs"]=False
    booleanKeys["BondBond Coeffs"]=False
    booleanKeys["BondAngle Coeffs"]=False
    booleanKeys["MiddleBondTorsion Coeffs"]=False
    booleanKeys["EndBondTorsion Coeffs"]=False
    booleanKeys["AngleTorsion Coeffs"]=False
    booleanKeys["AngleAngleTorsion Coeffs"]=False
    booleanKeys["BondBond13 Coeffs"]=False
    booleanKeys["AngleAngle Coeffs"]=False
    booleanKeys["Atoms"]=False
    booleanKeys["Velocities"]=False
    booleanKeys["Bonds"]=False
    booleanKeys["Angles"]=False
    booleanKeys["Dihedrals"]=False
    booleanKeys["Impropers"]=False
    booleanKeys["BlankLine"]=False

    # go through each line from the data file
    for line in readData:
        if not line.startswith("#"):
            if "#" in line:
                line,comment=line.split("#")

            if len(line.strip())==0 or line in ['\n', '\r\n']:
                booleanKeys["BlankLine"]=True
            else:
                booleanKeys["BlankLine"]=False

            if counter==1:
                comment=line
                comment=comment+" read by readDatFile.py (JPMC)"
                data["comment"]=comment
                counter+=1
                booleanKeys["Header"]=True
            elif len(line.strip())==0 or line in ['\n', '\r\n']:
                booleanKeys["BlankLine"]=True
            elif "Masses" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Masses"]=True
                data["Masses"]={}
            elif "Nonbond Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Nonbond Coeffs"]=True
                data["Nonbond Coeffs"]={}
            elif "Pair Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Pair Coeffs"]=True
                data["Pair Coeffs"]={}
            elif "Bond Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Bond Coeffs"]=True
                data["Bond Coeffs"]={}
            elif "Angle Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Angle Coeffs"]=True
                data["Angle Coeffs"]={}
            elif "Dihedral Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Dihedral Coeffs"]=True
                data["Dihedral Coeffs"]={}
            elif "Improper Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Improper Coeffs"]=True
                data["Improper Coeffs"]={}
            elif "BondBond Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["BondBond Coeffs"]=True
                data["BondBond Coeffs"]={}
            elif "BondAngle Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["BondAngle Coeffs"]=True
                data["BondAngle Coeffs"]={}
            elif "MiddleBondTorsion Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["MiddleBondTorsion Coeffs"]=True
                data["MiddleBondTorsion Coeffs"]={}
            elif "EndBondTorsion Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["EndBondTorsion Coeffs"]=True
                data["EndBondTorsion Coeffs"]={}
            elif "AngleTorsion Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["AngleTorsion Coeffs"]=True
                data["AngleTorsion Coeffs"]={}
            elif "AngleAngleTorsion Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["AngleAngleTorsion Coeffs"]=True
                data["AngleAngleTorsion Coeffs"]
            elif "BondBond13 Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["BondBond13 Coeffs"]=True
                data["BondBond13 Coeffs"]={}
            elif "AngleAngle Coeffs" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["AngleAngle Coeffs"]=True
                data["AngleAngle Coeffs"]={}
            elif "Atoms" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Atoms"]=True
                data["Atoms"]={}
            elif "Velocities" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Velocities"]=True
                data["Velocities"]={}
            elif "Bonds" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Bonds"]=True
                data["Bonds"]={}
            elif "Angles" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Angles"]=True
                data["Angles"]={}
            elif "Dihedrals" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Dihedrals"]=True
                data["Dihedrals"]={}
            elif "Impropers" in line:
                for key in booleanKeys:
                    booleanKeys[key]=False
                booleanKeys["Impropers"]=True
                data["Impropers"]={}
            
            
            elif booleanKeys["Header"]:
                if not booleanKeys["BlankLine"]:
                    if "atoms" in line:
                        number,stuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# atoms"]=number
                    elif "bonds" in line:
                        number,stuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# bonds"]=number
                    elif "angles" in line:
                        number,stuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# angles"]=number
                    elif "dihedrals" in line:
                        number,stuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# dihedrals"]=number
                    elif "impropers" in line:
                        number,stuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# impropers"]=number
                    elif "atom types" in line:
                        number,stuff,moreStuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# atom types"]=number
                    elif "bond types" in line:
                        number,stuff,moreStuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# bond types"]=number
                    elif "angle types" in line:
                        number,stuff,moreStuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# angle types"]=number
                    elif "dihedral types" in line:
                        number,stuff,moreStuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# dihedral types"]=number
                    elif "improper types" in line:
                        number,stuff,moreStuff=line.split()
                        number=number.strip()
                        number=int(number)
                        data["# improper types"]=number
                    elif "xlo" in line:
                        lo,hi,stuff,stuff=line.split()
                        lo=lo.strip()
                        hi=hi.strip()
                        lo=np.double(lo)
                        hi=np.double(hi)
                        data["xlo"]=lo
                        data["xhi"]=hi
                    elif "ylo" in line:
                        lo,hi,stuff,stuff=line.split()
                        lo=lo.strip()
                        hi=hi.strip()
                        lo=np.double(lo)
                        hi=np.double(hi)
                        data["ylo"]=lo
                        data["yhi"]=hi
                    elif "zlo" in line:
                        lo,hi,stuff,stuff=line.split()
                        lo=lo.strip()
                        hi=hi.strip()
                        lo=np.double(lo)
                        hi=np.double(hi)
                        data["zlo"]=lo
                        data["zhi"]=hi
            
            elif booleanKeys["Masses"]:

                if not booleanKeys["BlankLine"]:
                    line=line.strip()
                    aType,mass=line.split(" ",2)
                    aType=aType.strip()
                    mass=mass.strip()
                    aType=int(aType)
                    mass=np.double(mass)
                    data["Masses"][aType]=mass
            
            # Assume LJ potentials
            elif booleanKeys["Nonbond Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    aType,coeff1,coeff2=line.split()
                    aType=aType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    aType=int(aType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Nonbond Coeffs"][aType]={}
                    data["Nonbond Coeffs"][aType]["coeff1"]=coeff1
                    data["Nonbond Coeffs"][aType]["coeff2"]=coeff2
                
            elif booleanKeys["Pair Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    aType,coeff1,coeff2=line.split()
                    aType=aType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    aType=int(aType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Pair Coeffs"][aType]={}
                    data["Pair Coeffs"][aType]["coeff1"]=coeff1
                    data["Pair Coeffs"][aType]["coeff2"]=coeff2
            
            elif booleanKeys["Bond Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    bType,coeff1,coeff2=line.split()
                    bType=bType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    bType=int(bType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Bond Coeffs"][bType]={}
                    data["Bond Coeffs"][bType]["coeff1"]=coeff1
                    data["Bond Coeffs"][bType]["coeff2"]=coeff2
            
            elif booleanKeys["Angle Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    anType,coeff1,coeff2=line.split()
                    anType=anType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    anType=int(anType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Angle Coeffs"][anType]={}
                    data["Angle Coeffs"][anType]["coeff1"]=coeff1
                    data["Angle Coeffs"][anType]["coeff2"]=coeff2
            
            elif booleanKeys["Dihedral Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Dihedral Coeffs"][dType]={}
                    data["Dihedral Coeffs"][dType]["coeff1"]=coeff1
                    data["Dihedral Coeffs"][dType]["coeff2"]=coeff2
            
            elif booleanKeys["Improper Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    iType,coeff1,coeff2=line.split()
                    iType=iType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    iType=int(iType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["Improper Coeffs"][iType]={}
                    data["Improper Coeffs"][iType]["coeff1"]=coeff1
                    data["Improper Coeffs"][iType]["coeff2"]=coeff2
            
            elif booleanKeys["BondBond Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    anType,coeff1,coeff2=line.split()
                    anType=anType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    anType=int(anType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["BondBond Coeffs"][anType]={}
                    data["BondBond Coeffs"][anType]["coeff1"]=coeff1
                    data["BondBond Coeffs"][anType]["coeff2"]=coeff2
            
            elif booleanKeys["BondAngle Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    anType,coeff1,coeff2=line.split()
                    anType=anType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    anType=int(anType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["BondAngle Coeffs"][anType]={}
                    data["BondAngle Coeffs"][anType]["coeff1"]=coeff1
                    data["BondAngle Coeffs"][anType]["coeff2"]=coeff2
            
            elif booleanKeys["MiddleBondTorsion Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["MiddleBondTorsion Coeffs"][dType]={}
                    data["MiddleBondTorsion Coeffs"][dType]["coeff1"]=coeff1
                    data["MiddleBondTorsion Coeffs"][dType]["coeff2"]=coeff2
                    
            elif booleanKeys["EndBondTorsion Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["EndBondTorsion Coeffs"][dType]={}
                    data["EndBondTorsion Coeffs"][dType]["coeff1"]=coeff1
                    data["EndBondTorsion Coeffs"][dType]["coeff2"]=coeff2
                    
            elif booleanKeys["AngleTorsion Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["AngleTorsion Coeffs"][dType]={}
                    data["AngleTorsion Coeffs"][dType]["coeff1"]=coeff1
                    data["AngleTorsion Coeffs"][dType]["coeff2"]=coeff2
            
            elif booleanKeys["AngleAngleTorsion Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["AngleAngleTorsion Coeffs"][dType]={}
                    data["AngleAngleTorsion Coeffs"][dType]["coeff1"]=coeff1
                    data["AngleAngleTorsion Coeffs"][dType]["coeff2"]=coeff2
                    
            elif booleanKeys["BondBond13 Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    dType,coeff1,coeff2=line.split()
                    dType=dType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    dType=int(dType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["BondBond13 Coeffs"][dType]={}
                    data["BondBond13 Coeffs"][dType]["coeff1"]=coeff1
                    data["BondBond13 Coeffs"][dType]["coeff2"]=coeff2
                            
            elif booleanKeys["AngleAngle Coeffs"]:
                if not booleanKeys["BlankLine"]:
                    iType,coeff1,coeff2=line.split()
                    iType=iType.strip()
                    coeff1=coeff1.strip()
                    coeff2=coeff2.strip()
                    iType=int(iType)
                    coeff1=np.double(coeff1)
                    coeff2=np.double(coeff2)
                    data["AngleAngle Coeffs"][iType]={}
                    data["AngleAngle Coeffs"][iType]["coeff1"]=coeff1
                    data["AngleAngle Coeffs"][iType]["coeff2"]=coeff2
            
            elif booleanKeys["Atoms"]:
                if not booleanKeys["BlankLine"]:
                    if len(line.split())==7:
                        aID,mID,aType,q,x,y,z=line.split()
                    elif len(line.split())==10:
                        aID,mID,aType,q,x,y,z,nx,ny,nz=line.split()
                    aID=int(aID.strip())
                    mID=int(mID.strip())
                    aType=int(aType.strip())
                    q=np.double(q.strip())
                    x=np.double(x.strip())
                    y=np.double(y.strip())
                    z=np.double(z.strip())
                    data["Atoms"][aID]={}
                    data["Atoms"][aID]["molecule ID"]=mID
                    data["Atoms"][aID]["atom type"]=aType
                    data["Atoms"][aID]["charge"]=q
                    data["Atoms"][aID]["x"]=x
                    data["Atoms"][aID]["y"]=y
                    data["Atoms"][aID]["z"]=z
                    if len(line.split())==10:
                        nx=np.double(nx.strip())
                        ny=np.double(ny.strip())
                        nz=np.double(nz.strip())
                        data["Atoms"][aID]["nx"]=nx
                        data["Atoms"][aID]["ny"]=ny
                        data["Atoms"][aID]["nz"]=nz

            elif booleanKeys["Velocities"]:
                if not booleanKeys["BlankLine"]:
                    aID,vx,vy,vz=line.split()
                    aID=int(aID.strip())
                    vx=np.double(vx.strip())
                    vy=np.double(vy.strip())
                    vz=np.double(vz.strip())
                    data["Velocities"][aID]={}
                    data["Velocities"][aID]["vx"]=vx
                    data["Velocities"][aID]["vy"]=vy
                    data["Velocities"][aID]["vz"]=vz

            elif booleanKeys["Bonds"]:
                if not booleanKeys["BlankLine"]:
                    bID,bType,a1,a2=line.split()
                    bID=int(bID.strip())
                    bType=int(bType.strip())
                    a1=int(a1.strip())
                    a2=int(a2.strip())
                    data["Bonds"][bID]={}
                    data["Bonds"][bID]["bond type"]=bType
                    data["Bonds"][bID]["atom 1"]=a1
                    data["Bonds"][bID]["atom 2"]=a2
            
            elif booleanKeys["Angles"]:
                if not booleanKeys["BlankLine"]:
                    anID,anType,a1,a2,a3=line.split()
                    anID=int(anID.strip())
                    anType=int(anType.strip())
                    a1=int(a1.strip())
                    a2=int(a2.strip())
                    a3=int(a3.strip())
                    data["Angles"][anID]={}
                    data["Angles"][anID]["angle type"]=anType
                    data["Angles"][anID]["atom 1"]=a1
                    data["Angles"][anID]["atom 2"]=a2
                    data["Angles"][anID]["atom 3"]=a3

            elif booleanKeys["Dihedrals"]:
                if not booleanKeys["BlankLine"]:
                    dID,dType,a1,a2,a3,a4=line.split()
                    dID=int(dID.strip())
                    dType=int(dType.strip())
                    a1=int(a1.strip())
                    a2=int(a2.strip())
                    a3=int(a3.strip())
                    a4=int(a4.strip())
                    data["Dihedrals"][dID]={}
                    data["Dihedrals"][dID]["dihedral type"]=dType
                    data["Dihedrals"][dID]["atom 1"]=a1
                    data["Dihedrals"][dID]["atom 2"]=a2
                    data["Dihedrals"][dID]["atom 3"]=a3
                    data["Dihedrals"][dID]["atom 4"]=a4
                
            elif booleanKeys["Impropers"]:
                if not booleanKeys["BlankLine"]:
                    iID,iType,a1,a2,a3,a4=line.split()
                    iID=int(iID.strip())
                    iType=int(iType.strip())
                    a1=int(a1.strip())
                    a2=int(a2.strip())
                    a3=int(a3.strip())
                    a4=int(a4.strip())
                    data["Impropers"][iID]={}
                    data["Impropers"][iID]["improper type"]=iType
                    data["Impropers"][iID]["atom 1"]=a1
                    data["Impropers"][iID]["atom 2"]=a2
                    data["Impropers"][iID]["atom 3"]=a3
                    data["Impropers"][iID]["atom 4"]=a4

    # Print concluding message
    print("---------------------")
    print("readDatFile: OK")
    print("#####################")
                    
    return data

# Function reads a .trj file from LAMMPS and returns a numpy array with the trajectory info. 
def readTRJFile(fileName,output_IDMap=False,total_number_of_atoms=None):
    # Open trj file, read and store data, close the file.
    print("Opening and reading file...\n")
    print("Successfully read file.\n")

    local_ID=0
    local2global={}
    global2local={}

    # Initialize Boolean keys
    timeStep=False
    numOfAtoms=False
    box=False
    atoms=False

    # Initialize data storage array
    TRJdata=[]

    tsCounter=0

    if output_IDMap:
        NUMBER_OF_ATOMS=[]
        with open(fileName) as infile:
            for line in infile:
                # Detect TIMESTEP title line, turn on timeStep boolean key, turn off atoms boolean key
                if "TIMESTEP" in line:
                    timeStep=True
                    atoms=False
                # Detect NUMBER OF ATOMS line, turn on numOfAtoms boolean key, turn off timeStep boolean key
                elif "NUMBER OF ATOMS" in line:
                    numOfAtoms=True
                    timeStep=False
                # Detect BOX line, turn on box boolean key, turn off numOfAtoms boolean key
                elif "BOX" in line:
                    box=True
                    numOfAtoms=False
                    # initialize counter at 0 to store box size in correct dimmension (x=0,y=1,z=1)
                    counter=0
                # Detect ATOMS id line, turn on atoms boolean key, turn off box boolean key
                elif "ITEM: ATOMS id" in line:
                    atoms=True
                    box=False
                    # initialize counter at 0 to initialize "atoms" subdictionary when reading the first atom
                    counter=0
                # If the numOfAtoms boolean key is on, read and store the number of atoms
                elif numOfAtoms:
                    natoms=line.split()
                    natoms=''.join(natoms)
                    natoms=int(natoms)
                    NUMBER_OF_ATOMS.append(natoms)
                elif atoms:
                    id,atype,x,y,z=line.split()
                    id=int(id)
                    if output_IDMap:
                        if id not in global2local:
                            #print("global id "+str(id)+"not in global2local...")
                            local_ID+=1
                            #print("\tAssigning local id "+str(local_ID)+" to "+str(id))
                            global2local[id]=local_ID
                            local2global[local_ID]=id
                    elif not output_IDMap:
                        global2local[id]=id
                        local2global[id]=id

        print("max number of atoms: "+str(max(NUMBER_OF_ATOMS)))
        print("max local atom id "+str(local_ID))



    with open(fileName) as infile:
        for line in infile:
            # Detect TIMESTEP title line, turn on timeStep boolean key, turn off atoms boolean key
            if "TIMESTEP" in line:
                timeStep=True
                if tsCounter>0:
                    TRJdata.append(trajectory)
                tsCounter+=1
                #if tsCounter%10==0:
                #print("\tTS Lines Read: "+str(tsCounter)+"\n")
                atoms=False
            # Detect NUMBER OF ATOMS line, turn on numOfAtoms boolean key, turn off timeStep boolean key
            elif "NUMBER OF ATOMS" in line:
                numOfAtoms=True
                timeStep=False
            # Detect BOX line, turn on box boolean key, turn off numOfAtoms boolean key
            elif "BOX" in line:
                box=True
                numOfAtoms=False
                # initialize counter at 0 to store box size in correct dimmension (x=0,y=1,z=1)
                counter=0
            # Detect ATOMS id line, turn on atoms boolean key, turn off box boolean key
            elif "ITEM: ATOMS id" in line:
                atoms=True
                box=False
                # initialize counter at 0 to initialize "atoms" subdictionary when reading the first atom
                counter=0
            # If the timeStep boolean key is on, read and store the timestep
            elif timeStep:
                timestep=line.split()
                timestep=''.join(timestep)
                timestep=int(timestep)
            # If the numOfAtoms boolean key is on, read and store the number of atoms
            elif numOfAtoms:
                natoms=line.split()
                natoms=''.join(natoms)
                natoms=int(natoms)
            # If the box boolean key is on, check the counter to store under right direction (x=0,y=1,z=2)
            elif box:
                # if counter is zero, read and store xlo and xhi, and add one to the counter to move on to y
                if counter==0:
                    xlo,xhi=line.split()
                    xlo=xlo.strip()
                    xhi=xhi.strip()
                    xlo=np.double(xlo)
                    xhi=np.double(xhi)
                    counter+=1
                # if counter is one, read and store ylo and yhi, and add one to the counter to move on to z
                elif counter==1:
                    ylo,yhi=line.split()
                    ylo=ylo.strip()
                    yhi=yhi.strip()
                    ylo=np.double(ylo)
                    yhi=np.double(yhi)
                    counter+=1
                # if counter is two, read and store zlo and zhi (no need to add to counter, assume there will never be another line before atoms section header)
                elif counter==2:
                    zlo,zhi=line.split()
                    zlo=zlo.strip()
                    zhi=zhi.strip()
                    zlo=np.double(zlo)
                    zhi=np.double(zhi)
            # if atoms boolean key is on, create "atoms" subdictionary on the first line, and read and store the id, type, x, y, z of each atom
            elif atoms:
                if counter==0:
                    if not output_IDMap:
                        number_of_atoms=natoms
                    else:
                        number_of_atoms=int(local_ID)
                    trajectory=np.zeros((number_of_atoms, 4))
                    counter+=1
                id,atype,x,y,z=line.split()
                id=int(id)

                atype=int(atype)
                x=np.double(x)
                y=np.double(y)
                z=np.double(z)
                #print("atom "+str(id)+" is : "+str(IDMap[id])+" , total atoms: "+str(natoms))
                trajectory[global2local[id]-1]=[atype,x,y,z]
        print("Array was successfully created.")
        # after all lines have been read and stored appropriately, return data
        print("Converting to numpy array...")
        TRJdata=np.array(TRJdata)
    
    if not output_IDMap:
        return TRJdata
    else:
        return TRJdata, local2global, global2local

# Function reads a .trj file with per atom stress from LAMMPS and returns a numpy array with the stress trajectory info.
# FUTURE WORK: combine this function with readTRJFile since there is a lot of redundance
def readTRJFile_stresses(fileName,output_IDMap=False,total_number_of_atoms=None):
    # Open trj file, read and store data, close the file.
    print("Opening and reading file...\n")
    print("Successfully read file.\n")

    local_ID=0
    local2global={}
    global2local={}

    # Initialize Boolean keys
    timeStep=False
    numOfAtoms=False
    box=False
    atoms=False

    # Initialize data storage array
    TRJdata=[]

    tsCounter=0

    if output_IDMap:
        NUMBER_OF_ATOMS=[]
        with open(fileName) as infile:
            for line in infile:
                # Detect TIMESTEP title line, turn on timeStep boolean key, turn off atoms boolean key
                if "TIMESTEP" in line:
                    timeStep=True
                    atoms=False
                # Detect NUMBER OF ATOMS line, turn on numOfAtoms boolean key, turn off timeStep boolean key
                elif "NUMBER OF ATOMS" in line:
                    numOfAtoms=True
                    timeStep=False
                # Detect BOX line, turn on box boolean key, turn off numOfAtoms boolean key
                elif "BOX" in line:
                    box=True
                    numOfAtoms=False
                    # initialize counter at 0 to store box size in correct dimmension (x=0,y=1,z=1)
                    counter=0
                # Detect ATOMS id line, turn on atoms boolean key, turn off box boolean key
                elif "ITEM: ATOMS id" in line:
                    atoms=True
                    box=False
                    # initialize counter at 0 to initialize "atoms" subdictionary when reading the first atom
                    counter=0
                # If the numOfAtoms boolean key is on, read and store the number of atoms
                elif numOfAtoms:
                    natoms=line.split()
                    natoms=''.join(natoms)
                    natoms=int(natoms)
                    NUMBER_OF_ATOMS.append(natoms)
                elif atoms:
                    id,atype,xx,yy,zz,xy,xz,yz=line.split()
                    id=int(id)
                    if output_IDMap:
                        if id not in global2local:
                            #print("global id "+str(id)+"not in global2local...")
                            local_ID+=1
                            #print("\tAssigning local id "+str(local_ID)+" to "+str(id))
                            global2local[id]=local_ID
                            local2global[local_ID]=id
                    elif not output_IDMap:
                        global2local[id]=id
                        local2global[id]=id

        print("max number of atoms: "+str(max(NUMBER_OF_ATOMS)))
        print("max local atom id "+str(local_ID))



    with open(fileName) as infile:
        for line in infile:
            # Detect TIMESTEP title line, turn on timeStep boolean key, turn off atoms boolean key
            if "TIMESTEP" in line:
                timeStep=True
                if tsCounter>0:
                    TRJdata.append(trajectory)
                tsCounter+=1
                #if tsCounter%10==0:
                #print("\tTS Lines Read: "+str(tsCounter)+"\n")
                atoms=False
            # Detect NUMBER OF ATOMS line, turn on numOfAtoms boolean key, turn off timeStep boolean key
            elif "NUMBER OF ATOMS" in line:
                numOfAtoms=True
                timeStep=False
            # Detect BOX line, turn on box boolean key, turn off numOfAtoms boolean key
            elif "BOX" in line:
                box=True
                numOfAtoms=False
                # initialize counter at 0 to store box size in correct dimmension (x=0,y=1,z=1)
                counter=0
            # Detect ATOMS id line, turn on atoms boolean key, turn off box boolean key
            elif "ITEM: ATOMS id" in line:
                atoms=True
                box=False
                # initialize counter at 0 to initialize "atoms" subdictionary when reading the first atom
                counter=0
            # If the timeStep boolean key is on, read and store the timestep
            elif timeStep:
                timestep=line.split()
                timestep=''.join(timestep)
                timestep=int(timestep)
            # If the numOfAtoms boolean key is on, read and store the number of atoms
            elif numOfAtoms:
                natoms=line.split()
                natoms=''.join(natoms)
                natoms=int(natoms)
            # If the box boolean key is on, check the counter to store under right direction (x=0,y=1,z=2)
            elif box:
                # if counter is zero, read and store xlo and xhi, and add one to the counter to move on to y
                if counter==0:
                    xlo,xhi=line.split()
                    xlo=xlo.strip()
                    xhi=xhi.strip()
                    xlo=np.double(xlo)
                    xhi=np.double(xhi)
                    counter+=1
                # if counter is one, read and store ylo and yhi, and add one to the counter to move on to z
                elif counter==1:
                    ylo,yhi=line.split()
                    ylo=ylo.strip()
                    yhi=yhi.strip()
                    ylo=np.double(ylo)
                    yhi=np.double(yhi)
                    counter+=1
                # if counter is two, read and store zlo and zhi (no need to add to counter, assume there will never be another line before atoms section header)
                elif counter==2:
                    zlo,zhi=line.split()
                    zlo=zlo.strip()
                    zhi=zhi.strip()
                    zlo=np.double(zlo)
                    zhi=np.double(zhi)
            # if atoms boolean key is on, create "atoms" subdictionary on the first line, and read and store the id, type, x, y, z of each atom
            elif atoms:
                if counter==0:
                    if not output_IDMap:
                        number_of_atoms=natoms
                    else:
                        number_of_atoms=int(local_ID)
                    trajectory=np.zeros((number_of_atoms, 8))
                    counter+=1
                id,atype,xx,yy,zz,xy,xz,yz=line.split()
                id=int(id)

                atype=int(atype)
                xx=np.double(xx)
                yy=np.double(yy)
                zz=np.double(zz)
                xy=np.double(xy)
                xz=np.double(xz)
                yz=np.double(yz)
                #print("atom "+str(id)+" is : "+str(IDMap[id])+" , total atoms: "+str(natoms))
                trajectory[global2local[id]-1]=[atype,xx,yy,zz,xy,xz,yz]
        print("Array was successfully created.")
        # after all lines have been read and stored appropriately, return data
        print("Converting to numpy array...")
        TRJdata=np.array(TRJdata)
    
    if not output_IDMap:
        return TRJdata
    else:
        return TRJdata, local2global, global2local