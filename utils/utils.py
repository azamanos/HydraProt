import gzip
import struct
import torch
import h5py
import numpy as np
import torch.nn.functional as F
from multiprocessing import Pool
import matplotlib.pyplot as plt

class Config(object):
    '''
    Empty class object to save various parameters for the training and inference of the models.
    '''
    def __init__(self):
        return

class PDB(object):
    '''
    Class object to load PDB structure files of pdb or cif format.

    Parameters
    ----------
    filename : str
        Path to file.

    ignore_waters : bool
        To ignore or not the waters in the file, default False.

    ignore_other_HETATM : bool
        To ignore or not the HETATMs in the file, excluding waters, default False.

    multi_models : bool
        To include in the data or not multiple models of the structure in the file, default False.

    Attributes
    ----------
    natoms : int
        Number of protein atoms of all the chains in the structure.

    water_coords : numpy array
        Coordinates of water atoms, note that here we only keep oxygen atoms, shape (W,3).

    wb : numpy array
        B factor of water coordinates, shape (W).

    HETATM_coords : numpy array
        Coordinates of HETATM atoms, excluding water molecules, shape (H,3).

    HETATM_name : numpy array
        Names of HETATM atoms, excluding water molecules, shape (H).

    HETATM_num : numpy array
        Number of HETATM instance, excluding water molecules, shape (H).

    HETATM_atomnum : numpy array
        Atom number of HETATM atoms, excluding water molecules, shape (H).

    HETATM_atomtype : numpy array
        Atom type of HETATM atoms, excluding water molecules, shape (H).

    SSE : numpy array
        Empty numpy array ready to keep SSE info, shape (N).

    resolution : float
        Resolution of structure.

    SSEraw : numpy array
        Info of ranges for SSE info, shape (S,3)

    atomnum : numpy array
        Atom number of protein atoms, shape (N).

    atomname : numpy array
        Atom name of protein atoms, shape (N).

    atomalt : numpy array
        Alternative atoms for protein atoms, shape (N).

    resname : numpy array
        Residue name that protein atom belongs, shape (N).

    atomtype : numpy array
        Atom type of protein atoms, shape (N).

    resnum : numpy array
        Residue number of protein residues, shape (N).

    resalt : numpy array
        Alternative residues for protein residues, shape (N).

    chain : numpy array
        Chain that atom belongs, shape (N).

    coords : numpy array
        Atom coordinates of protein atoms, shape (N,3).

    occupancy : numpy array
        Occupancy of protein atoms, shape (N).

    b : numpy array
        B factor of protein atoms, shape (N).

    self.cella : float
        Unit cell parameters, length of a axis.

    self.cellb : float
        Unit cell parameters, length of b axis.

    self.cellc : float
        Unit cell parameters, length of c axis.

    self.cellalpha : float
        Unit cell parameters, angle of a axis.

    self.cellbeta : float
        Unit cell parameters, angle of b axis.

    self.cellgamma : float
        Unit cell parameters, angle of c axis.
    '''
    def __init__(self, filename, ignore_waters=False, ignore_other_HETATM=False, multi_models=False):
        #Define lists and variables
        self.natoms, self.water_coords, self.wb = 0, [], []
        self.HETATM_coords, self.HETATM_name, self.HETATM_chain, self.HETATM_num, self.HETATM_atomnum, self.HETATM_atomtype = [],[],[],[],[],[]
        self.SSE, self.resolution, self.SSEraw = [], None, []
        self.atomnum, self.atomname, self.atomalt, self.resname, self.atomtype, self.resnum, self.resalt, self.chain, self.coords = [],[],[],[],[],[],[],[],[]
        self.occupancy, self.b  = [],[]
        #Check if file is in cif format
        cif = False
        if filename.split('.')[-1] == 'cif' or filename.split('.')[-2] == 'cif':
            cif = True
        #If file is compressed uncompress it
        if filename.split('.')[-1] == 'gz':
            with gzip.open(filename, 'rt', encoding='utf-8') as file:
                f = file.readlines()
        #Else just open it
        else:
            with open(filename, 'r') as file:
                f = file.readlines()
        #If the format is pdb
        if not cif:
            #Start reading the PDB file
            for i, line in enumerate(f):
                sline = line.split()
                #If line starts with 'ATOM'
                if line[:4]=='ATOM':
                    #Keep protein's atoms info.
                    self.atomnum.append(int(float(sline[1])))
                    self.atomname.append(line[12:16].strip())
                    self.atomalt.append(line[16])
                    self.resname.append(line[17:21].strip())
                    atomtype = sline[-1]
                    if len(atomtype)-1:
                        try:
                            int(atomtype[1])
                            atomtype = atomtype[0]
                        except:
                            atomtype = atomtype[0].upper() + atomtype[1].lower()
                    self.atomtype.append(atomtype)
                    self.resnum.append(int(float(line[22:26])))
                    self.resalt.append(line[26])
                    self.chain.append(line[21])
                    self.coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    self.occupancy.append(float(line[56:60]))
                    self.b.append(float(line[60:66]))
                    #self.charge[atom] = line[78:80].strip('\n')
                    #self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                    self.natoms += 1
                    continue
                #If line starts with 'HETATM'
                if line[:6] == 'HETATM':
                    #Keep waters
                    if not ignore_waters and line[13] == 'O' and ((line[17:20]=='HOH') or (line[17:20]=='TIP') or (line[17:20]=='WAT')):
                        self.water_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        self.wb.append(float(line[60:66]))
                        continue
                    #Keep the rest hetatm
                    if not ignore_other_HETATM and sline[-1][0]!='H':
                        self.HETATM_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        self.HETATM_name.append(line[17:21].strip())
                        self.HETATM_chain.append(line[21])
                        self.HETATM_num.append(int(float(line[22:26])))
                        self.HETATM_atomnum.append(int(float(line[6:12])))
                        atomtype = sline[-1]
                        if len(atomtype)-1:
                            try:
                                int(atomtype[1])
                                atomtype = atomtype[0]
                            except:
                                atomtype = atomtype[0].upper() + atomtype[1].lower()
                        self.HETATM_atomtype.append(atomtype)
                        continue
                #Here you can add more self objects from Header.
                #Resolution info in pdb format files.
                if line[:22] == 'REMARK   2 RESOLUTION.':
                    self.resolution = sline[3]
                    continue
                #Helix info in pdb format files.
                if line[:5] == 'HELIX':
                    self.SSEraw.append([range(int(line[20:25]),int(line[32:37])+1), line[19], 'H'])
                    continue
                #Beta sheet info in pdb format files.
                if line[:5] == 'SHEET':
                    self.SSEraw.append([range(int(line[22:26]),int(line[33:37])+1), line[21], 'S'])
                    continue
                #Crystal info in pdb format files.
                if line[:6] == 'CRYST1':
                    self.cella, self.cellb, self.cellc = float(sline[1]), float(sline[2]), float(sline[3])
                    self.cellalpha, self.cellbeta, self.cellgamma = float(sline[4]), float(sline[5]), float(sline[6])
                    continue
                #Return structure only of Model 1 if there are multiple Models
                if sline[0] == 'MODEL' and not cif and not multi_models and sline[1] != '1':
                    break
        #If the format is cif
        else:
            b_strands = False
            #Start reading the PDB file
            for i, line in enumerate(f):
                sline = line.split()
                #If line starts with 'ATOM'
                if line[:4]=='ATOM':
                    #First check if you have multiple models and you dont want them
                    if int(sline[-1])-1 and not multi_models:
                        break
                    self.atomnum.append(int(float(sline[1])))
                    self.atomname.append(sline[3])
                    self.atomalt.append(line[22])
                    self.resname.append(sline[5])
                    atomtype = sline[2]
                    if len(atomtype) == 2:
                        atomtype = atomtype[0].upper() + atomtype[1].lower()
                    self.atomtype.append(atomtype)
                    self.resnum.append(int(float(sline[-5])))
                    self.chain.append(sline[6])
                    self.coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                    self.occupancy.append(float(sline[13]))
                    self.b.append(float(sline[14]))
                    self.natoms += 1
                    continue
                #If line starts with 'HETATM'
                if line[:6] == 'HETATM':
                    if not ignore_waters and sline[2] == 'O' and ((sline[5]=='HOH') or (sline[5]=='TIP') or (sline[5]=='WAT')):
                        #Keep coordinates of water Oxygens.
                        self.water_coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                        self.wb.append(float(sline[14]))
                        continue
                    if not ignore_other_HETATM and sline[2]!='H':
                        self.HETATM_coords.append([float(sline[10]), float(sline[11]), float(sline[12])])
                        self.HETATM_name.append(sline[5])
                        self.HETATM_chain.append(sline[7])
                        self.HETATM_num.append(int(float(sline[16])))
                        self.HETATM_atomnum.append(int(float(sline[1])))
                        self.HETATM_atomtype.append(sline[2])
                        continue
                #Here you can add more self objects from Header.
                #Resolution info in cif format files.
                if line[:25] == '_reflns.d_resolution_high' or line[:33] == '_em_3d_reconstruction.resolution ':
                    self.resolution = sline[1]
                    continue
                #Helix info in cif format files.
                if line[:6] == 'HELX_P':
                    self.SSEraw.append([range(int(sline[-7]),int(sline[-4])+1), sline[-8], 'H'])
                    continue
                #Beta sheet info in cif format files.
                if line[:35] == '_struct_sheet_range.end_auth_seq_id':
                    b_strands = True
                    continue
                if b_strands:
                    if line[:1] == '#':
                        b_strands=False
                        continue
                    self.SSEraw.append([range(int(sline[-4]),int(sline[-1])+1), sline[-2], 'S'])
                    continue
                #Crystal info in cif format files.
                if line[:5] == '_cell':
                    if sline[0] == '_cell.length_a':
                        self.cella = float(sline[1])
                    if sline[0] == '_cell.length_b':
                        self.cellb = float(sline[1])
                    if sline[0] == '_cell.length_c':
                        self.cellc = float(sline[1])
                    if sline[0] == '_cell.angle_alpha':
                        self.cellalpha = float(sline[1])
                    if sline[0] == '_cell.angle_beta':
                        self.cellbeta = float(sline[1])
                    if sline[0] == '_cell.angle_gamma':
                        self.cellgamma = float(sline[1])
                    continue
        #Return structure when every atom and water atom have been searched.
        try:
            self.resolution = float(self.resolution)
        except:
            pass
        #Prepare SecondaryStructureElements info
        self.SSE = np.zeros((self.natoms), dtype=np.dtype((str,1)))
        self.SSEraw = np.array(self.SSEraw, dtype=object)
        #Turn every list to numpy array
        for attribute, value in vars(self).items():
            if type(value)==list:
                setattr(self, attribute, np.array(value))
        return

    def remove_waters(self):
        idx = np.where((self.resname=="HOH") | (self.resname=="TIP"))
        self.remove_atoms_from_object(idx)

    def remove_by_atomtype(self, atomtype):
        idx = np.where((self.atomtype==atomtype))
        self.remove_atoms_from_object(idx)

    def remove_by_atomname(self, atomname):
        idx = np.where((self.atomname==atomname))
        self.remove_atoms_from_object(idx)

    def remove_by_atomnum(self, atomnum):
        idx = np.where((self.atomnum==atomnum))
        self.remove_atoms_from_object(idx)

    def remove_by_resname(self, resname):
        idx = np.where((self.resname==resname))
        self.remove_atoms_from_object(idx)

    def remove_by_resnum(self, resnum):
        idx = np.where((self.resnum==resnum))
        self.remove_atoms_from_object(idx)

    def remove_by_chain(self, chain):
        idx = np.where((self.chain==chain))
        self.remove_atoms_from_object(idx)

    def remove_atoms_from_object(self, idx):
        mask = np.ones(self.natoms, dtype=bool)
        mask[idx] = False
        self.atomnum = self.atomnum[mask]
        self.atomname = self.atomname[mask]
        self.atomalt = self.atomalt[mask]
        self.resalt = self.resalt[mask]
        self.resname = self.resname[mask]
        self.resnum = self.resnum[mask]
        self.chain = self.chain[mask]
        self.coords = self.coords[mask]
        self.occupancy = self.occupancy[mask]
        self.b = self.b[mask]
        self.atomtype = self.atomtype[mask]
        self.SSE = self.SSE[mask]
        #self.charge = self.charge[mask]
        #self.nelectrons = self.nelectrons[mask]
        self.natoms = len(self.atomnum)

    def rearrange_resalt(self):
        #Keep indexes where you have added residues
        ind = np.where(self.resalt!=' ')[0]
        if not len(ind):
            return
        #Find chains of these added residues
        resalt_ch = self.chain[ind]
        #Find unique chains
        diff_chains = np.unique(resalt_ch)
        #For each unique chain id
        for ch in diff_chains:
            #Keep indexes of added residues only for your chain
            ind_resalt_ch = ind[np.where(resalt_ch==ch)]
            #Keep last index of your chain's residues
            last_ch_ind = np.where(self.chain==ch)[0][-1]+1
            #For each atom in chain in added residue
            for atom in ind_resalt_ch:
                #Find where added residue starts
                if self.resalt[atom-1]!=self.resalt[atom] and self.resnum[atom-1] == self.resnum[atom]:
                    #Add to all consquent residue +1 number
                    self.resnum[atom:last_ch_ind] += 1
                #If added residues are before residue number, change the numbering of the original residue +1
                if atom+1+1>self.natoms:
                    continue
                if self.resalt[atom+1] == ' ' and self.resnum[atom+1] == self.resnum[atom]:
                    self.resnum[atom+1:last_ch_ind] += 1
        #Remove alternative residues indexes
        self.resalt[ind] = ' '

def clean_pdb_alternative_atoms(pdb):
    '''
    Remove alterative atoms in pdb protein coordinates

    Parameters
    ----------
    pdb : class
        object of PDB class

    Returns
    -------
    pdb : class
        cleaned PDB class object from alternative atoms.
    '''
    delete = np.where(np.isin(pdb.atomalt, (' ', 'A', '.'))==False)[0]
    #If you have indexes to delete proceed
    if len(delete):
        #Remove selected atoms if any.
        pdb.remove_atoms_from_object(delete)
    return

def clean_pdb_atomname_and_resname(pdb):
    '''
    Remove non canonical atomnames and resnames.

    Parameters
    ----------
    pdb : class
        object of PDB class

    Returns
    -------
    pdb : class
        cleaned PDB class object from non canonical atomnames and resnames.
    '''
    #Find where you have non canonical atomnames and resnames
    non_canonical_atomnames = np.where(np.isin(pdb.atomname, atomnames)==False)[0]
    non_canonical_resnames = np.where(np.isin(pdb.resname, resnames)==False)[0]
    #If you have indexes
    if len(non_canonical_atomnames) or len(non_canonical_resnames):
        #Compute the union.
        delete = np.union1d(non_canonical_atomnames, non_canonical_resnames)
        #Remove selected atoms if any.
        pdb.remove_atoms_from_object(delete)
    return

def clean_pdb(pdb):
    '''
    Remove alterative atoms in pdb protein coordinates

    Parameters
    ----------
    pdb : class
        object of PDB class

    Returns
    -------
    pdb : class
        cleaned PDB class object from alternative atoms and non canonical atomnames and resnames.
    '''
    alt_atoms = np.where(np.isin(pdb.atomalt, (' ', 'A', '.'))==False)[0]
    #Find where you have non canonical atomnames and resnames
    non_canonical_atomnames = np.where(np.isin(pdb.atomname, atomnames)==False)[0]
    non_canonical_resnames = np.where(np.isin(pdb.resname, resnames)==False)[0]
    #If you have indexes to delete proceed
    if len(non_canonical_atomnames) or len(non_canonical_resnames) or len(alt_atoms):
        #Compute the union.
        delete = np.union1d(alt_atoms, np.union1d(non_canonical_atomnames, non_canonical_resnames))
        #Remove selected atoms if any.
        pdb.remove_atoms_from_object(delete)
    return

def coordinates_to_dmatrix(a_coords, b_coords):
    '''
    Creates distance matrix for numpy.array input.

    Parameters
    ----------
    a_coords : numpy.array
        numpy.array of shape (N,3) that contains coordinates information.

    b_coords : numpy.array
        numpy.array of shape (M,3) that contains coordinates information.

    Returns
    -------
    numpy.array of shape (N,M), the distance matrix of a_coords and b_coords.
    '''
    a, b = torch.from_numpy(a_coords), torch.from_numpy(b_coords)
    return np.array(torch.cdist(a,b))

def batchify_dmatrix_computation(a_coords, b_coords, batch_size=500):
    '''
    Function to compute faster d_min for large dmatrix, in order to keep memory low.

    Parameters
    ----------
    a_coords : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    b_coords : numpy.array
        numpy array of shape (M,3) that contains coordinates information.

    batch_size : int
        size of batch per dmatrix computation, default 500.

    Returns
    -------
    d_matrix : numpy.array
        distance matrix array of shape (N,M).
    '''
    #Keep length of a and b coordinates
    a_coords_len, b_coords_len = len(a_coords), len(b_coords)
    if a_coords_len<batch_size or b_coords_len<batch_size:
        return coordinates_to_dmatrix(a_coords, b_coords)
    #Initialize d_matrix
    d_matrix = np.zeros((a_coords_len,b_coords_len))
    #Find the batches according to batch size
    a_coords_batches, b_coords_batches = int(np.ceil(a_coords_len/batch_size))+1, int(np.ceil(b_coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    a_batch_loop, b_batch_loop = np.linspace(0, a_coords_len, a_coords_batches, dtype=int), np.linspace(0, b_coords_len, b_coords_batches, dtype=int)
    #Initialize d_min array
    d_min = np.zeros((len(a_batch_loop)-1, b_coords_len))
    #For each batch for a coords.
    for i, a_batch in enumerate(a_batch_loop[:-1]):
        #For each batch for b coords
        for j, b_batch in enumerate(b_batch_loop[:-1]):
            a_indexes, b_indexes = slice(a_batch, a_batch_loop[i+1]), slice(b_batch, b_batch_loop[j+1])
            #Compute dmatrix
            d_matrix[a_indexes,b_indexes] = coordinates_to_dmatrix(a_coords[a_indexes], b_coords[b_indexes])
    #Return the final d_matrix.
    return d_matrix

def batchify_dmatrix_computation_for_dmin(a_coords, b_coords, batch_size=500, axis=0):
    '''
    Function to compute faster d_min for large dmatrix, in order to keep memory low.

    Parameters
    ----------
    a_coords : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    b_coords : numpy.array
        numpy array of shape (M,3) that contains coordinates information.

    batch_size : int
        size of batch per dmatrix computation, default 500.

    axis : int
        axis along which you want to compute min values, default 0, corresponds to min of b_coords distance from a_coords.

    Returns
    -------
    dmin : numpy.array
        array of shape (1,M) for default value of axis 0, contains minimum distance of b_coords from all a_coords.
    '''
    #Keep length of a and b coordinates
    a_coords_len, b_coords_len = len(a_coords), len(b_coords)
    #Find the batches according to batch size
    a_coords_batches, b_coords_batches = int(np.ceil(a_coords_len/batch_size))+1, int(np.ceil(b_coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    a_batch_loop, b_batch_loop = np.linspace(0, a_coords_len, a_coords_batches, dtype=int), np.linspace(0, b_coords_len, b_coords_batches, dtype=int)
    #Initialize d_min array
    d_min = np.zeros((len(a_batch_loop)-1, b_coords_len))
    #For each batch for a coords.
    for i, a_batch in enumerate(a_batch_loop[:-1]):
        #For each batch for b coords
        for j, b_batch in enumerate(b_batch_loop[:-1]):
            #Compute dmatrix
            d = coordinates_to_dmatrix(a_coords[a_batch:a_batch_loop[i+1]], b_coords[b_batch:b_batch_loop[j+1]])
            #Find min values for axis
            d_min[i][b_batch:b_batch_loop[j+1]] = np.min(d,axis=axis)
    #Return the final d_min matrix.
    return np.min(d_min,axis=axis)

def preprocess_pdb_for_unet(pdb, radius, include_hetatm = True):
    '''
    Creates random transformation matrix with translation and rotation for 3D coordinates.

    Parameters
    ----------
    pdb : class
        PDB class object, parsed pdb file.

    radius : int
        Radius of waters around protein to keep.

    include_hetatm : bool
        To include hetatm coordinates or not, default True

    Returns
    -------
    pdb : class
        updated pdb.water_coords for your class object

    atom_hetatm_coords : numpy.array
        numpy array of shape (N,3) with coordinate information.

    atom_hetatm_atomtype : numpy.array
        numpy array of shape (N,1) with atomtype information.
    '''
    #Clean duplicate residues/atoms
    clean_pdb(pdb)
    #Concatenate atom and hetatm coordinates and atomtype information
    if include_hetatm and len(pdb.HETATM_atomtype):
        atom_hetatm_coords = np.concatenate((pdb.coords,pdb.HETATM_coords),axis=0)
        atom_hetatm_atomtype = np.concatenate((pdb.atomtype,pdb.HETATM_atomtype),axis=0)
    else:
        atom_hetatm_coords = pdb.coords
        atom_hetatm_atomtype = pdb.atomtype
    #Find where you have known atoms
    known_atomtypes = np.isin(atom_hetatm_atomtype, ['C','N','O','S'])
    known_indexes = np.where(known_atomtypes == True)[0]
    #unk_indexes = np.where(known_atomtypes == False)[0]
    atom_hetatm_coords, atom_hetatm_atomtype = atom_hetatm_coords[known_indexes], atom_hetatm_atomtype[known_indexes]
    atom_hetatm_atomtype[np.where(atom_hetatm_atomtype=='C')] = 0
    atom_hetatm_atomtype[np.where(atom_hetatm_atomtype=='N')] = 1
    atom_hetatm_atomtype[np.where(np.isin(atom_hetatm_atomtype, ['O','S']))] = 2
    return pdb, atom_hetatm_coords, atom_hetatm_atomtype.astype(int)

def random_transformation_matrix():
    '''
    Creates random transformation matrix with translation and rotation for 3D coordinates.

    Returns
    -------
    transformation_matrix : numpy.array
        numpy array of shape (4,4), with rotation information in the upper left (3,3) pixels and translation information in last column)
    '''
    transformation_matrix = np.zeros((4,4))
    #Compute random translation
    tx, ty, tz = np.random.rand(3)
    #Add translation to final column
    transformation_matrix[:,3] = tx, ty, tz, 1
    #Compute random rotation
    rx, ry, rz = np.random.rand(3)*2*np.pi
    #Compute cosines and sines
    cos_a, sin_a = np.cos(rz), np.sin(rz)
    cos_b, sin_b = np.cos(ry), np.sin(ry)
    cos_c, sin_c = np.cos(rx), np.sin(rx)
    #Add rotation to the 3,3 upper left matrix
    z_rotation = np.array([[cos_a,-sin_a, 0],[sin_a, cos_a, 0],[0, 0, 1]])
    y_rotation = np.array([[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 0, cos_b]])
    x_rotation = np.array([[1, 0, 0], [0, cos_c, -sin_c], [0, sin_c, cos_c]])
    #Compute final rotation matrix for all three axis
    rotation_matrix = z_rotation@y_rotation@x_rotation
    transformation_matrix[:3,:3] = rotation_matrix
    return transformation_matrix

def random_flip_matrix():
    '''
    Creates random flip matrix for 3D coordinates, flip operations can be combined for all three axis.

    Returns
    -------
    flip_matrix : numpy.array
        a diagonal numpy array of shape (3,3), with flip information in the diagonal, -1 flip, 1 no flip, axis x -> 0, y -> 1, z -> 2.
    '''
    flip_matrix = np.zeros((3,3))
    flip_matrix[0,0], flip_matrix[1,1], flip_matrix[2,2] = np.random.choice((1,-1)), np.random.choice((1,-1)), np.random.choice((1,-1))
    return flip_matrix

def create_3Dgrid(coordinates, vs, pad = 2):
    '''
    Create 3D matrix for your protein, with cubic voxel size and padding of your choice.

    Parameters
    ----------
    coordinates : numpy.array
        coordinates in the continues 3D space, (N,3) shape array.

    vs : float
        the voxel size of your predifined 3D grid, float.

    pad : int
        extra space around the edges of your protein coordinates, default 2.

    Returns
    -------
    grid : numpy.array
        3D array constructed to contain your protein.

    origin : numpy.array
        origin of your coordinates, given when 3D grid was created to place your coordinates, (3) shape array, tuple or list.

    grid_indexes : numpy.array
        numpy array of shape (N,3) with the indexes of the coordinates
    '''
    #Find max size of each side.
    true_dim = np.ceil(np.max(coordinates,0)-np.min(coordinates,0))
    #Calculate the dimensions of the 3D grid.
    grid_dim = np.ceil((true_dim+pad*2)/vs)
    #Expand towards 50 divided grid dimensions
    exp_grid_dim = (np.ceil(grid_dim/50)*50).astype(int)
    # exp_grid_dim = np.max(exp_grid_dim)
    # exp_grid_dim = (exp_grid_dim, exp_grid_dim, exp_grid_dim)
    #Compute final pad, first fix for voxel size, remember true_dim has to scaled by vs.
    exp_pad = ((exp_grid_dim-(true_dim/vs))/2)*vs
    #Compute origin
    origin = np.min(coordinates,0)-exp_pad
    #Create grid
    grid = np.zeros(exp_grid_dim)
    #Find the matching coordinates in your freshly created 3D grid.
    grid_indexes = coordinates_to_indexes(coordinates, origin, vs)
    #Mark voxels that contain protein atoms.
    grid[tuple(grid_indexes.T)] = 1
    return grid, origin, grid_indexes

def create_submaps(original_map, box_size = 64, core_size = 50, data_type = np.float32):
    '''
    Function that creates submaps from an original map given box and core size.

    Parameters
    ----------
    original_map : numpy.array
        numpy array of shape (M,K,L) be careful, all M,K,L have to %50 = 0.

    box size : int
        shape of box size, default 64

    core_size : int
        shape of core size, default 50

    data_type : type
        data type of array that will be returned, default np.float32.

    Returns
    -------
    submaps : numpy.array
        array that contains the submaps of the original map.
    '''
    #Keep image shape
    map_shape = np.shape(original_map)
    #Compute pad of the padded map
    pad = (box_size-core_size)//2
    #Initiliaze padded image of original map
    padded_map = np.pad(original_map, pad)
    #Initilize submaps list
    submaps = list()
    #Starting point in padded_map is 0
    x, y, z = 0, 0, 0
    #While z index is not violating the size of original image shape
    while (z < map_shape[2]):
        #Get the next chunk of the padded image
        next_submap = padded_map[x : x+box_size, y : y+box_size, z : z+box_size]
        #Appended to submaps list
        submaps.append(next_submap)
        #Continue to the next chunk on the x axis
        x += core_size
        #If x index extends the x max
        if x >= map_shape[0]:
            #Increment y axis to the next chunk
            y += core_size
            #And reset x axis
            x = 0
            #If y index also extends the y max
            if y >= map_shape[1]:
                #Increment z axis to the next chunk
                z += core_size
                #And reset x and y axis
                x, y = 0, 0
    #When every chunk has been remove return
    return np.array(submaps, dtype=data_type)

def reconstruct_map(submaps, box_size = 64, core_size = 50, o_dim = None, data_type = np.float32):
    '''
    Function that reconstructs original shape map from given submaps, box and core size.

    Parameters
    ----------
    submaps : numpy.array
        numpy array of shape (X,M,K,L) be careful, all M,K,L have to %50 = 0, X is the number of submaps.

    box size : int
        shape of box size, default 64

    core_size : int
        shape of core size, default 50

    o_dim : tuple
        tuple with the original shape of map, default None.

    data_type : type
        data type of array that will be returned, default np.float32.

    Returns
    -------
    reconstructed_map : numpy.array
        array of the recostructed original map.
    '''
    #If dimensions are not given probably dimensions are the same at x, y, z.
    if not o_dim:
        #Compute dimensions
        o_dim = int(np.shape(submaps)[0]**(0.333334))
        o_dim = [int(i * o_dim) for i in (50,50,50)]
    #Extraction start and end of the submap, remember you only need the core
    s = (box_size - core_size)//2
    e = s + core_size
    #Initialize reconstruction map
    reconstructed_map = np.zeros(tuple(o_dim))
    #Initialize counter
    i = 0
    #For each submap corresponding to z axis
    for z in range(o_dim[2]//core_size):
        #For each submap corresponding to y axis
        for y in range(o_dim[1]//core_size):
            #For each submap corresponding to x axis
            for x in range(o_dim[0]//core_size):
                #Fill reconstruction map
                reconstructed_map[x*core_size:(x+1)*core_size, y*core_size:(y+1)*core_size, z*core_size:(z+1)*core_size] = submaps[i][s:e, s:e, s:e]
                i += 1
    #Return reconstruction map with given data_type
    return np.array(reconstructed_map, dtype=np.float32)

def prepare_prediction_unet_data(atom_hetatm_coords, atom_hetatm_atomtype, unet_params):
    '''
    Creates data to input in the U-net model for prediction.

    Parameters
    ----------
    atom_hetatm_coords : numpy.array
        numpy array of shape (N,3) with coordinate information.

    atom_hetatm_atomtype : numpy.array
        numpy array of shape (N,1) with atomtype information.

    unet_params : class
        class object contains the following parameters

        vs : float
            voxel size of the grid you want to create.

        pad : int
            padding around protein for the grid.

        cross : numpy.array
            array with the coordinates shape to create backbone mask.

    Returns
    -------
    data : numpy.array
        array of shape (M,3,64,64,64) with information ready to enter the U-net model.

    submaps_length : int
        the value of original size of submaps L, where L>=M.

    submaps_indexes : numpy.array
        numpy array with the indexes of M and where to be placed in a L length array (L,3,64,64,64).

    origin : numpy.array
        numpy array with length 3 contains the origin of the protein coordinates.
    '''
    #Create the 3D grid of your protein
    grid, origin, indexes = create_3Dgrid(atom_hetatm_coords, unet_params.vs, unet_params.pad)
    #Initialize model input and target arrays
    inp = np.zeros((3,)+grid.shape)
    for i in range(3):
        ind_to_fill = tuple(indexes[np.where(atom_hetatm_atomtype==i)].T)
        #Fill input array
        inp[i][ind_to_fill] = 1
    inp_s = []
    for item in inp:
        inp_s.append(create_submaps(item))
    data = np.swapaxes(np.array(inp_s),0,1)
    #Original submaps shape
    submaps_length = data.shape[0]
    #Compute where you have input in your submaps
    submaps_sum = np.sum(data[:,:,7:57,7:57,7:57],axis=tuple(range(1,5)))
    #Submaps indexes to input
    submaps_indexes = np.where(submaps_sum>0)
    #Keep only filled data
    data = data[submaps_indexes]
    return data, submaps_length, submaps_indexes, grid.shape, origin

def save_h5_data(f, group_name, coo, num=2):
    '''
    Saves coo matrix data in h5 file.

    Parameters
    ----------
    f : h5 file
        opened h5 file.

    group_name : str
        str name of submaps id to save.

    coo : tuple
        submap information in coo matrix format.

    num : int
        number of info arrays to save, default 2.

    Returns
    -------
    '''
    #Check if submap already exists.
    if group_name in f:
        try:
            t = f.get(group_name)
            tuple(t.get('0'))
            return
        except:
            pass
    group = f.create_group(group_name)
    for i in range(num):
        group.create_dataset(str(i), data=coo[i])
    return

#Submaps to h5 file
def save_training_submaps(data, h5_dir, pdbid, num=2):
    '''
    Save training submaps in h5 file.

    Parameters
    ----------
    data : numpy.array
        array with submaps data, array of shape (N,4,64,64,64) where N the number of submaps.

    h5_dir : str
        str name of directory to save h5 file.

    pdbid : str
        str name of pdb id.

    num : int
        number of info arrays to save, default 2.

    Returns
    -------
    barcode : list
        list with saved submaps.
    '''
    #h5 ID
    h5_id = np.random.randint(10,100)
    #Initialize barcode list
    barcode = []
    #For each submap in data
    for i, submap in enumerate(data):
        #Transform it to coo matrix
        coo = dense_ones_to_coo(submap)
        #If core of the input submaps has data, save it
        if submap[:-1,7:57,7:57,7:57].sum() > 0:
            #Load h5 file
            with h5py.File(f'{h5_dir}/{h5_id}.h5', 'a') as f:
                save_h5_data(f, f'{pdbid}_{i}', coo, num)
            #Add info in list
            barcode.append(f'{h5_id}_{pdbid}_{i}')
    return barcode

def load_training_submaps(data_barcode, h5_dir):
    '''
    Function to load training submaps

    Parameters
    ----------
    data_barcode : str
        info in str format that is the unique bardcode for a give submap. In the form of file_pdbID_submapID.

    h5_dir : str
        str name of directory to save h5 file.

    Returns
    -------
    coo : tuple
        Info of submap given in coo format.
    '''
    #Get h5 IF
    h5_id = data_barcode.split('_')[0]
    #Open the correct file
    with h5py.File(f'{h5_dir}/{h5_id}.h5', 'r') as f:
        #Get your submap
        t = f.get(data_barcode[3:])
        #Get coo matrix
        coo = (tuple(t.get('0')),np.array(t.get('1')))
    return coo

#Submaps to h5 file
def save_validation_submaps(data, h5_dir, pdbid, water_coords, origin, num=6):
    '''
    Function to save validation submaps.

    Parameters
    ----------
    data : numpy.array
        array with submaps data, array of shape (N,3,64,64,64) where N the number of submaps.

    h5_dir : str
        str name of directory to save h5 file.

    pdbid : str
        str name of pdb id.

    water_coords : numpy.array
        array of shape (M,3) with coordinates of water molecules.

    origin : numpy.array
        array of shape (3) with the origin of the pdb coordinates

    num : int
        number of info arrays to save, default 2.

    Returns
    -------
    '''
    #Initialize barcode list
    submaps_length, saved_submaps = data.shape, []
    #For each submap in data
    for i, submap in enumerate(data):
        if submap[:3,7:57,7:57,7:57].sum() > 0:
            saved_submaps.append(i)
    data = data[saved_submaps]
    #Transform it to coo matrix
    data = dense_ones_to_coo(data)
    data = data+(saved_submaps, submaps_length, water_coords, origin)
    #Load h5 file
    with h5py.File(f'{h5_dir}/{pdbid}.h5', 'a') as f:
        save_h5_data(f, f'{pdbid}', data, num)
    return

def load_validation_submaps(pdbid, h5_dir):
    '''
    Function to load validation submaps

    Parameters
    ----------
    pdbid : str
        pdb ID.

    h5_dir : str
        str name of directory to save h5 file.

    Returns
    -------
    coo : tuple
        Info of submap given in coo format.
    '''
    #Open the correct file
    with h5py.File(f'{h5_dir}/{pdbid}.h5', 'r') as f:
        #Get your submap
        t = f.get(pdbid)
        #Get coo matrix
        coo, saved_submaps, submaps_length, water_coords, origin = (tuple(t.get('0')),np.array(t.get('1'))), np.array(t.get('2')), np.array(t.get('3')), np.array(t.get('4')), np.array(t.get('5'))
    return coo, saved_submaps, submaps_length, water_coords, origin

def load_training_coordinates(pdbid, h5_dir):
    '''
    Function to load training submaps

    Parameters
    ----------
    pdbid : str
        pdb ID.

    h5_dir : str
        str name of directory to save h5 file.

    Returns
    -------
    water_coords : numpy array
        Water coordinates, shape (M,3).

    atom_hetatm_coords : numpy array
        Atom and hetatm coordinates, shape (N,3).

    atom_hetatm_atomtype : numpy array
        Atom and hetatm atom types, shape (N).
    '''
    #Open the correct file
    with h5py.File(f'{h5_dir}/data.h5', 'r') as f:
        #Get your submap
        t = f.get(pdbid)
        #Get coordinate info
        water_coords, atom_hetatm_coords, atom_hetatm_atomtype = np.array(t.get('0')), np.array(t.get('1')), np.array(t.get('2'))
    return water_coords, atom_hetatm_coords, atom_hetatm_atomtype

def load_validation_coordinates(pdbid, h5_dir):
    '''
    Function to load training submaps

    Parameters
    ----------
    pdbid : str
        pdb ID.

    h5_dir : str
        str name of directory to save h5 file.

    Returns
    -------
    water_coords : numpy array
        Water coordinates, shape (M,3).

    atom_hetatm_coords : numpy array
        Atom and hetatm coordinates, shape (N,3).

    atom_hetatm_atomtype : numpy array
        Atom and hetatm atom types, shape (N).
    '''
    #Open the correct file
    with h5py.File(f'{h5_dir}/{pdbid}.h5', 'r') as f:
        #Get your submap
        t = f.get(pdbid)
        #Get coordinate info
        water_coords, atom_hetatm_coords, atom_hetatm_atomtype = np.array(t.get('0')), np.array(t.get('1')), np.array(t.get('2'))
    return water_coords, atom_hetatm_coords, atom_hetatm_atomtype

def transform_coordinates(water_coords, atom_hetatm_coords, flip = False):
    '''
    Transforms coordinates (translation and rotation) with or without flip

    Parameters
    ----------
    water_coords : numpy.array
        water coordinates array of shape (M,3).

    atom_hetatm_coords : numpy.array
        numpy array of shape (N,3) with coordinate information.

    flip : bool
        True or False if your want to flip or not, accordingly; default False.

    Returns
    -------
    water_coords : numpy.array
        transformed water coordinates array of shape (M,3).

    atom_hetatm_coords : numpy.array
        transformed numpy array of shape (N,3) with coordinate information.
    '''
    #Prepare atom coordinates
    atom_hetatm_coords = np.concatenate((atom_hetatm_coords,np.ones((len(atom_hetatm_coords),1))), axis=1).T
    #Prepare water coordinates
    water_coords = np.concatenate((water_coords,np.ones((len(water_coords),1))), axis=1).T
    #Create transformation matrix
    transformation_matrix = random_transformation_matrix()
    #Transform atom and water coordinates
    atom_hetatm_coords = transformation_matrix@atom_hetatm_coords
    water_coords = transformation_matrix@water_coords
    if flip:
        #Then create flip_matrix
        flip_matrix = random_flip_matrix()
        #Flip accordingly atom and water coordinates
        atom_hetatm_coords = flip_matrix@atom_hetatm_coords[:3,:]
        water_coords = flip_matrix@water_coords[:3,:]
        return water_coords.T, atom_hetatm_coords.T
    else:
        return water_coords[:3,:].T, atom_hetatm_coords[:3,:].T

def coordinates_to_indexes(coordinates, origin, vs=0.8):
    '''
    Map your coordinates to predifined 3D matrix indexes, given a certain cubic voxel size.

    Parameters
    ----------
    coordinates : numpy.array
        coordinates in the continues 3D space, (N,3) shape array.

    origin : list, tuple or numpy.array
        origin information comes from the predifined 3D matrix, in order to place coordinates into the correct voxels. (3) shape array, tuple or list.

    vs : float
        voxel size of your predifined 3D grid, default 0.8.

    Returns
    -------
    mapind : numpy.array
        numpy array of shape (N,3) with the indexes of the coordinates that correspond to the given origin and voxel size 3D array.
    '''
    #Copy your coordinates.
    mapind = np.copy(coordinates)
    #Fix coordinates, remember the origin defines the center of the first voxel, you want the corner, thus add -vs/2.
    mapind[:,0], mapind[:,1], mapind[:,2] = mapind[:,0]-(origin[0]-(vs/2)), mapind[:,1]-(origin[1]-(vs/2)), mapind[:,2]-(origin[2]-(vs/2))
    #Scale coordinates according to voxel size.
    mapind /= vs
    #Map indexes are by definition integers.
    return mapind.astype(int)

def indexes_to_coordinates(indexes, origin, vs=0.8):
    '''
    Return your coordinates positions from 3D map indexes, given a certain cubic voxel size.

    Parameters
    ----------
    indexes : numpy.array
        indexes of your map which contain a atom position, (N,3) shape array.

    origin : list, tuple or numpy.array
        origin of your coordinates, given when 3D grid was created to place your coordinates, (3) shape array, tuple or list.

    vs : float
        voxel size of your 3D map, default 0.8.

    Returns
    -------
    coordinates : numpy.array
        numpy array of shape (N,3) with coordinates of the given indexes according to the given origin and voxel size.
    '''
    #Return your indexes multiplied by voxel size and add the origin.
    return vs*indexes+origin

def prepare_train_validation_unet_data(water_coords, atom_hetatm_coords, atom_hetatm_atomtype, vs, pad, validation=False):
    '''
    Creates data to input in the U-net model for prediction.

    Parameters
    ----------
    water_coords : numpy.array
        water coordinates array of shape (M,3).

    atom_hetatm_coords : numpy.array
        numpy array of shape (N,3) with coordinate information.

    atom_hetatm_atomtype : numpy.array
        numpy array of shape (N,1) with atomtype information.

    vs : float
        voxel size of the grid you want to create.

    pad : int
        padding around protein for the grid.

    validation : bool
        To return data for validation or training of 3D Unet, default False.

    Returns
    -------
    coo_submaps : array
        Submaps in coo matrix format, ready for training.

    OR

    data : array
        array of shape (M,3,64,64,64) with information ready to enter the U-net model.

    submaps_length : int
        the value of original size of submaps L, where L>=M.

    submaps_indexes : array
        numpy array with the indexes of M and where to be placed in a L length array (L,3,64,64,64).

    origin : array
        numpy array with length 3 contains the origin of the
    '''
    #Create the 3D grid of your protein
    grid, origin, indexes = create_3Dgrid(atom_hetatm_coords, vs, pad)
    #Compute water indexes for the above grid, only for waters close to 6 angstroms from protein.
    water_indexes = coordinates_to_indexes(water_coords, origin, vs)
    #Initialize model input and target arrays
    data = np.zeros((4,)+grid.shape)
    for i in range(3):
        ind_to_fill = tuple(indexes[np.where(atom_hetatm_atomtype==i)].T)
        #Fill input array
        data[i][ind_to_fill] = 1
    #Fill targer array (waters)
    data[-1][tuple(water_indexes.T)] = 1
    data_s = []
    for item in data:
        data_s.append(create_submaps(item))
    data = np.swapaxes(np.array(data_s),0,1)
    #Original submaps shape
    submaps_shape = data.shape
    #Compute where you have input in your submaps
    submaps_sum = np.sum(data[:,:-1,7:57,7:57,7:57],axis=tuple(range(1,5)))
    #Submaps indexes to input
    submaps_indexes = np.where(submaps_sum>0)
    #Keep only filled data
    data = data[submaps_indexes]
    if validation:
        #Return validation data
        return [data, submaps_indexes, submaps_shape, water_coords, origin, np.array(grid.shape)]
    else:
        #Return training data
        return [dense_ones_to_coo(submap, dtp = int) for submap in data]

#Convert a full/dense array to a sparse coo type matrix.
def dense_ones_to_coo(array, dtp = int):
    '''
    Transforms dense data arrays of bool type to sparse coo type matrix.

    Parameters
    ----------
    array : numpy.array
        numpy array (2D or 3D) that represents your sparse data

    dtp : type
        data type of your saved data, default int

    Returns
    -------
    tuple with information in a coo matrix format (indexes, shape of original array).
    '''
    return tuple((np.where(array!=0), np.shape(array)))
    #return tuple((np.where(array!=0), array[array!=0].astype(int), np.shape(array)))

#Convert back from a sparse coo matrix to the original full array.
def coo_to_dense_ones(coo, dtp='float64'):
    '''
    Transforms sparse coo type matrix back to dense data arrays of bool data type.

    Parameters
    ----------
    coo : tuple
        coo matrix information, indexes and array shape.

    dtp : type
        data type of your saved data, default float64.

    Returns
    -------
    numpy.array original dense matrix.
    '''
    array = np.zeros(coo[-1], dtype = dtp)
    array[coo[0]] = 1
    return array

def torch_coo_to_dense_ones(coo, dtp=torch.float64):
    '''
    Transforms sparse coo type matrix back to dense data for torch tensors of bool data type.

    Parameters
    ----------
    coo : tuple
        coo matrix information, indexes and array shape.

    dtp : type
        data type of your saved data, default torch.float64.

    Returns
    -------
    torch.Tensor original dense matrix.
    '''
    array = torch.zeros(tuple(coo[-1][0].tolist()), dtype = dtp)
    array[coo[0]] = 1
    return array

#Convert a full/dense array to a sparse coo type matrix.
def dense_to_coo(array, dtp = int):
    '''
    Transforms dense data arrays to sparse coo type matrix.

    Parameters
    ----------
    array : numpy.array
        original numpy array (2D or 3D) to transformed into coo type matrix.

    dtp : type
        data type of your saved data, default int

    Returns
    -------
    tuple with information in a coo matrix format (indexes, values, shape of original array).
    '''
    indexes = np.where(array!=0)
    return tuple((indexes, array[indexes].astype(int), np.shape(array)))

#Convert back from a sparse coo matrix to the original full array.
def coo_to_dense(coo, dtp='float64'):
    '''
    Transforms sparse coo type matrix back to dense data arrays.

    Parameters
    ----------
    coo : tuple
        coo matrix information, indexes, values and array shape.

    dtp : type
        data type of your saved data, default float64.

    Returns
    -------
    numpy.array original array (2D or 3D).
    '''
    array = np.zeros(coo[-1], dtype = dtp)
    array[coo[0]] = coo[1]
    return array

def remove_duplicates(predicted_coords, predicted_weights, dist, batch_size = 50):
    '''
    Function that removes duplicates simply, created for post processing of the 3D Unet, removes nearby water coordinates based on their scores.

    Parameters
    ----------
    predicted_coords : numpy.array
        numpy array of shape (N,3) of the predicted water coordinates.

    predicted_weights : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    dist : float
        distance value to look after for duplicates.

    batch_size : int
        batch size value to process for duplicates, default 50.

    Returns
    -------
    todelete : numpy.array
        indexes of predicted water coordinates to delete as duplicates.
    '''
    #Make a copy of your predicted_coords and predicted_weights
    predicted_coords_refined, predicted_weights_refined = np.copy(predicted_coords), np.copy(predicted_weights)
    coords_len = len(predicted_coords)
    #Find the batches according to batch size
    coords_batches = int(np.ceil(coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    batch_loop = np.linspace(0, coords_len, coords_batches, dtype=int)
    #todelete list will keep indexes you want to remove
    todelete = []
    #Start computing for each batch
    for b_i, batch in enumerate(batch_loop[:-1]):
        indexes = slice(batch, batch_loop[b_i+1])
        predicted_coords_batch = predicted_coords[indexes]
        #Calculate distances of predicted waters
        d_matrix = coordinates_to_dmatrix(predicted_coords_batch, predicted_coords)
        #Find duplicates within certain distance
        duplicates = np.unique(np.where(d_matrix<dist)[0])
        #Sort duplicates by the larger predicted weight index to the smallest predicted weight index.
        #duplicates = duplicates[np.argsort(-predicted_weights[duplicates])]
        #Keep the original indexing of duplicates
        duplicates_original_indexing = duplicates+batch
        #For each duplicate
        for i,j in zip(duplicates,duplicates_original_indexing):
            #Keep indexes of close predicted waters
            closeby = np.where(d_matrix[i]<dist)[0]
            #Remove j from closeby
            closeby_r = np.delete(closeby, np.argwhere(closeby==j)[0])
            #If your coordinate has the higher prediction in the region
            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                todelete += closeby_r.tolist()
            else:
                todelete.append(j)
    return np.unique(todelete)

def matching_points(points_indexes, d_min, d_argmin, points_matched):
    '''
    Calculate pairs between two coordinate sets.

    Parameters
    ----------
    points_indexes : numpy.array
        indexes of ground truth points to check, int array of shape (K), where K<=M.

    d_min : numpy.array
        minimum distance value for each ground truth point, acquired from np.min(d,1), float array of shape (M,1).

    d_argmin : numpy.array
        position of minimum distance prediction for each ground truth point, acquired from np.argmin(d,1), int array of shape (M,1).

    points_matched : list
        list that containes indexes of matches between the two sets, int list of shape (W,2).

    Returns
    -------
    points_matched : list
        list that containes indexes of matches between the two sets, int list of shape (W+N,2).

    duplicates : list
        indexes of found duplicates to be searched again for possible pairing.
    '''
    duplicates, settled = [], []
    #Iterate for each GT water.
    for i in points_indexes:
        #If i in duplicates continue
        if i in duplicates or i in settled:
            continue
        #Find if prediction match is also the arg minimum for other ground truth waters
        argmin_match = d_argmin[i]
        argmin_matches = np.where(d_argmin==argmin_match)[0]
        #If the length of argmin_matches is just 1, the matching is unique.
        if not len(argmin_matches)-1:
            points_matched.append([i, argmin_match])
            continue
        #Else find the closest GT water to the predicted water, and assign it.
        argmin_of_argmin_matches = argmin_matches[np.argmin(d_min[argmin_matches])]
        points_matched.append([argmin_of_argmin_matches, d_argmin[argmin_of_argmin_matches]])
        #Then you have to search the possibility that the ground truth waters left might have other potential matches.
        duplicates += argmin_matches[argmin_matches!=argmin_of_argmin_matches].tolist()
        #Also keep history of settled, in order to be computed only once.
        settled.append(argmin_of_argmin_matches)
    return points_matched, duplicates

def match_waters(d, d_min, d_argmin, thresh):
    '''
    Optimization function to match optimally each pair between two water coordinate sets (ground truth and predicted).

    Parameters
    ----------
    d : numpy.array
        distance matrix between the two coordinate sets, float array of shape (M,N)

    d_min : numpy.array
        minimum distance value for each ground truth point, acquired from np.min(d,1), float array of shape (M,1).

    d_argmin : numpy.array
        position of minimum distance prediction for each ground truth point, acquired from np.argmin(d,1), int array of shape (M,1).

    thresh : float
        distance threshold to match predicted and ground truth waters.

    Returns
    -------
    water_matched : list
        list that containes indexes of matches between the two sets, int list of shape (P,2).
    '''
    #If min values are larger than your threshold, set the position tag to -1.
    d_argmin_copy = np.copy(d_argmin)
    d_argmin_copy[d_min>thresh] = -1
    #Keep water indexes that have a match smaller or equal to your threshold.
    water_indexes = np.where(d_min<=thresh)[0]
    #First round of water matches
    water_matched = []
    water_matched, duplicates = matching_points(water_indexes, d_min, d_argmin_copy, water_matched)
    #While you have possible matches loop!
    while len(duplicates):
        #Matched pairs to array
        water_matched_arr = np.array(water_matched)
        #Copy the original distance matrix and max values of already found pair participants
        d_reduced = np.copy(d)
        d_reduced[water_matched_arr[:,0],:], d_reduced[:,water_matched_arr[:,1]] = 100, 100
        #Recalculate d_min and d_argmin
        d_min_reduced = np.min(d_reduced, 1)
        d_argmin_reduced = np.argmin(d_reduced,1)
        #If no pairs with distances equal or lower than threshold return
        if not len(np.where(d_min_reduced<=thresh)[0]):
            return water_matched
        #For distances higher than your threshold remove candidates
        d_argmin_reduced[d_min_reduced>thresh] = -1
        #Keep water indexes that have left as candidates.
        water_indexes = np.where(d_min_reduced<=thresh)[0]
        #Compute water matches that have remain.
        water_matched, duplicates = matching_points(water_indexes, d_min_reduced, d_argmin_reduced, water_matched)
    return water_matched

def compute_multiprocess_validation_match_waters(thresholds, pred_coords, scores, cap, reduced_water_coords):
    '''
    Function that executes computation of recall, precision and F1 for different cap values on prediction scores/coordinates, for multiprocess allocation.

    Parameters
    ----------
    thresholds : list, tuple or numpy.array
        distance thresholds to match predicted and ground truth waters.

    pred_coords : numpy.array
        predicted water coordinates from your model, float array of shape (N,3).

    scores : numpy.array
        scores of predicted water coordinates from your model, float array of shape (N,1).

    cap : float.
        cap value to threshold scores and consequently predicted waters.

    reduced_water_coords : numpy.array
        ground truth water coordinates only within radius Angstrom distance from protein coordinates, float array of shape (M,3).

    Returns
    -------
    cap_match_results : numpy.array
        match water results for each threshold, array of shape (len(thresholds)*3).
    '''
    cap_match_results = np.zeros((len(thresholds)*3))
    #Threshold predicted coordinates according to score
    thresh_water_coords = pred_coords[np.where(scores>cap)]
    #If there are no predicted water coords return
    if not len(thresh_water_coords):
        return cap_match_results
    #Calculate Distance matrix
    d = coordinates_to_dmatrix(reduced_water_coords, thresh_water_coords)
    try:
        #try to calculate closest prediction value for each GT water
        d_min = np.min(d, 1)
    except ValueError:  #raise if `d` is empty return
        return cap_match_results
    #Calculate closest prediction position for each GT water
    d_argmin = np.argmin(d,1)
    for disind, thresh in enumerate(thresholds):
        #Match waters
        water_matches = match_waters(d, d_min, d_argmin, thresh)
        #Keep data to calculate metrics
        cap_match_results[3*disind:3+3*disind] = len(water_matches), max(1e-4, len(thresh_water_coords)), max(1e-4,len(reduced_water_coords))
    return cap_match_results

def allocate_multiprocess_validation_match_waters(water_coords, pred_coords, scores, cap_values, thresholds, threads):
    '''
    Function that allocates to multiprocess computation of recall, precision and F1 for different cap values on prediction scores/coordinates.

    Parameters
    ----------
    water_coords : numpy.array
        ground truth water coordinates only within radius Angstrom distance from protein coordinates, float array of shape (M,3).

    pred_coords : numpy.array
        predicted water coordinates from your model, float array of shape (N,3).

    scores : numpy.array
        scores of predicted water coordinates from your model, float array of shape (N,1).

    cap_values : numpy.array
        different cap values to search for capping water coordinates scores, float array of shape (M)

    thresholds : list, tuple or numpy.array
        distance thresholds to match predicted and ground truth waters.

    threads : int
        number of processes to start concurrently, depends mainly on memory capacity of your machine and CPU capabilities.

    Returns
    -------
    match_results : array
        array with matching results between prediction and ground truth waters for every given cap value.
    '''
    #Initiallize your pool
    pool = Pool(processes=threads)
    match_results = pool.starmap(compute_multiprocess_validation_match_waters, [(thresholds, pred_coords, scores, cap, water_coords) for cap in cap_values])
    pool.close()
    pool.join()
    return np.array(match_results)

def allocate_validation_match_waters(water_coords, pred_coords, scores, cap_values, thresholds):
    '''
    Function that allocates to multiprocess computation of recall, precision and F1 for different cap values on prediction scores/coordinates.

    Parameters
    ----------
    water_coords : numpy.array
        ground truth water coordinates only within radius Angstrom distance from protein coordinates, float array of shape (M,3).

    pred_coords : numpy.array
        predicted water coordinates from your model, float array of shape (N,3).

    scores : numpy.array
        scores of predicted water coordinates from your model, float array of shape (N,1).

    cap_values : numpy.array
        different cap values to search for capping water coordinates scores, float array of shape (M)

    thresholds : list, tuple or numpy.array
        distance thresholds to match predicted and ground truth waters.

    Returns
    -------
    match_results : array
        array with matching results between prediction and ground truth waters for every given cap value.
    '''
    return np.array([compute_multiprocess_validation_match_waters(thresholds, pred_coords, scores, cap, water_coords) for cap in cap_values])

def plot_metrics_per_cap_value(cap_values, mean_match_results, tind, t, epoch, save_path):
    '''
    Plots Recall, Precision, F1 per cap value for validation set.

    Parameters
    ----------
    cap_values : numpy.array
        cap values to threshold scores and consequently predicted waters, float array.

    mean_match_results : numpy.array
        mean results for each validation instance, float array of shape (len(cap_values),len(threshold)*3)

    tind : int
        index of threshold value.

    t : float
        threshold value.

    epoch : int
        number of epoch.

    Returns
    -------
    '''
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(cap_values, mean_match_results[:,0+3*tind], '-', label='Recall')
    plt.plot(cap_values, mean_match_results[:,1+3*tind], '-', label='Precision')
    plt.plot(cap_values, mean_match_results[:,2+3*tind], '-', label='F1')
    argmax_F1 = np.argmax(mean_match_results[:,2+3*2], 0)
    cap_value_argmax_F1 = np.round(cap_values[argmax_F1],2)
    argmax_F1_value = np.round(mean_match_results[:,2+3*tind][argmax_F1], 3)
    argmax_recall_value = np.round(mean_match_results[:,1+3*tind][argmax_F1], 3)
    argmax_precision_value = np.round(mean_match_results[:,0+3*tind][argmax_F1], 3)
    plt.plot(cap_value_argmax_F1, argmax_F1_value, 'x', color='C2', label=f'{argmax_F1_value}/{cap_value_argmax_F1}')
    plt.plot(cap_value_argmax_F1, argmax_precision_value, 'x', color='C0', label=f'{argmax_precision_value}')
    plt.plot(cap_value_argmax_F1, argmax_recall_value, 'x', color='C1', label=f'{argmax_recall_value}')
    plt.legend()
    plt.title(f'Metrics per cap value, {t} Å distance, Epoch {epoch}')
    plt.xlabel('Cap value', fontsize=14)
    plt.ylabel('Metric score', fontsize=14)
    plt.xticks(np.linspace(cap_values[0], cap_values[-1], int(np.ceil(len(cap_values)/2))), fontsize=7)
    plt.yticks(np.linspace(0,1,21), fontsize=7)
    plt.savefig(f'{save_path}Metrics_per_cap_value_epoch_{epoch}_{t}_Å.png')
    plt.close()

def save_checkpoint(state, epoch, filename='check.pth.tar'):
    '''
    Function to save checkpoint of a pytorch model.

    Parameters
    ----------
    state : torch.model
        state of the model after the end of an epoch.

    epoch : int
        number of epoch that corresponds to the models state.

    filename : str
        filename to save the checkpoint, default 'check.pth.tar'.

    Returns
    -------
    '''
    print(f'=> Saving checkpoint, epoch {epoch}.')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, epoch, print_load = True):
    '''
    Function to load checkpoint of a pytorch model.

    Parameters
    ----------
    checkpoint : torch.load
        loaded checkpoint of the model.

    model : torch.model
        pytorch model

    epoch : int
        number of epoch that corresponds to the models state.

    Returns
    -------
    model : torch.model
        pytorch model with loaded weights of the checkpoint.
    '''
    if print_load:
        print(f"=> Loading checkpoint, epoch {epoch}.")
    model.load_state_dict(checkpoint["state_dict"])

def save_metrics(thresholds, writer, recall, precision, F1, sel_cap_value, dset, epoch):
    '''
    Save metrics to tensorboard writer

    Parameters
    ----------
    thresholds : torch.load
        loaded checkpoint of the model.

    writer : tensorboard.writer
        tensorboard writer.

    recall : float
        recall results of the epoch.

    precision : float
        precision results of the epoch.

    F1 : float
        F1 results of the epoch.

    sel_cap_value : float
        selected cap value.

    dset : str
        the set that you are writing metrics for.

    epoch : int
        number of epoch that corresponds to the results.

    Returns
    -------
    '''
    for i, radius in enumerate(thresholds):
        writer.add_scalar(f'{dset}_recall_{radius}', recall[i], epoch)
        writer.add_scalar(f'{dset}_precision_{radius}', precision[i], epoch)
        writer.add_scalar(f'{dset}_F1_{radius}', F1[i], epoch)
    writer.add_scalar(f'Selected_cap_value', sel_cap_value, epoch)

def write_loss(writer, loss, epoch):
    '''
    Save loss to tensorboard writer

    Parameters
    ----------
    writer : tensorboard.writer
        tensorboard writer.

    loss : float
        loss of the epoch.

    epoch : int
        number of epoch that corresponds to the results.

    Returns
    -------
    '''
    writer.add_scalar(f'Train_Loss', loss, epoch)

## Until this point every function needed for unet has been written ##

               ## Functions and data for MLP follow ##

def find_bondtype(emb):
    '''
    Find bondtype.

    Parameters
    ----------
    emb: list
        contains info of embedding pair and bond, length 3.

    Returns
    -------
    returns bond type
    '''
    bond = ''
    #Check if bond is double
    if emb[2]:
        bond = 'D'
    #return bond type
    return f'{bond}{emb[0][0]}{emb[1][0]}'

def cos_sin_angle(q_atom, p_water, bonded_atom):
    '''
    Calculate cosine and sine angles of q_atom point in the triangle p_water---q_atom---bonded_atom.

    cos(θ) = (a dot b) / |a|*|b|

    sin(θ) = |(a cross b)| / |a|*|b|

    Parameters
    ----------
    q_atom: numpy.array
        coordinates of pair atom of the protein

    p_water: numpy.array
        coordinates of pair atom of the water molecule

    bonded_atom: numpy.array
        coordinates atom bonded to q_atom

    Returns
    -------
    returns cosine and sine angles.
    '''
    #Find the vector starting from q atom to p water
    wq = p_water - q_atom
    #Find the vector starting from q atom to its bonded atom
    bq = bonded_atom - q_atom
    #Compute denominator
    denom = (np.linalg.norm(wq) * np.linalg.norm(bq))
    #Return cosine and sine.
    return [np.dot(wq, bq)/denom, np.linalg.norm(np.cross(wq, bq))/denom]

def compute_radian(cos, sin):
    '''
    Compute radians given cosine and sine values

    Parameters
    ----------
    cos : float
        float value of cosine in the space of [-1,1]

    sin : float
        float value of sine in the space of [-1,1]

    Returns
    -------
    radian : float
        float value in radians in the space of [0,2p]
    '''
    #You are at the first quarter
    if cos>=0 and sin>=0:
        radian = np.arccos(cos)
    #You are at the second quarter
    elif cos<0 and sin>=0:
        radian = np.arccos(cos)
    #You are at the third quarter
    elif cos<0 and sin<0:
        radian = np.arcsin(sin)+3*np.pi/2
    #You are at the fourth quarter
    else:
        radian = np.arcsin(sin)+2*np.pi
    return radian

def compute_angle_term(pdb, pos):
    '''
    Compute the angle term for q_atom(pos) and p_water(water_pos) pair.

    Parameters
    ----------
    pdb : class
        object of PDB class

    pos: int
        index of atom position in the PDB class self arrays.

    Returns
    -------
    returns all the bonding info of the protein structure.
    '''
    info = []
    #If position is negative, your window just start from the start.
    if pos < 40:
        window = 0
    else:
        #else check only for the previous 40 atoms.
        window = pos-40
    #connectivity with atom in the same amino acid.
    #Find indexes of the same amino acid atoms.
    same = np.where(pdb.resnum[window:pos+40] == pdb.resnum[pos])[0]+window
    if len(same):
        try:
            #If there are any search the bonded atom to yours.
            for bonded in bonds_int[pdb.resname[pos]][pdb.atomname[pos]]:
                # If there are any in the protein
                if bonded in pdb.atomname[same]:
                    #Keep index of the bonded atom.
                    s_ind = same[np.where(pdb.atomname[same] == bonded)][0]
                    if pdb.chain[s_ind] == pdb.chain[pos]:
                        bond_type = 0 # pre assign bond type.
                        #Keep index of the bonded atom.
                        s_ind = same[np.where(pdb.atomname[same] == bonded)][0]
                        #Look if bond type is double.
                        if double_bonds_int.get(pdb.resname[pos]).get(pdb.atomname[s_ind]) == pdb.atomname[pos]:
                            bond_type = 1 #assign double bond type.
                        #Get bond type index
                        bond_type = bondtypesdict[find_bondtype([inverse_atomnamesdict[pdb.atomname[pos]],
                                                                inverse_atomnamesdict[pdb.atomname[s_ind]],
                                                                bond_type])]
                        #Keep info of bonded atomtype, bond type, and angle.
                        info += [bond_type, s_ind]
                else:
                    info += [0,0]
        except:
            info += [0,0]
    #previous amino acid connectivity, 26 is the index for 'N' atomtype.
    if not pdb.atomname[pos] - 26:
        # Check if there is the previous amino acid.
        prev_r = np.where(pdb.resnum[window:pos] == pdb.resnum[pos]-1)[0]+window
        # Check if in the previous amino acid there is 'C' atom type, 2 is the index for 'C'.
        prev_C = np.where(pdb.atomname[prev_r] == 2)[0]
        # If prec_C exists.
        if len(prev_C):
            #Keep the index of 'C'
            p_ind = prev_r[prev_C][0]
            if pdb.chain[p_ind] == pdb.chain[pos]:
                info += [1, p_ind]
        else:
            info += [0,0]
    #next aa
    # connectivity with next amino acid, 2 is the index for 'C'
    if not pdb.atomname[pos] - 2:
        # Check if there is next amino acid.
        next_r = np.where(pdb.resnum[pos:pos+40] == pdb.resnum[pos]+1)[0]+pos
        # Check if in the previous amino acid there is 'N' atom type, 26 is the index for 'N'.
        next_N = np.where(pdb.atomname[next_r] == 26)[0]
        # If next_N exists.
        if len(next_N):
            #Keep the index of the atom
            n_ind = next_r[next_N][0]
            if pdb.chain[n_ind] == pdb.chain[pos]:
                info += [1, n_ind]
        else:
            info += [0,0]
    return info

def precompute_embedding(pdb):
    '''
    Precomputes embedding info, concerning bonds.

    Parameters
    ----------
    pdb : class
        object of PDB class

    Returns
    -------
    returns all the bonding info of the protein structure.
    '''
    info = np.zeros((len(pdb.coords),6))
    #PD = coordinates_to_dmatrix(pdb.coords, pdb.coords)
    for i in range(len(pdb.coords)):
        ti = compute_angle_term(pdb, i)
        info[i,:len(ti)] = ti
    return info

def multiprocess_embedding_computation(pdb, water_coords, height, w_ind, info, distance=4, need_labels=False):
    '''
    Creates embedding during inference with multiprocessing, for the isolevel coordinates.

    Parameters
    ----------
    pdb : class
        object of PDB class

    water_coords : numpy.array
        the water coordinates for which you will create the embedding, float array of shape (M,3).

    height : int
        approximate (row) dimension of embedding array.

    w_ind : int
        water index, defines each water coordinate.

    info : numpy.array
        given info for computed angle terms for the protein structure.

    distance : int
        distance to look for protein atom pair and neighbors, float default 4(Angstrom).

    need_labels : bool
        to return or not labels for the embedding, default False.

    Returns
    -------
    emb : array
        array of the embedding information.

    IF need_labels = True

    distance_from_water : array
        an array with distances from ground truth waters for the candidate waters.
    '''
    e_ind = 0
    #Calculate distance matrices
    D = coordinates_to_dmatrix(water_coords, pdb.coords)
    emb = np.zeros((height,6))
    # Iterate through waters
    for i, line in enumerate(D):
        prot_coords = np.where(line<=distance)[0]
        if not len(prot_coords):
            #Continue to the next water
            w_ind += 1
            continue
        # Iterate through protein atoms within 4 Angstrom distance.
        for j in prot_coords:
            #Keep info of atom
            ti = info[j]
            #for each bonded atom
            b_info = []
            for a in (0,2,4):
                #If no distance from bonded atom, there is no other bonded atom and break
                if not ti[a]:
                    b_info += [0]
                    continue
                #Index of bonded atom
                ind = int(ti[a+1])
                #Add info into b_info list, ATOMNAME, BONDTYPE, ANGLE
                cos, sin = cos_sin_angle(pdb.coords[j], water_coords[i], pdb.coords[ind])
                b_info += [compute_radian(cos, sin)]
            #Concatenate Distance Term with Angle Term
            terms = [pdb.atomname[j], D[i,j]]+b_info
            #Add it to embedding.
            emb[e_ind,:-1] = terms
            #Add water identifier in last position.
            emb[e_ind,-1] = w_ind
            #Move to next embedding line.
            e_ind += 1
        #Next water index
        w_ind += 1
    if need_labels:
        #Distance matrix between predicted and ground truth waters and distance matrix between predicted waters and protein coordinates
        water_d = coordinates_to_dmatrix(water_coords, pdb.water_coords)
        #Since you needed labels, compute the minimum distances of candidate coordinates from ground truth waters
        return emb[:e_ind], np.min(water_d,1)
    else:
        return emb[:e_ind]

def allocate_multiprocess_embedding(pdb, candidate_coordinates, process_batch_size, threads, distance=4, info = [], need_labels = False):
    '''
    Function that given your water coordinates and machine capabilities (ram, threads),
    allocates to different processes parts of the coordinates to compute at the same time
    the embedding for your whole coordinate set.

    Parameters
    ----------
    pdb : class
        object of PDB class

    candidate_coordinates : numpy.array
        the water coordinates for which you will create the embedding, float array of shape (M,3).

    process_batch_size : int
        number of water coordinates to process each processor job, depends on memory and process power of CPU.

    threads : int
        number of processes to start concurrently, depends mainly on memory capacity of your machine and CPU capabilities.

    distance : float
        distance to look for protein atom pair and neighbors, float default 4 Angstrom.

    info : list
        if you have precomputed bonds information give it to skip recomputation.

    need_labels : bool
        to return or not labels for the embedding, default False.

    Returns
    -------
    emb : numpy.array
        array of the embedding information.

    IF need_labels = True

    distance_from_water : numpy.array
        an array with distances from ground truth waters for the candidate waters.
    '''
    #Compute your process intervals
    candidate_coordinates_len = len(candidate_coordinates)
    emb_loop = np.linspace(0, candidate_coordinates_len, int(np.ceil(candidate_coordinates_len/process_batch_size))+1, dtype=int)
    e_len = len(emb_loop)
    #Number of processes you will need.
    num_processes = e_len-2
    if not len(info):
        #Compute protein coords embedding
        info = precompute_embedding(pdb)
    #Approximate embedding height
    emb_height = emb_loop[1]*int(distance**2)
    #Initiallize your pool
    pool = Pool(processes=threads)
    emb_out = pool.starmap(multiprocess_embedding_computation, [(pdb , candidate_coordinates[emb_loop[i-1]:emb_loop[i]], emb_height,\
                                                                 emb_loop[i-1], np.copy(info), distance, need_labels) for i in range(1, len(emb_loop))])
    pool.close()
    pool.join()
    if need_labels:
        emb_out = np.array(emb_out, dtype=object)
        return np.concatenate([x for x in emb_out[:,0]], 0), np.concatenate([x for x in emb_out[:,1]], 0), info
    else:
        return np.concatenate([x for x in emb_out], 0).astype(float)

def embedding_multiprocess(pdb, i_coords, process_batch_size, threads=8, distance=4, need_labels = False):
    '''
    Main embedding function with the use of multiprocessing, that returns all embedding info of the isolevel of a protein.

    Parameters
    ----------
    pdb : class
        object of PDB class

    i_coords : numpy.array
        coordinates of the candidate water positions, float array of shape (N,3).

    process_batch_size : int
        number of water coordinates to process each processor job, depends on memory and process power of CPU.

    threads : int
        number of processes to start concurrently, depends mainly on memory capacity of your machine and CPU capabilities, default 8.

    distance : float
        distance to look for protein atom pair and neighbors, float default 4 Angstrom.

    need_labels : bool
        to return or not labels for the embedding, default False.

    Returns
    -------
    pdb : class
        object of PDB class, cleaned from non canonical atoms, hydrogens and alternative atoms.

    emb : numpy.array
        array of the embedding information.

    IF need_labels = True

    distance_from_water : numpy.array
        an array with distances from ground truth waters for the candidate waters.
    '''
    if type(pdb) == str:
        pdb = PDB(pdb)
    #First clean any hydrogens
    pdb.remove_by_atomtype('H')
    #Remove atomnames not included in you dataset and alternative atoms.
    clean_pdb(pdb)
    #Transform to integers pdb.atomname and pdb.resname
    pdb.atomname = np.array([atomnamesdict[x] for x in pdb.atomname])
    pdb.resname = np.array([resnamedict[x] for x in pdb.resname])
    #Then just compute embedding via multiprocess and return
    return pdb, allocate_multiprocess_embedding(pdb, i_coords, process_batch_size, threads, distance, need_labels = need_labels)

def reduce_embedding_length(over_emb, ids_len, i, emb_thresh, emb_len):
    '''
    Reduce size of embedding to default threshold by sorting distances from protein atoms,
    top emb_thresh closest protein atoms will be kept to be passed from model.

    Parameters
    ----------
    over_emb : torch.Tensor
        torch.Tensor of shape (ids_len*i, emb_len) includes the embedding of the over embedded waters.

    ids_len : int
        number of over embedded waters.

    i : int
        number of embedding > emb_thresh.

    emb_thresh : int
        threshold of input embedding.

    emb_len : int
        length of the embedding vector features.

    Returns
    -------
    torch.Tensor
        shape (ids_len*emb_thresh, emb_len)
    '''
    #Initialize torch tensor that will keep sliced over embedded waters
    cut_emb = np.zeros((ids_len*emb_thresh,emb_len))
    #For each unique water in over_emb
    for ind in range(1,1+ids_len):
        water_slice = slice((ind-1)*i,ind*i)
        #Keep the embedding of the water
        temp_emb = over_emb[water_slice]
        #Sort the embeddings by their distances water - protein_atom and keep the first [:emb_thresh] elements
        temp_emb_sorted = np.argsort(temp_emb[:,1])[:emb_thresh]
        #Inport the sliced embedding into their new vector
        cut_emb[(ind-1)*emb_thresh:ind*emb_thresh] = over_emb[water_slice][temp_emb_sorted]
    return cut_emb

def reform_dataset(emb, emb_thresh):
    '''
    Function that reforms the number of embedding per water coordinate to a steady number.

    Parameters
    ----------
    emb : numpy.array
        water embeddings of different length each.

    emb_thresh : int
        threshold of embedding number.

    Returns
    -------
    c : numpy.array
        reformed embeddings.

    batch_unique_waters : numpy.array
        array with indexes of waters that had embedding, note some waters don't have any embedding.
    '''
    embedding_length = emb.shape[1]
    #Waters of batch
    batch_waters = emb[:,-1].astype(int)
    #Unique IDs of batch
    batch_unique_waters = np.unique(batch_waters)
    #Number of water pairs
    water_ids_bincount = np.bincount(batch_waters)
    #Compute cummulative sum of water_ids_bincount
    cum_water_ids_bincount = np.cumsum(water_ids_bincount,0)
    #Number of pairs in the current batch
    unique_water_bins = np.unique(water_ids_bincount).astype(int)
    #Prepare output tensor of Statistical Reduction
    c = np.zeros((batch_unique_waters[-1]-batch_unique_waters[0]+1, emb_thresh, embedding_length-1))
    #For each number of pair lengths
    if unique_water_bins[0]==0:
        unique_water_bins = unique_water_bins[1:]
    for i in unique_water_bins:
        #Range of water ID in a.
        trange = np.arange(-i, 0)
        #Find IDs have i pairs
        pairs_water_ids = np.where(water_ids_bincount==i)
        #Pair water IDs for c tensor, just keep interval batch_unique_waters[0] - batch_unique_waters[-1], else ram explodes.
        pairs_water_ids_interval = np.add(pairs_water_ids[0],-batch_unique_waters[0])
        ids_len = len(pairs_water_ids[0])
        #Compute indexes for i pairs IDs, for a or batch_waters tensor
        i_pairs_indexes = np.reshape(cum_water_ids_bincount[pairs_water_ids][...,None]+trange, (ids_len*i,))
        #Keep current embedding
        current_embedding = emb[i_pairs_indexes]
        #If embedding lenght is bigger than embedding threshold
        if i > emb_thresh:
            #Reduce embedding to embedding threshold
            current_embedding = reduce_embedding_length(current_embedding, ids_len, i, emb_thresh, embedding_length)
            i = emb_thresh
         #Compute embedding transformation
        emb_tr = np.reshape((current_embedding[:,:-1]), ((ids_len, i, embedding_length-1)))
        #Compute the correct zero padding for the second from last dimension
        p4d = (0,0,0,emb_thresh-i)
        #Fix your tensor
        emb_tr = np.array(F.pad(torch.tensor(emb_tr),p4d))
        #Send it to input tensor
        c[pairs_water_ids_interval] = emb_tr
    #Keep the final results, remember to subtract batch_unique_waters[0] from indexes.
    c = c[np.add(np.where(water_ids_bincount>0)[0],-batch_unique_waters[0])]
    return c, batch_unique_waters

def save_embedding_dataset(f, pdb_id, info, items_to_save):
    '''
    Function to save in H5 file

    Parameters
    ----------
    f : h5 file
        opened h5 file.

    pdb_id : str
        id of pdb you want to save under.

    info : numpy.array
        all the info you want to save.

    items_to_save : set
        items ids to save in h5 file, can be also a range.
    '''
    #Check if submap already exists, then you want to update.
    if pdb_id in f:
        #Require the already existed group
        group = f.require_group(pdb_id)
        #And create the needed dataset
        try:
            for i, j in enumerate(items_to_save):
                group.create_dataset(str(j), data=info[i])
        except:
            return
    else:
        group = f.create_group(pdb_id)
        for i in items_to_save:
            group.create_dataset(str(i), data=info[i])
    return

def load_embedding_dataset(pdb_id, path, items_to_load):
    '''
    Load an H5 file

    Parameters
    ----------
    pdb_id : str
        id of pdb you want to load.

    path : str
        path to your h5 file.

    items_to_load : set
        items ids to load from h5 file, can be also a range.

    Returns
    -------
        list of numpy.arrays with your saved data.
    '''
    with h5py.File(path, 'r') as f:
        t = f.get(pdb_id)
        return [np.array(t.get(str(x))) for x in items_to_load]

def shuffle_along_axis(a, axis, batch_size=2000):
    '''
    Shuffles array along a selected axis, created to shuffle water embedding order.

    Parameters
    ----------
    a : numpy.array
        array with embeddings, for example shape can be (N,10,5).

    axis : int
        selected axis to shuffle, for the embedding the axis should be 1.

    batch_size : int
        shuffle in batches to make the function faster, default 2000.
    '''
    a_len = a.shape[0]
    #Find number of batches
    batches_num = int(np.ceil(a_len/batch_size))+1
    #Compute batch loop
    batch_loop = np.linspace(0, a_len, batches_num, dtype=int)
    #Start computing for each batch
    for b_i, batch in enumerate(batch_loop[:-1]):
        b = slice(batch, batch_loop[b_i+1])
        idx = np.random.rand(*a[b].shape).argsort(axis=axis)
        for i in range(1,a[b].shape[axis+1]):
            idx[:,:,i] = idx[:,:,0]
        a[b] = np.take_along_axis(a[b],idx,axis=axis)
    return a

def remove_and_refine_duplicates(predicted_coords, predicted_weights, dist, batch_size = 50):
    '''
    Function that removes duplicates and refines predicted coordinates, created for post processing of the mlp model, removes nearby water coordinates based on their scores, and refines coordinates by applying weighted average.

    Parameters
    ----------
    predicted_coords : numpy.array
        numpy array of shape (N,3) of the predicted water coordinates.

    predicted_weights : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    dist : float
        distance value to look after for duplicates.

    batch_size : int
        batch size value to process for duplicates, default 50.

    Returns
    -------
    predicted_coords_refined : numpy.array
        refined predicted coordinates of the water molecules, shape (N,3).

    predicted_weights_refined : numpy.array
        numpy array of shape (N,1) with the corresponding prediction scores for the predicted_coords.

    todelete : numpy.array
        indexes of predicted water coordinates to delete as duplicates.
    '''
    #Make a copy of your predicted_coords and predicted_weights
    predicted_coords_refined, predicted_weights_refined = np.copy(predicted_coords), np.copy(predicted_weights)
    coords_len = len(predicted_coords)
    #Find the batches according to batch size
    coords_batches = int(np.ceil(coords_len/batch_size))+1
    #Compute batch loops for a and b coordinates
    batch_loop = np.linspace(0, coords_len, coords_batches, dtype=int)
    #todelete list will keep indexes you want to remove
    todelete = []
    #Start computing for each batch
    for b_i, batch in enumerate(batch_loop[:-1]):
        indexes = slice(batch, batch_loop[b_i+1])
        predicted_coords_batch = predicted_coords[indexes]
        #Calculate distances of predicted waters
        d_matrix = coordinates_to_dmatrix(predicted_coords_batch, predicted_coords)
        #Find duplicates within certain distance
        duplicates = np.unique(np.where(d_matrix<dist)[0])
        #Sort duplicates by the larger predicted weight index to the smallest predicted weight index.
        #duplicates = duplicates[np.argsort(-predicted_weights[duplicates])]
        #Keep the original indexing of duplicates
        duplicates_original_indexing = duplicates+batch
        #For each duplicate
        for i,j in zip(duplicates,duplicates_original_indexing):
            #Keep indexes of close predicted waters
            closeby = np.where(d_matrix[i]<dist)[0]
            #Remove j from closeby
            closeby_r = np.delete(closeby, np.argwhere(closeby==j)[0])
            #If your coordinate has the higher prediction in the region
            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                #Keep to delete the coordinates around it
                todelete += closeby_r.tolist()
                #Compute weighted average for coordinates within 1.4 Angstrom
                #closeby = closeby[np.where(d_matrix[i][closeby]<1.4)[0]]
                #Keep indexes of close predicted waters
                region_indexes = closeby.tolist()
                #Compute the weights of the region
                region_weights = predicted_weights[region_indexes]
                #Sum the weights of the regions
                weight_sum = np.sum(region_weights)
                #Compute the weighted coordinates
                weighted_coords = predicted_coords[region_indexes]*np.expand_dims(region_weights/weight_sum, axis=1)
                predicted_coords_refined[j] = np.sum(weighted_coords, axis=0)
                #predicted_weights_refined[j] = weight_sum/len(region_weights)
            else:
                todelete.append(j)
    return predicted_coords_refined, predicted_weights_refined, np.unique(todelete)

## Prediction Utilities ##

def load_txt_pdb_ids(text_path):
    '''
    Open txt file with PDB IDs, delimeter should be comma or next line.

    Parameters
    ----------
    text_path : str
        path to the text file.

    Returns
    -------
    returns pdb ids in python list.
    '''
    #Open file
    with open(text_path, 'r') as f:
        #Read the lines
        info = f.readlines()
    #If length of the list is one, you have only one line, thus delimeter should be comma.
    if len(info)==1:
        #Split the str by comma
        split_by_comma = info[0].split(',')
        #If you have multiple entries remove \n from the last entry.
        if len(split_by_comma)>1:
            split_by_comma[-1] = split_by_comma[-1][:4]
            return split_by_comma
        else:
        #Else return the only entry and remove \n.
            return [split_by_comma[0][:4],]
    else:
        #If the delimeter is \n
        split_by_next_line = []
        #For each line
        for line in info:
            #Append without the delimeter
            split_by_next_line.append(line[:4])
        return split_by_next_line

def rmsd(a,b):
    '''
    Compute root mean squared distance (RMSD).

    Parameters
    ----------
    a : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    b : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    Returns
    -------
    RMSD metric.
    '''
    return np.linalg.norm(a-b)/(len(a)**0.5)

def align_and_compute_rmsd(a,b):
    '''
    For given coordinates with different axis position compute translation and rotation and then return root mean squared distance (RMSD).

    Parameters
    ----------
    a : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    b : numpy.array
        numpy array of shape (N,3) that contains coordinates information.

    Returns
    -------
    RMSD metric.
    '''
    if len(a) == 0 or len(b) == 0:
        return 0
    a_translation = np.sum(a,axis=0)/len(a)
    a = np.copy(a-a_translation)
    b_translation = np.sum(b,axis=0)/len(b)
    b = np.copy(b-b_translation)
    aT = a.T
    C = np.dot(aT, b)
    U, S, VT = np.linalg.svd(C)
    if np.linalg.det(C) < 0:
        VT[2,:] *= -1.0
    Q = np.dot(U, VT)
    a = np.dot(a,Q)
    return rmsd(a,b)

def write_pdb_ca(coords, filename):
    """Write PDB file format using pdb object as input."""
    records = []
    for i in range(len(coords)):
        atomnum = '%7i' % i
        atomname = '%3s' % 'CA'
        atomalt = '%1s' % ' '
        resnum = '%4i' % i
        resname = '%3s' % 'ALA'
        chain = '%1s' % 'A'
        x = '%8.3f' % coords[i,0]
        y = '%8.3f' % coords[i,1]
        z = '%8.3f' % coords[i,2]
        o = '% 6.2f' % 1.
        b = '%4.2f' % 1.
        atomtype = '%2s' % 'C'
        charge = '%2s' % '  '
        records.append([f'ATOM{atomnum:<8}{atomname}  {resname} {chain}{resnum:<8}{x}{y}{z}{o}{b}          {atomtype}{charge}'])
    np.savetxt(filename, records, fmt='%80s'.encode('ascii'))

def write_pdb_waters(water_coordinates, b_scores, filename):
    """Write PDB file format using water coordinates as input."""
    records = []
    for i, (water, b_factor) in enumerate(zip(water_coordinates, b_scores)):
        atomnum = '%5i' % i
        atomname = '%1s' % 'O'
        atomalt = '%1s' % ' '
        resnum = '%4i' % i
        resname = '%3s' % 'HOH'
        chain = '%1s' % 'A'
        x = '%8.3f' % water[0]
        y = '%8.3f' % water[1]
        z = '%8.3f' % water[2]
        o = '%6.2f' % 1.
        b = '%4.2f' % b_factor
        atomtype = '%2s' % 'O'
        charge = '%2s' % '  '
        records.append([f'HETATM{atomnum:<7}{atomname}   {resname} {chain}{resnum:<8}{x}{y}{z}{o}  {b}          {atomtype}{charge}'])
    np.savetxt(filename, records, fmt='%80s'.encode('ascii'))

#Atom names
atomnames = ['CA', 'C', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', #1-17
               'O', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'OH', #18-25
               'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ', 'NH1', 'NH2',#26-34
               'SG', 'SD']#35-36

atomtypes = ['C', 'N', 'O', 'S']

#Index dict of atomnames
atomnamesdict = {item:ind+1 for ind, item in enumerate(atomnames)}
#Index dict of atomtypes
atomtypesdict = {item:ind+1 for ind, item in enumerate(atomtypes)}
#Inverse atomnamesdict
inverse_atomnamesdict = {v: k for k, v in atomnamesdict.items()}
#Inverse atomtypesdict
inverse_atomtypesdict = {v: k for k, v in atomtypesdict.items()}

bondtypesdict = {'NC':1,'CN':1,'DNC':2,'DCN':2,'CC':3,'DCC':4,'OC':5,'CO':5,'DOC':6,'DCO':6, 'SC':7, 'CS':7}

#Bonds Information
BB_bonds = {'N':['CA'], 'CA':['N','CB','C'], 'C':['CA', 'O'], 'O':['C']}
CB_bonds = {'CB':['CA','CG']}
CB_CG_bonds = dict(CB_bonds, **{'CG':['CB','CD']})
bonds = {
    'ALA':dict(BB_bonds, **{'CB':['CA']}),
    'VAL':dict(BB_bonds, **{'CB':['CA','CG1','CG2'], 'CG1':['CB'],'CG2':['CB']}),
    'LEU':dict(BB_bonds, **CB_bonds, **{'CG':['CB','CD1','CD2'], 'CD1':['CG'], 'CD2':['CG']}),
    'ILE':dict(BB_bonds, **{'CB':['CA','CG1','CG2'], 'CG1':['CB','CD1'], 'CG2':['CB'], 'CD1':['CG1']}),
    'ASP':dict(BB_bonds, **CB_bonds, **{'CG':['CB','OD1','OD2'], 'OD1':['CG'], 'OD2':['CG']}),
    'GLU':dict(BB_bonds, **CB_CG_bonds, **{'CD':['CG','OE1','OE2'], 'OE1':['CD'], 'OE2':['CD']}),
    'ASN':dict(BB_bonds, **CB_bonds, **{'CG':['CB','OD1','ND2'], 'OD1':['CG'], 'ND2':['CG']}),
    'GLN':dict(BB_bonds, **CB_CG_bonds, **{'CD':['CG','OE1','NE2'], 'OE1':['CD'], 'NE2':['CD']}),
    'PRO':dict({'N':['CA','CD'], 'CA':['N','CB','C'], 'C':['CA', 'O'], 'O':['C'], 'CD':['CG', 'N']}, **CB_CG_bonds),
    'PHE':dict(BB_bonds, **CB_bonds, **{'CG':['CB','CD1','CD2'], 'CD1':['CG','CE1'], 'CD2':['CG','CE2'],
                                        'CE1':['CD1','CZ'], 'CE2':['CD2','CZ'], 'CZ':['CE1','CE2']}),
    'TRP':dict(BB_bonds, **CB_bonds, **{'CG':['CB','CD1','CD2'], 'CD1':['CG','NE1'], 'CD2':['CG','CE2','CE3'], 'NE1':['CD1','CE2'],
                                        'CE2':['CD2','NE1','CZ2'],'CE3':['CD2','CZ3'],'CZ2':['CE2','CH2'],'CZ3':['CE3','CH2'],'CH2':['CZ2','CZ3']}),
    'LYS':dict(BB_bonds, **CB_CG_bonds, **{'CD':['CG','CE'], 'CE':['CD','NZ'], 'NZ':['CE']}),
    'CYS':dict(BB_bonds, **{'CB':['CA','SG'],'SG':['CB']}),
    'MET':dict(BB_bonds, **CB_bonds, **{'CG':['CB','SD'],'SD':['CG','CE'],'CE':['SD']}),
    'TYR':dict(BB_bonds, **CB_bonds, **{'CG':['CB','CD1','CD2'],'CD1':['CG','CE1'],'CD2':['CG','CE2'],
                                        'CE1':['CD1','CZ'],'CE2':['CD2','CZ'],'CZ':['CE1','CE2','OH'],'OH':['CZ']}),
    'ARG':dict(BB_bonds, **CB_CG_bonds, **{'CD':['CG','NE'], 'NE':['CD','CZ'], 'CZ':['NE','NH1','NH2'], 'NH1':['CZ'], 'NH2':['CZ']}),
    'HIS':dict(BB_bonds, **CB_bonds, **{'CG':['CB','ND1','CD2'], 'ND1':['CG','CE1'], 'CD2':['CG','NE2'], 'CE1':['ND1','NE2'], 'NE2':['CE1','CD2']}),
    'SER':dict(BB_bonds, **{'CB':['CA','OG'],'OG':['CB']}),
    'THR':dict(BB_bonds, **{'CB':['CA','OG1','CG2'], 'OG1':['CB'], 'CG2':['CB']}),
    'GLY':{'N':['CA'], 'CA':['N','C'],'C':['CA', 'O'], 'O':['C']}
}

resnames = list(bonds.keys())

dihedral_BB = {'N':['C'], 'CA':['O'], 'C':['N'], 'O':['CA']}
dihedral_CB = {'CB':['N']}
dihedral_CB_CG = dict(dihedral_CB, **{'CG':['CA']})
dihedral_bonds = {
    'ALA':dict(dihedral_BB, **dihedral_CB),
    'VAL':dict(dihedral_BB, **dihedral_CB, **{'CG1':['CA'],'CG2':['CA']}),
    'LEU':dict(dihedral_BB, **dihedral_CB_CG, **{'CD1':['CB'], 'CD2':['CB']}),
    'ILE':dict(dihedral_BB, **dihedral_CB, **{'CG1':['CA'], 'CG2':['CA'], 'CD1':['CB']}),
    'ASP':dict(dihedral_BB, **dihedral_CB_CG, **{'OD1':['CB'], 'OD2':['CB']}),
    'GLU':dict(dihedral_BB, **dihedral_CB_CG, **{'CD':['CB'], 'OE1':['CG'], 'OE2':['CG']}),
    'ASN':dict(dihedral_BB, **dihedral_CB_CG, **{'OD1':['CB'], 'ND2':['CB']}),
    'GLN':dict(dihedral_BB, **dihedral_CB_CG, **{'CD':['CB'], 'OE1':['CG'], 'NE2':['CG']}),
    'PRO':dict(dihedral_BB, **dihedral_CB_CG, **{'CD':['CB']}),
    'PHE':dict(dihedral_BB, **dihedral_CB_CG, **{'CD1':['CB'], 'CD2':['CB'], 'CE1':['CG'], 'CE2':['CG'], 'CZ':['CD1']}),
    'TRP':dict(dihedral_BB, **dihedral_CB_CG, **{'CD1':['CB'], 'CD2':['CB'], 'NE1':['CG'], 'CE2':['CG'],
                                                 'CE3':['CG'], 'CZ2':['CD2'],'CZ3':['CD2'],'CH2':['CE2']}),
    'LYS':dict(dihedral_BB, **dihedral_CB_CG, **{'CD':['CB'], 'CE':['CG'], 'NZ':['CD']}),
    'CYS':dict(dihedral_BB, **{'CB':['N'],'SG':['CA']}),
    'MET':dict(dihedral_BB, **dihedral_CB_CG, **{'SD':['CB'],'CE':['CG']}),
    'TYR':dict(dihedral_BB, **dihedral_CB_CG, **{'CD1':['CB'],'CD2':['CB'], 'CE1':['CG'],'CE2':['CG'],'CZ':['CD1'],'OH':['CE1']}),
    'ARG':dict(dihedral_BB, **dihedral_CB_CG, **{'CD':['CB'], 'NE':['CG'], 'CZ':['CD'], 'NH1':['NE'], 'NH2':['NE']}),
    'HIS':dict(dihedral_BB, **dihedral_CB_CG, **{'ND1':['CB'], 'CD2':['CB'], 'CE1':['CG'], 'NE2':['CG']}),
    'SER':dict(dihedral_BB, **{'CB':['N'],'OG':['CA']}),
    'THR':dict(dihedral_BB, **{'CB':['N'], 'OG1':['CA'], 'CG2':['CA']}),
    'GLY':dict(dihedral_BB)
}

resnamedict = {item:ind+1 for ind, item in enumerate(bonds.keys())}
#bonds dictionary encoded with integers
bonds_int = {}
for v1, k1 in bonds.items():
    bonds_int[resnamedict[v1]] = {}
    for v2, k2 in k1.items():
        bonds_int[resnamedict[v1]][atomnamesdict[v2]] = []
        for atoms in k2:
            bonds_int[resnamedict[v1]][atomnamesdict[v2]].append(atomnamesdict[atoms])

#bonds dictionary encoded with integers
dihedral_int = {}
for v1, k1 in dihedral_bonds.items():
    dihedral_int[resnamedict[v1]] = {}
    for v2, k2 in k1.items():
        dihedral_int[resnamedict[v1]][atomnamesdict[v2]] = []
        for atoms in k2:
            dihedral_int[resnamedict[v1]][atomnamesdict[v2]].append(atomnamesdict[atoms])

bb_double_bond = {'C':'O','O':'C'}
double_bonds = {
    'ALA':bb_double_bond,
    'VAL':bb_double_bond,
    'LEU':bb_double_bond,
    'ILE':bb_double_bond,
    'ASP':dict(bb_double_bond, **{'CG':'OD1','OD1':'CG'}),
    'GLU':dict(bb_double_bond, **{'CD':'OE1','OE1':'CD'}),
    'ASN':dict(bb_double_bond, **{'CG':'OD1','OD1':'CG'}),
    'GLN':dict(bb_double_bond, **{'CD':'OE1','OE1':'CD'}),
    'PRO':bb_double_bond,
    'PHE':dict(bb_double_bond, **{'CG':'CD1','CD1':'CG','CD2':'CE2','CE2':'CD2','CE1':'CZ','CZ':'CE1'}),
    'TRP':dict(bb_double_bond, **{'CG':'CD1','CD1':'CG','CD2':'CE2','CE2':'CD2','CE2':'CZ3','CZ3':'CE2','CZ2':'CH2','CH2':'CZ2'}),
    'LYS':bb_double_bond,
    'CYS':bb_double_bond,
    'MET':bb_double_bond,
    'TYR':dict(bb_double_bond, **{'CG':'CD1','CD1':'CG','CD2':'CE2','CE2':'CD2','CE1':'CZ','CZ':'CE1'}),
    'ARG':dict(bb_double_bond, **{'CZ':'NH1','NH1':'CZ'}),
    'HIS':dict(bb_double_bond, **{'CG':'CD2','CD2':'CG','ND1':'CE1','CE1':'ND1'}),
    'SER':bb_double_bond,
    'THR':bb_double_bond,
    'GLY':bb_double_bond,
}

#double bonds dictionary encoded with integers
double_bonds_int = {}
for v1, k1 in double_bonds.items():
    double_bonds_int[resnamedict[v1]] = {}
    for v2, k2 in k1.items():
        double_bonds_int[resnamedict[v1]][atomnamesdict[v2]] = atomnamesdict[k2]
