import numpy as np
from utils.utils import PDB, clean_pdb, batchify_dmatrix_computation, batchify_dmatrix_computation_for_dmin, match_waters

class WATERS(object):
    '''
    Class object to load only PDB water from files of pdb format.

    Parameters
    ----------
    filename : str
        Path to file.

    Attributes
    ----------
    water_coords : numpy array
        Coordinates of water atoms, note that here we only keep oxygen atoms, shape (W,3).

    wb : numpy array
        B factor of water coordinates, shape (W).
    '''
    def __init__(self, filename):
        self.water_coords, self.wb = [], []
        if filename.split('.')[-1] == 'gz':
            with gzip.open(filename, 'rt', encoding='utf-8') as file:
                f = file.readlines()
        else:
            with open(filename, 'r') as file:
                f = file.readlines()
        lineslen = len(f)
        for i in range(lineslen):
            line = f[i]
            sline = line.split()
            if line[:6] == 'HETATM':
                if line[13] == 'O' and ((line[17:20]=='HOH') or (line[17:20]=='TIP') or (line[17:20]=='WAT')):
                    self.water_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    try:
                        self.wb.append(float(line[61:68]))
                    except:
                        pass
                    continue
        self.water_coords = np.array(self.water_coords)
        self.wb = np.array(self.wb)
        return

def test_evaluation(prediction_paths, reference_paths, cap, radius, thresholds):
    '''
    Function to compute evaluation metrics on your test set.

    Parameters
    ----------
    prediction_paths : list
        list of paths of all prediction pdb files, lenght (N).

    reference_paths : list
        list of paths of all reference pdb files, lenght (N).

    cap : float
        Cap value to select your water coordinates, if cap=0 no capping is applied.

    radius : float
        Radius distance to keep water coordinates from non hydrogen protein atoms.

    thresholds : tuple or list
        Threshold distance values to count as hit predicted water from ground truth water

    Returns
    -------
    match_results : numpy array
        Contains all the results to compute the evaluation metrics, shape (N,3*len(thresholds))

    rmsd : list
        List that computed all the one by one distances of ground truth from TP predicted waters.
    '''
    #Define array to keep matching results
    match_results = np.zeros((len(prediction_paths), 3*len(thresholds)))
    #Define list to keep rmsd results
    rmsd = []
    for i in range(len(thresholds)):
        rmsd.append([])
    for i, paths in enumerate(zip(prediction_paths, reference_paths)):
        pred_path, ref_path = paths
        #Load prediction pdb
        pred = WATERS(pred_path)
        #Load reference pdb
        pdb = PDB(ref_path)
        #Clean ref pdb
        clean_pdb(pdb)
        #First cap prediction accordingly
        if cap:
            pred.water_coords = pred.water_coords[pred.wb>=cap]
        #Calculate Distance matrix between pdb.water_coords and pdb.coords.
        d = batchify_dmatrix_computation(pdb.water_coords, pdb.coords)
        #Keep waters within radius
        water_IDs = np.where(d<=radius)[0]
        #Get water coordinates only within radius Angstrom distance from protein and HETATM
        reduced_water_coords = pdb.water_coords[np.unique(water_IDs)]
        #Calculate Distance matrix between pred.water_coords and pdb.coords.
        d = batchify_dmatrix_computation(pred.water_coords, pdb.coords)
        #Keep waters within radius
        water_IDs = np.where(d<=radius)[0]
        #Get water coordinates only within radius Angstrom distance from protein and HETATM
        pred.water_coords = pred.water_coords[np.unique(water_IDs)]
        if not len(pred.water_coords):
            for disind, thresh in enumerate(thresholds):
                #Match waters
                match_results[i,3*disind:3+3*disind] = 0, 1e-3, len(reduced_water_coords)
        else:
            #Calculate Distance matrix
            d = batchify_dmatrix_computation(reduced_water_coords, pred.water_coords)
            #Calculate closest prediction position for each GT water
            dargmin = np.argmin(d,1)
            #Keep these min values for each GT water
            dmin = d[np.array(list(range(len(d)))),np.argmin(d,1)]
            #Compute metrics
            water_matches_all = []
            for disind, thresh in enumerate(thresholds):
                #Match waters
                water_matches_all.append(match_waters(d, dmin, dargmin, thresh))
                match_results[i,3*disind:3+3*disind] = len(water_matches_all[-1]), len(pred.water_coords), len(reduced_water_coords)
                #RMSD
                water_matches_rmsd = np.array(water_matches_all[-1])
                if len(water_matches_rmsd):
                    diff = np.sum((reduced_water_coords[water_matches_rmsd[:,0]]-pred.water_coords[water_matches_rmsd[:,1]])**2,axis=1)
                    rmsd[disind] += diff.tolist()
    return match_results, rmsd

def test_pp_complex_evaluation(prediction_paths, reference_paths, complex_reference_path, cap, radius, complex_distance, thresholds, fingerprint):
    '''
    Function to compute evaluation metrics on the specific protein-protein complex test set.

    Parameters
    ----------
    prediction_paths : list
        list of paths of all prediction pdb files, lenght (N).

    reference_paths : list
        list of paths of all reference pdb files, lenght (N).

    cap : float
        Cap value to select your water coordinates, if cap=0 no capping is applied.

    radius : float
        Radius distance to keep water coordinates from non hydrogen protein atoms.

    thresholds : tuple or list
        Threshold distance values to count as hit predicted water from ground truth water

    Returns
    -------
    match_results : numpy array
        Contains all the results to compute the evaluation metrics, shape (N,3*len(thresholds))

    rmsd : list
        List that computed all the one by one distances of ground truth from TP predicted waters.
    '''
    #Define array to keep matching results
    match_results = np.zeros((len(prediction_paths), 3*len(thresholds)))
    #Define list to keep rmsd results
    rmsd = []
    for i in range(len(thresholds)):
        rmsd.append([])
    for i, paths in enumerate(zip(prediction_paths, reference_paths)):
        #Define paths
        pred_path, ref_path = paths
        #Find fingerprint
        f = fingerprint[np.where(fingerprint==ref_path.split('.')[0].split('/')[-1])[0][0]][0]
        #Load PDB files
        pred = WATERS(pred_path)
        pdb_1 = PDB(f'{complex_reference_path}{f}_pro_b1.pdb')
        pdb_2 = PDB(f'{complex_reference_path}{f}_pro_b2.pdb')
        #pdb_w = PDB(f"{pdbpath_ref}{f}_water.pdb")
        pdb = PDB(ref_path)
        #Clean PDBs
        clean_pdb(pdb)
        clean_pdb(pdb_1)
        clean_pdb(pdb_2)
        #Find the atoms that define the interface between the two proteins
        d = np.where(batchify_dmatrix_computation(pdb_1.coords, pdb_2.coords)<complex_distance)
        pdb_1_coords, pdb_2_coords = pdb_1.coords[d[0]],pdb_2.coords[d[1]]
        #Find ground truth waters within the cavity of these selected protein atoms
        reduced_water_coords = pdb.water_coords[np.intersect1d(np.where(batchify_dmatrix_computation_for_dmin(pdb_1_coords, pdb.water_coords)<radius)[0],\
                                                               np.where(batchify_dmatrix_computation_for_dmin(pdb_2_coords, pdb.water_coords)<radius)[0])]
        #First cap prediction accordingly
        if cap:
            pred.water_coords = pred.water_coords[pred.wb>=cap]
        #Find predicted waters within the cavity of these selected protein atoms
        pred.water_coords = pred.water_coords[np.intersect1d(np.where(batchify_dmatrix_computation_for_dmin(pdb_1_coords, pred.water_coords)<radius)[0],\
                                                             np.where(batchify_dmatrix_computation_for_dmin(pdb_2_coords, pred.water_coords)<radius)[0])]
        #If you have neither predicted nor ground truth coordinates
        if not len(pred.water_coords) and not len(reduced_water_coords):
            for disind, thresh in enumerate(thresholds):
                #Match waters
                match_results[i,3*disind:3+3*disind] = 0, 1e-4, 1e-4
            continue
        #If there are no predicted water coordinates
        if not len(pred.water_coords):
            for disind, thresh in enumerate(thresholds):
                #Match waters
                match_results[i,3*disind:3+3*disind] = 0, 1e-4, len(reduced_water_coords)
            continue
        #If there are no ground truth water coordinates
        if not len(reduced_water_coords):
            for disind, thresh in enumerate(thresholds):
                #Match waters
                match_results[i,3*disind:3+3*disind] = 0, len(pred.water_coords), 1e-4
            continue
        #Calculate Distance matrix of water coordinates
        d = batchify_dmatrix_computation(reduced_water_coords, pred.water_coords)
        #Calculate closest prediction position for each GT water
        dargmin = np.argmin(d,1)
        #Keep these min values for each GT water
        dmin = d[np.array(list(range(len(d)))),np.argmin(d,1)]
        #Compute metrics
        water_matches_all = []
        for disind, thresh in enumerate(thresholds):
            #Match waters
            water_matches_all.append(match_waters(d, dmin, dargmin, thresh))
            match_results[i,3*disind:3+3*disind] = len(water_matches_all[-1]), len(pred.water_coords), len(reduced_water_coords)
            #RMSD
            water_matches_rmsd = np.array(water_matches_all[-1])
            if len(water_matches_rmsd):
                diff = np.sum((reduced_water_coords[water_matches_rmsd[:,0]]-pred.water_coords[water_matches_rmsd[:,1]])**2,axis=1)
                rmsd[disind] += diff.tolist()
    return match_results, rmsd
