import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.options.mode.chained_assignment = None

MAX_DISPLACEMENT = 50

def read_dataset(datasets : str, type : str, lag_amout = 0, with_angle = False, with_position = False, full_dataset = False):
        
        
        df = pd.read_csv(datasets)
        if with_position:
            x = df[["x", "y","dx", "dy", "dt"]] # i removed dt ! 
        else:
            x = df[["dx", "dy", "dt"]]
        
        x['dt'] = x['dt'] / 1000
 
        if with_angle:
            x["angle_tan"] = np.arctan2(df["dy"], df["dx"])
            if full_dataset:
                compute_velocities(x)
                compute_accelerations(x)
                compute_jerk(x)
                compute_curvature(x)
                compute_angular_velocity(x)


        targets = df[["x_to", "y_to"]]
        y = construct_ground_truth(df[['dx','dy']],df[["x", "y"]], df[["x_to", "y_to"]], type)

        for k in range(lag_amout -1):
            x[f"dx_{k+1}"] = x["dx"].shift(k+1)
            x[f"dy_{k+1}"] = x["dy"].shift(k+1)
            if with_angle:
                x[f"angle_tan_{k+1}"] = x["angle_tan"].shift(k+1)


        x.fillna(0, inplace=True)
        traj = df[["x", "y"]]

        
        return x, y, targets, traj

class FittsDataset(Dataset):
    def __init__(self,x, y):        
        self.data = x
        self.y_gt = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data.iloc[idx, :].to_numpy(), self.y_gt.iloc[idx, :].to_numpy()
        return self.data[idx, :], self.y_gt[idx, :]
    
class FittsDatasetSeq(Dataset):
    def __init__(self,x, y, sequence_length):        
        self.data = x
        self.y_gt = y
        self.seq_l = sequence_length

    def __len__(self):
        return len(self.data) - self.seq_l

    def __getitem__(self, idx):
        # return self.data.iloc[idx:idx+self.seq_l, :].to_numpy(), self.y_gt.iloc[idx + self.seq_l, :].to_numpy()
        return self.data[idx:idx+self.seq_l, :], self.y_gt[idx + self.seq_l, :]

# target width is hardcoded, need to get him from dataset to ! 
def construct_ground_truth(displacement, cursor_pose, target_pose, type, change_only_dir = False, target_width = 60):
    """
        
        - vec ground truth is the best dx between Ct and Tt it is the believed  
            (IE at each instant it s a straight line leading to the target)
        - second type is an smoothed version of the trajectory to fit a more natural lines between positions and targets
    """
    
    y = {}
    if type == "vec":   
            dx = (target_pose['x_to'] - cursor_pose['x'])
            dy = (target_pose['y_to'] - cursor_pose['y'])
            mag = np.sqrt(dx**2 + dy**2)
            # np.clip(mag.to_numpy(), 0, MAX_DISPLACEMENT, mag)
            angle = np.arctan2(dy.to_numpy(), dx.to_numpy())
            # scale magnitude
            min_mag = 0
            max_mag = MAX_DISPLACEMENT
            mag = (mag - min_mag) / (max_mag - min_mag)



            y['dx'] = mag * np.cos(angle)
            y['dy'] = mag * np.sin(angle)

            # we need to say that if the cursor is in target dx is 0 ! 
            dist_cursor_target = np.sqrt(dx**2 + dy**2)
            indexes = np.where(dist_cursor_target < target_width/2)[0]
            if indexes.shape[0] > 0:    
                y['dx'][indexes] = 0
                y['dy'][indexes] = 0
    if type == "dir":
            # get target direction 
            xtarget_from_cursor = (target_pose['x_to'] - cursor_pose['x'])
            ytarget_from_cursor = (target_pose['y_to'] - cursor_pose['y'])
            dcursor_x = displacement['dx']
            dcursor_y = displacement['dy']
            angle_to_target = np.arctan2(ytarget_from_cursor.to_numpy(), xtarget_from_cursor.to_numpy())
            dcursor_mag = np.sqrt(dcursor_x**2 + dcursor_y**2)

            y['dx'] = dcursor_mag * np.cos(angle_to_target)
            y['dy'] = dcursor_mag  * np.sin(angle_to_target)
        # target_pose['x_to'] - cursor_pose['x']

    # elif type == "smooth":
    #     y['dx'] = target_pose['x_to'] - cursor_pose['x']
    #     y['dy'] = target_pose['y_to'] - cursor_pose['y']
    return pd.DataFrame(y)

def preprocess_dataset(x, y, scaler_type = "minmax", feature_range = (-1, 1)):
    
    y = y.to_numpy()
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
        x = scaler.fit_transform(x)
    elif scaler_type == "std":
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    elif scaler_type == "custom":
        # it is a scaler with my values, not the sklearn one (-1, 1)
        data_min = np.array([0, 0, -MAX_DISPLACEMENT, -MAX_DISPLACEMENT, 0])
        data_max = np.array([1920, 1080, MAX_DISPLACEMENT, MAX_DISPLACEMENT, 500])
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.data_max_ = data_max
        scaler.data_min_ = data_min
        x = scaler.transform(x)
    return x, y, scaler    

def compute_velocities(x):
    """ Compute mouse velocities with x is the dataset
        this function return a new df with columns vdx and vdy and v 

    Args:
        x (df): DataFrame with columns dx and dy and dt at least
    """

    x["vx"] = x["dx"] / x["dt"]
    x["vy"] = x["dy"] / x["dt"]
    x["v"] = np.sqrt(x["vx"]**2 + x["vy"]**2)
    
    return x

def compute_accelerations(x):
    """ Compute mouse accelerations with x is the dataset
        this function return a new df with columns a, ji, wi

    Args:
        x (df): DataFrame with columns v and dt
    """
    dv = np.diff(x["v"], prepend=0)
    x["a"] = dv / x["dt"]  
    return x

def compute_jerk(x):
    """ Compute mouse jerks with x is the dataset
        this function return a new df with columns j

    Args:
        x (df): DataFrame with columns a and dt
    """
    da = np.diff(x["a"], prepend=0)
    x["j"] = da / x["dt"]  
    return x

def compute_angular_velocity(x):
    """ Compute mouse angular_velocity with x is the dataset
        this function return a new df with columns w

    Args:
        x (df): DataFrame with columns vx and vy
    """



    x["w"] = x["angle_tan"] / x["dt"]
    return x

def compute_curvature(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    dtheta = np.diff(x["angle_tan"], prepend=0)
    s = np.cumsum(np.sqrt(x["dx"]**2 + x["dy"]**2))
    ds = np.diff(s, prepend=0)
    x["c"] = dtheta / ds
    return x


if __name__ == "__main__":
    
    write_to_file = True
    exp_name = "P0_C0"
    x, y, _=read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{exp_name}.csv", "vec", True, True)
    
    print("x : \n", x)
    print("y : \n", y)

    # print("y mag : \n", np.sqrt(y['dx']**2 + y['dy']**2))
    # if write_to_file:
    #     for col_name in y.columns:
    #         y.rename(columns={col_name: col_name + "_true"}, inplace=True)
        
    #     data = pd.concat([x, y], axis=1)

    #     path = "datasets_java/P0_C0_x.csv"
    #     data.to_csv(path, index=False)

    print("input feature name : ", list(x.columns))
    print("output feature name : ", list(y.columns))

