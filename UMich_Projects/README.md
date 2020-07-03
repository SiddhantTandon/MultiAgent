# AEROSP 740: Multi-Agent Control Course Projects

## Instructor: Dr. Dimitra Panagou
## Author: Siddhant Tandon
## Semester: Winter 2020

### Gorodetsky Group: Computational Autonomy
#### Author: Siddhant Tandon
#### Date: June 29 2020

###########################################################################################################################
#### Description
 - This code classifies the Atari GC human players' datasets into experts and non-xperts
 - Requirements - Trajectory dataset of human played games in separate files with total reward information
 - This jupyter notebook is using Ms Pacman dataset for demonstration purposes
 
###########################################################################################################################
#### Paper
This code is used on the dataset compiled by CMU. The research paper is titled ['Atari Grand Challenge Dataset'](https://arxiv.org/pdf/1705.10998.pdf).

### Preliminary

First download the Atari GC [version 1 or version 2 dataset](http://atarigrandchallenge.com/data) for any or all games. 

Next open the downloaded zip file and extract the dataset to the desired directory for use. 


```python
# All the libraries used are declared here
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns
import shutil
from IPython.display import display
```


```python
# Directory name goes here
dirname = '..//Grand_Atari_Dataset//mspacman//atari_v1//trajectories//mspacman' # change to your directory!
```

Please change the above directory name to the directory where the Atari GC v1 or v2 folders are saved locally. For example - mspacman trajectory files (.txt) can be obtained using `dirname = '..//Grand_Atari_Dataset//mspacman//atari_v1//trajectories//mspacman'`. Please note that while using windows operating system, the path address will have to changed from `\` to `//` for python to recognize it as a path.   

#### Preliminary - Player Count


```python
# counting number of human players' datasets
count = 0 # counter
save_path = {} # directory to save all filenames, not necessarily needed
for path in os.listdir(dirname): # for loop that goes through all the files in the path given under dirname
    save_path[path] = path
    if os.path.isfile(os.path.join(dirname, path)): # if the files exist under this directory then we increase counter size
        count += 1
print('Number of files = {} \n'.format(count)) # printing the number of files representing played games by human players
```

    Number of files = 382 
    
    

#### Preliminary - Player Dataframes 


```python
# creating dataframe for each of the player datasets
all_files = glob.glob(dirname + "/*.txt") # selecting all the .txt files under the directory
d = {} # dictionary to save all ourr dataframes
retrieved_files = {} # dictionary is to save the file names
for f in save_path:
    retrieved_files[f] = save_path[f].split('.txt')[0] # splitting .txt from file names

#display(retrieved_files)
```


```python
# split dictionary into keys and values 
keys_files = [] 
values_files = [] 
items = retrieved_files.items() 
for item in items: 
    keys_files.append(item[0]), values_files.append(item[1])
    
```


```python
# saving file numbers
retrieved_files_values = pd.DataFrame(values_files, columns =[
   'Value']) # this variable stores all the file names as a dataframe now

display(retrieved_files_values)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.txt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.txt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>102.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103.txt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>377</th>
      <td>95.txt</td>
    </tr>
    <tr>
      <th>378</th>
      <td>96.txt</td>
    </tr>
    <tr>
      <th>379</th>
      <td>97.txt</td>
    </tr>
    <tr>
      <th>380</th>
      <td>98.txt</td>
    </tr>
    <tr>
      <th>381</th>
      <td>99.txt</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 1 columns</p>
</div>



```python
# saving files as a dataframe in a dictionary of dataframes
for f in all_files:
    d[f] = pd.read_csv(f, sep = ",", header=1) # dictionary now saves all .txt files as dataframes which are callable
    
```


```python
# creating an array which stores all the final scores of players
max_rewards = []  # array to store all the max scores which are read from the dataframes   
for df,k in d.items():
    df = d.get(df) # calling each dataframe
    rew = df['score'].max()
    max_rewards.append(rew) # list
max_rewards_vals = pd.DataFrame(max_rewards, columns=['Value']) # stores values of all max_rewards
max_rewards_files = max_rewards_vals.copy() # copy, not a deep copy, to operate on for later
max_rewards_files['File_Num'] = retrieved_files_values # adding the file name next to its own max reward

max_rewards_files.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>File_Num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3930</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7390</td>
      <td>102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10761</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot
sns.distplot(max_rewards)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21de1ca1e48>




![png](output_14_1.png)


The above distribution plot shows how the rewards (x-axis) are distributed. From the plot we can notice that majority scores are between 0 to 4000. We will use this distribution to figure out what split we want in the dataset to classify experts and non-experts. 


```python
# Sorted list of rewards and file number  
rew_sort = max_rewards_vals.copy() # copy for operations
rew_sort.sort_values(by = ["Value"], inplace=True) # sorting rewards
max_rewards_files.sort_values(by=['Value'], inplace = True) # sorting the original variable on rewards column,
# this will rearrange the entire arrow of the reward, hence there
# wont be any need of creating keys for each reward to their file names

max_rewards_files.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>File_Num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144</th>
      <td>10</td>
      <td>329</td>
    </tr>
    <tr>
      <th>149</th>
      <td>10</td>
      <td>339</td>
    </tr>
    <tr>
      <th>47</th>
      <td>10</td>
      <td>167</td>
    </tr>
    <tr>
      <th>137</th>
      <td>10</td>
      <td>32</td>
    </tr>
    <tr>
      <th>326</th>
      <td>10</td>
      <td>611</td>
    </tr>
  </tbody>
</table>
</div>



### Finding Experts


```python
# selecting experts
cut_threshold = np.percentile(rew_sort,80) # here we use an 80 - 20 split on the dataset to classify non-experts and experts

print('Threshold rewards = {}'.format(cut_threshold)) # prints the cutoff to be an expert 
```

    Threshold rewards = 4286.0
    


```python
experts = [] # array to store expert filenames
experts_bool = (rew_sort['Value'] >= cut_threshold) # boolean array which shows True for experts
experts_bool = pd.DataFrame(experts_bool) # converting bollean array to dataframe
ids = experts_bool.loc[experts_bool['Value'] == True] # extracting the indices of each expert
experts = max_rewards_files[max_rewards_files.index.isin(ids.index)] # using the indicies from aboe to extract the rewards as well as filenames of each expert
pd.options.display.max_columns = None
display(experts)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>File_Num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122</th>
      <td>4290</td>
      <td>294</td>
    </tr>
    <tr>
      <th>266</th>
      <td>4320</td>
      <td>532</td>
    </tr>
    <tr>
      <th>304</th>
      <td>4360</td>
      <td>584</td>
    </tr>
    <tr>
      <th>111</th>
      <td>4370</td>
      <td>274</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4390</td>
      <td>21</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>12881</td>
      <td>81</td>
    </tr>
    <tr>
      <th>227</th>
      <td>12961</td>
      <td>466</td>
    </tr>
    <tr>
      <th>256</th>
      <td>13471</td>
      <td>520</td>
    </tr>
    <tr>
      <th>265</th>
      <td>14401</td>
      <td>531</td>
    </tr>
    <tr>
      <th>57</th>
      <td>18981</td>
      <td>178</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 2 columns</p>
</div>


### Copying Experts data to New Path

### A. Screens (Frames)

Depending on the operating system the following is subject to change. 
This code will run on windows platform. 
**Note - make sure that a new folder for experts does not already exist in the diseried target directory. It will give an error.** 


```python
# creating a new list for filenames of experts
experts_files = experts['File_Num']
experts_files.head()
```




    122    294
    266    532
    304    584
    111    274
    78      21
    Name: File_Num, dtype: object




```python
# converting the series to string type and dropping the indicies to get new indicies starting from 0
experts_files = experts_files.apply(str)
experts_files.reset_index(inplace = True, drop = True)
experts_files.head()
```




    0    294
    1    532
    2    584
    3    274
    4     21
    Name: File_Num, dtype: object




```python
# changing directory to screens to make new folder for experts and copy all their respective folders to the new folder
dirname_images = '..//Grand_Atari_Dataset//mspacman//atari_v1//screens//mspacman' # change to your directory!
dirname_images_exp = '..//Grand_Atari_Dataset//mspacman//atari_v1//screens//experts' # change to your directory!
# dirname_images is your directory path to screens folder containing all the images for each expert
# dirname_imgaes_exp is the new folder where you want to copy all the experts. 
```


```python
experts_copy_files = [] # save all the file paths of experts
for files in experts_files:
    experts_copy_files.append(dirname_images +"//"+ files) # adding the filenames as strings to the end of the paths
```

#### convert this 'markdown' cell to 'code' cell  and execute the cell to copy files locally
for index,files in enumerate(experts_copy_files):
    old = files # paths from the screen folder
    new = dirname_images_exp+"//"+ str(experts_files[index]) # paths to the new folder
    new_path = shutil.copytree(old,new) # function which copies all the images inside each expert folder to its new location

### B. Trajectories (txt files)

Similarily we can also move the trajectory files of experts from their source folder to a new expert folder. **Note here we do need to create a new folder called 'experts' in the desired directory before copying files.** 


```python
retrieved_files_traj = {} # dictionary is to save the file names
for f in save_path:
    retrieved_files_traj[f] = save_path[f].split('\\')[-1]
#display(retrieved_files_traj)
```


```python
# split dictionary into keys and values 
keys_files = [] 
values_files = [] 
items = retrieved_files_traj.items() 
for item in items: 
    keys_files.append(item[0]), values_files.append(item[1])
    
# saving file numbers
retrieved_files_traj_values = pd.DataFrame(values_files, columns =[
   'Value']) # this variable stores all the file names as a dataframe now

display(retrieved_files_traj_values)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.txt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.txt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>102.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>103.txt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>377</th>
      <td>95.txt</td>
    </tr>
    <tr>
      <th>378</th>
      <td>96.txt</td>
    </tr>
    <tr>
      <th>379</th>
      <td>97.txt</td>
    </tr>
    <tr>
      <th>380</th>
      <td>98.txt</td>
    </tr>
    <tr>
      <th>381</th>
      <td>99.txt</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 1 columns</p>
</div>



```python
max_rewards_traj_files = max_rewards_vals.copy() # copy, not a deep copy, to operate on for later
max_rewards_traj_files['File_Num'] = retrieved_files_traj_values # adding the file name next to its own max reward
max_rewards_traj_files.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>File_Num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3930</td>
      <td>1.txt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310</td>
      <td>10.txt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>100.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7390</td>
      <td>102.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10761</td>
      <td>103.txt</td>
    </tr>
  </tbody>
</table>
</div>




```python
max_rewards_traj_files.sort_values(by=['Value'], inplace = True) # sorting the original variable on rewards column
experts_traj = max_rewards_traj_files[max_rewards_files.index.isin(ids.index)]
display(experts_traj)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>File_Num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>122</th>
      <td>4290</td>
      <td>294.txt</td>
    </tr>
    <tr>
      <th>266</th>
      <td>4320</td>
      <td>532.txt</td>
    </tr>
    <tr>
      <th>304</th>
      <td>4360</td>
      <td>584.txt</td>
    </tr>
    <tr>
      <th>111</th>
      <td>4370</td>
      <td>274.txt</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4390</td>
      <td>21.txt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>12881</td>
      <td>81.txt</td>
    </tr>
    <tr>
      <th>227</th>
      <td>12961</td>
      <td>466.txt</td>
    </tr>
    <tr>
      <th>256</th>
      <td>13471</td>
      <td>520.txt</td>
    </tr>
    <tr>
      <th>265</th>
      <td>14401</td>
      <td>531.txt</td>
    </tr>
    <tr>
      <th>57</th>
      <td>18981</td>
      <td>178.txt</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 2 columns</p>
</div>



```python
# creating a new list for filenames of experts
experts_traj_files = experts_traj['File_Num']
experts_traj_files.head()
```




    122    294.txt
    266    532.txt
    304    584.txt
    111    274.txt
    78      21.txt
    Name: File_Num, dtype: object




```python
# converting the series to string type and dropping the indicies to get new indicies starting from 0
experts_traj_files = experts_traj_files.apply(str)
experts_traj_files.reset_index(inplace = True, drop = True)
experts_traj_files.head()
```




    0    294.txt
    1    532.txt
    2    584.txt
    3    274.txt
    4     21.txt
    Name: File_Num, dtype: object




```python
# dirname for copying expert trajectroy files into a new folder
dirname_traj = '..//Grand_Atari_Dataset//mspacman//atari_v1//trajectories//mspacman' # change to your directory!
dirname_traj_exp = '..//Grand_Atari_Dataset//mspacman//atari_v1//trajectories//experts' # change to your directory!
```


```python
experts_copy_traj_files = [] # save all the file paths of experts
for files in experts_traj_files:
    experts_copy_traj_files.append(dirname_traj +"//"+ files) # adding the filenames as strings to the end of the paths
#display(experts_copy_traj_files)
```

#### convert this 'markdown' cell to 'code' cell  and execute the cell to copy files locally
for index,files in enumerate(experts_copy_traj_files):
    old_traj = files # paths from the screen folder
    new_traj = dirname_traj_exp+"//"+ experts_traj_files[index] # paths to the new folder
    new_path = shutil.copy(old_traj,new_traj) # function which copies all the images inside each expert folder to its new location

Now you should see both 'experts' folder containing the frames of all experts and trajectories (.txt) files where you desired to save them! 

The above process can be repeated for the other games under the Atari GC Dataset. 

################################################### End of Code ###########################################################
