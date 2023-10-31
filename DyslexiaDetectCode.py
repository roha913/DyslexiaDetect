# This opens the Zipfile containing the data
import zipfile
data_zip = zipfile.ZipFile('Recording DataLore.zip', 'r')
data_zip.extractall('extraction')
data_zip.close()

# This gets the list of files
import os
file_list = os.listdir('extraction/Recording Data')

# This is a function which, given the path, gets the data in the file and returns the position of the eyes
def get_eye_positions(file_name): #file_name is the file name
    file_data = open('extraction/Recording Data/' + file_name + '/A1R.txt', 'r') #opening the file
    file_content = file_data.read() #the contents of the file
    rows = file_content.split("\n") #reading it row by row
    eye_positions = [] #an empty list to contain the positions 
    for i in range(1, (len(rows) - 1)):
        columns = rows[i].replace(",",".").split("\t") #getting the columns for each row
        eye_positions.append(0.5*(float(columns[3]) + float(columns[1]))) #adding the positions to the list
    return(eye_positions)

# This imports the discrete fourier transform(DFT)
# The discrete fourier transform algorithm approximates the data as a sum of sine and cosine waves
# It gives the amplitudes of the waves of different frequencies
from numpy.fft import fft
from math import cos

def get_freqs_for_4_amps(file_name): #getting the frequencies corresponding to the four highest amplitudes
    dft_amplitudes = fft(get_eye_positions(file_name)) #amplitudes from the DFT
    amplitudes = [] #Storing amplitudes
    indices = [] #Storing indices, from which we calculate the frequencies
    for i in range(len(dft_amplitudes)):
        if(round(abs(dft_amplitudes[i].real), 3) != 0.0): #Parsing out the amplitudes which are so small, they are negligible
            amplitudes.append(dft_amplitudes[i].real) #storing amplitudes
            indices.append(i) #storing indices
    sorted_amplitudes = sorted(amplitudes) #sorting the amplitudes so we can pick the highest ones
    amp_freq_dict = dict(zip(amplitudes, indices)) #creating a dictionary between the amplitudes and the corresponding frequencies

    top_four_amps = sorted_amplitudes[-4:] #The four highest amplitudes
    corresponding_freqs = [amp_freq_dict[amp] for amp in top_four_amps] #corresponding frequencies

    #This converts the indices to frequencies and returns a list
    return [cos(6.28*freq/len(amplitudes)) for freq in corresponding_freqs]    

#Setting up our data
amplitude_data = [] #the four amplitudes for each file. A 185 by 4 list. 
dyslexic_labels = [] #If they are dyslexic
for i in range(len(file_list) - 1):
    amplitude_data.append(get_freqs_for_4_amps(file_list[i]))
    dyslexic_labels.append(1 if file_list[i][-1] == '1' or file_list[i][-1] == '2' else 0) #Whether or not they are dyslexic is indicated in the file name

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
amplitude_data = scaler.fit_transform(amplitude_data) #Scaling the data    

from sklearn import svm

# The four import statements below are for testing our model
from sklearn.metrics import recall_score #Recall score is the proportion of dyslexics which the model identifies. 
from sklearn.metrics import precision_score as precise
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score #k-cross validation

k_folds = KFold(n_splits = 5) #k = 5; k-cross validation will use 5 pairs of training and testing data

#The three groups of lines below test out different kernels. The score is the mean of the recall scores for each pair. The linear kernel seems to be the best performing, with a mean score of around 0.86.

# Sigmoid kernel
clf = svm.SVC(kernel = 'sigmoid')
sigmoid_scores = cross_val_score(clf, amplitude_data, dyslexic_labels, cv = k_folds, scoring = make_scorer(recall_score))
print("Sigmoid most four:",  sigmoid_scores, "mean: ", sum(sigmoid_scores)/len(sigmoid_scores))

# RBF kernel
clf = svm.SVC(kernel = 'rbf')
rbf_scores = cross_val_score(clf, amplitude_data, dyslexic_labels, cv = k_folds, scoring = make_scorer(recall_score))
print("RBF most four:",  rbf_scores, "mean: ", sum(rbf_scores)/len(rbf_scores))

# Linear kernel
clf = svm.SVC(kernel = 'linear')
linear_scores = cross_val_score(clf, amplitude_data, dyslexic_labels, cv = k_folds, scoring = make_scorer(recall_score))
print("Linear most four:",  linear_scores, "mean: ", sum(linear_scores)/len(linear_scores))
