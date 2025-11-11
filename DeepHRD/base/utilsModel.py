#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

import numpy as np
import torch
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
from PIL import Image, PngImagePlugin
PngImagePlugin.SAFEBLOCK = 1048576
import pandas as pd
import os
from scipy.stats import norm
import random
import time
import sys

def safe_open(path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")  # strip ICC profile
        return img
    except ValueError as e:
        if "Decompressed Data Too Large" in str(e):
            print(f"[WARNING] Skipping corrupted ICC profile in {path}")
            img = Image.open(path, formats=["PNG"])
            img.info.pop("icc_profile", None)
            img = img.convert("RGB")
            return img
        raise


def runMultiGpuTraining (i, iModels, pythonVersion, outputPath, batch_size, dropoutRate, resolution, workers, epochs, checkpointModel=None):
	for currentModel in iModels:
		if resolution == "5x":
			testCommand = pythonVersion + " base/train_mp.py --train_lib " + os.path.join(outputPath, "trainData.pt") + " --val_lib " + os.path.join(outputPath, "valData.pt") +  " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) +  " --batch_size " + str(batch_size) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) + " --epochs " + str(epochs)
		elif resolution == "20x":
			testCommand = pythonVersion + " base/train_mp.py --train_lib " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1), "trainData20x.pt") + " --val_lib " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1), "valData20x.pt") +  " --output " + os.path.join(outputPath, "training_20x_m" + str(currentModel+1)) +  " --batch_size " + str(batch_size) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) + " --epochs " + str(epochs)
		else:
			print("Resolution " + resolution + " is not currently supported.")
			sys.exit()
		# time.sleep(random.randrange(0, 4))
		if checkpointModel:
			testCommand += " --model " + checkpointModel
		os.system(testCommand)
		torch.cuda.empty_cache()



def runMultiGpuInference (i, iModels, pythonVersion, outputPath, modelPath, batch_size, dropoutRate, resolution, workers, BN_reps):
	print("IMODELS")
	print(iModels)
	for currentModel in iModels:
		print(f"currentModel:{currentModel}")
		if resolution == '5x':
			# Non-dropout inference for extracting features of each tile.
			testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(currentModel+1) + ".pth") +  " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers)
			testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_" + resolution + ".tsv")
			
			# time.sleep(random.randrange(0, 4))
			os.system(testCommand)
			os.system(testCommand2)
			


			# Additional inference for all BN-reps with the specified dropout rate (default 0.2).
			if dropoutRate > 0:
				testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(currentModel+1) + ".pth") + " --batch_size " + str(batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers)
				testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions_" + resolution + ".csv")
				os.system(testCommand)
				os.system(testCommand3)
				torch.cuda.empty_cache()	

		else:
			# Non-dropout inference for extracting features of each tile.
			testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i+1), "ROI", "testData20x.pt") + " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(currentModel+1) + ".pth") +  " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers)
			testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_" + resolution + ".tsv")
			# time.sleep(random.randrange(0, 4))
			os.system(testCommand)
			os.system(testCommand2)


			# Additional inference for all BN-reps with the specified dropout rate (default 0.2).
			if dropoutRate > 0:
				testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i+1), "ROI", "testData20x.pt")+ " --output " + os.path.join(outputPath, "m" + str(currentModel+1)) + " --model " + os.path.join(modelPath,resolution + "_m" + str(i+1) + ".pth") + " --batch_size " + str(batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(dropoutRate) + " --resolution " + resolution + " --workers " + str(workers) 		
				testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "m" + str(currentModel+1), "predictions_" + resolution + ".csv")
				os.system(testCommand)
				os.system(testCommand3)
				torch.cuda.empty_cache()	
		


def generateFeatureVectorsUsingBestModels (i, iModels, project, projectPath, pythonVersion, outputPath, batch_size, dropoutRate, resolution, bestModels, checkpointModel=None):
	for l, currentModel in enumerate(iModels):
		modelPath = os.path.join(outputPath, "training_m" + str(currentModel+1))

		# Select best checkpoint or use the model number specified by the user
		existingCheckpointModels = [x for x in os.listdir(modelPath) if ".pth" in x]
		existingCheckpointModelNumbers = [int(x.split("checkpoint_best_5x_")[1].split(".")[0]) for x in existingCheckpointModels if ".pth" in x]
		if bestModels[l] != None:
			bestModel = os.path.join(modelPath, existingCheckpointModels[existingCheckpointModelNumbers.index(bestModels[currentModel])])
		else:
			bestModel = os.path.join(modelPath, existingCheckpointModels[existingCheckpointModelNumbers.index(max(existingCheckpointModelNumbers))])

		# Run Train, Validation, and test data through best checkpoint from above
		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "trainData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_train.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_train.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)

		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "valData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_val.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_val.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)

		testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "testData.pt") + " --output " + os.path.join(outputPath, "training_m" + str(currentModel+1)) + " --model " + bestModel + " --batch_size " + str(batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
		testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "predictions_test.csv")
		testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors.tsv") + " " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_test.tsv")
		os.system(testCommand)
		os.system(testCommand2)
		os.system(testCommand3)	

		torch.cuda.empty_cache()



def runMultiGpuROIs (i, iModels, project, projectPath, pythonVersion, outputPath, maxROI, max_cpu, stain_norm, removeBlurry, predict=False):
	for currentModel in iModels:
		if predict:
	
			roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " +  os.path.join(outputPath, "m" + str(currentModel+1), "ROI") + " --objectiveFile " + \
                                            os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath, project) + " --tileConv " + \
                                            os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath, "m" + str(currentModel+1), "feature_vectors_test_5x.tsv") + \
                                            " --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu) + " --predict"


		else:
			    roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " +  os.path.join(outputPath, "training_20x_m" + str(currentModel+1)) + " --objectiveFile " + \
                            os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath, project) + " --tileConv " + \
                            os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_test.tsv") + \
                            " --train_lib " + os.path.join(outputPath, "trainData.pt") + " --feature_vectors_train " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_train.tsv") + \
                            " --val_lib " + os.path.join(outputPath, "valData.pt") + " --feature_vectors_val " + os.path.join(outputPath, "training_m" + str(currentModel+1), "feature_vectors_val.tsv") + \
                            " --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu)
		if removeBlurry:
			 roiCommand +=" --removeBlurry"
		if stain_norm:
			 roiCommand +=" --stain_norm"
			

		os.system(roiCommand)
		torch.cuda.empty_cache()



def selectBestModel (predictionsPath):
	predictions5x = pd.read_csv(predictionsPath, header=0, index_col=0)
	avgPredictions = predictions5x.loc[:,predictions5x.columns.str.endswith("AverageProb")]
	bestModels = pd.DataFrame(index=predictions5x.index, columns=avgPredictions.columns)

	for sample in predictions5x.index:
		bestModels.loc[sample] = abs(avgPredictions.loc[sample] - float(predictions5x.loc[sample, 'Ensemble-Probability']))

	bestModel = bestModels.mean(axis=0).astype(float).idxmin().split("-")[0]

	return(bestModel)

def collectSampleIndex (file, tileConversionMatrix):
	sample = file.split("/")[-1].split(".")[0]
	sampleIndex = int(tileConversionMatrix.loc[sample].iloc[0])
	if sampleIndex < 10:
		sampleIndex = "00" + str(sampleIndex)
	elif 100 > sampleIndex >= 10:
		sampleIndex = "0" + str(sampleIndex)
	else:
		sampleIndex = str(sampleIndex)
	return(sampleIndex)



def z_test (x, mu, sigma):
	'''
	Performs a z-test for statistical comparisons of simulated and original data.

	Parameters:
			x	->	observed number in original sample (mutation count; int)
		   mu	->	average number observed in simulations (average mutation count; float)
		sigma	->	standard deviation in simulations (float)

	Returns:
		z	->	z-score (float)
		p	->	associated p_value (float)
	'''
	z = (x-mu)/sigma
	p = 2*min(norm.cdf(z), 1-norm.cdf(z))
	return(z, p)


def multiResolution (outputPath, nModels, dropoutRate, threshold):
	mat5x = pd.read_csv(os.path.join(outputPath, "predictions_5x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"), index_col=0, header=0)
	mat20x = pd.read_csv(os.path.join(outputPath, "predictions_20x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"), index_col=0, header=0)
	# samples = list(set(mat5x.index & mat20x.index))
	samples = list(set.intersection(set(mat5x.index), set(mat20x.index)))
	mat5x = mat5x.loc[samples]
	mat20x = mat20x.loc[samples]

	finalMat = pd.DataFrame(columns=["file", "target", "HRD-prediction", "Multi-Res-prediction", "LowerCI", "UpperCI", "p-value"])
	finalMat['file'] = samples
	finalMat = finalMat.set_index('file')
	finalMat['target'] = list(mat20x['target'])
	finalMat['Multi-Res-prediction'] = list((mat5x['Ensemble-Probability'] + mat20x['Ensemble-Probability'])/2)
	finalMat['HRD-prediction'] = (finalMat['Multi-Res-prediction'] > threshold).astype(int)

	ensemblePredictions = pd.concat([mat5x.loc[:,mat5x.columns.str.endswith("Prob")], mat20x.loc[:,mat20x.columns.str.endswith("Prob")]], axis=1)

	finalMat['LowerCI'] = list(ensemblePredictions.quantile(0.025, axis=1))
	finalMat['UpperCI'] = list(ensemblePredictions.quantile(0.975, axis=1))
	for sample in finalMat.index:
		predictions = ensemblePredictions.loc[sample]
		z, p = z_test(threshold, np.mean(predictions), np.std(predictions))
		finalMat.loc[sample, 'p-value'] = p

	finalMat.to_csv(os.path.join(outputPath, "DeepHRD_report_5x_20x_n" + str(nModels) + "_dropout" + str(dropoutRate) + ".csv"))


def combinePredictions (resolution, outputPath, nModels, dropoutRate):

	stdev = 1
	# bestModel = None
	for i in range(nModels):
		# modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions_" + resolution + "_m" + str(i+1) + "_" + str(dropoutRate) + "_temp.csv")
		# modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions.csv")
		modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions_" + resolution + ".csv")
		newPredictions = pd.read_csv(modelPath, header=0, index_col=0)
		if i == 0:
			finalPredictions = newPredictions
			finalPredictions['Ensemble-LowerCI'] = 0
			finalPredictions['Ensemble-UpperCI'] = 0
			finalPredictions = finalPredictions.rename(columns={'probability':'Ensemble-Probability'})

		else:
			finalPredictions['Ensemble-Probability'] += newPredictions['probability']
			finalPredictions = pd.concat([finalPredictions, newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")]], axis=1)
		finalPredictions["m" + str(i+1) + "-AverageProb"] = newPredictions['probability']
		finalPredictions["m" + str(i+1) + "-LowerCI"] = newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")].quantile(0.025, axis=1)
		finalPredictions["m" + str(i+1) + "-UpperCI"] = newPredictions.loc[:,newPredictions.columns.str.startswith("BN_rep")].quantile(0.975, axis=1)

		
	finalPredictions['Ensemble-Probability'] = finalPredictions['Ensemble-Probability']/nModels
	finalPredictions['Ensemble-LowerCI'] = finalPredictions.loc[:,finalPredictions.columns.str.startswith("BN_rep")].quantile(0.025, axis=1)
	finalPredictions['Ensemble-UpperCI'] = finalPredictions.loc[:,finalPredictions.columns.str.startswith("BN_rep")].quantile(0.975, axis=1)
	finalPredictions = finalPredictions.loc[:,~finalPredictions.columns.str.startswith("BN_rep")]
	finalPredictions.to_csv(os.path.join(outputPath, "predictions_" + resolution + "_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"))



def groupTopKtilesTesting (groups, data,k):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide.

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		k:			[int]	The number of top k tiles to consider for each slide. The default is 1; using 
							only the maximum predicted tile probabilite as the final probability for the entire 
							tissue slide (standard MIL assumption).

	Returns:
		1. The indeces for the top k tiles with hightest probabilites for each tissue slide. If k=1, this will
		   return a single tile index for each slide.
		2. The indeces to directly access the relevant slide indeces for each tile
		3. The probabilities each corresponding top tile

	'''
# 	print("\n--- INSIDE groupTopKtilesProbabilities ---")
# 	print(f"Length of 'groups' received: {len(groups)}")
# 	print(f"Length of 'data' (probs) received: {len(data)}")
# 	print(f"Value of 'nmax' (number of slides) received: {nmax}")
# 	print("------------------------------------------\n")
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-k:] = True
	index[:-k] = groups[k:] != groups[:-k]
	return (list(order[index]), list(groups[index]), list(data[index]))


def groupTopKtiles (groups, data,k=1):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide. 

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		k:			[int]	The number of top k tiles to consider for each slide. The default is 1; using 
							only the maximum predicted tile probabilite as the final probability for the entire 
							tissue slide (standard MIL assumption).

	Returns:
		The indeces for the top k tiles with hightest probabilites for each tissue slide. If k=1, this will
		return a single tile index for each slide.
	'''
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-k:] = True
	index[:-k] = groups[k:] != groups[:-k]
	return (list(order[index]))


def groupTopKtilesProbabilities (groups, data, nmax):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Groups the top k tiles from each slide.

	Parameters:
		groups:		[list]	The tissue slide indeces for each corresponding tile.
		data:		[list]	The probabilites after running an inference pass over the dataset
		nmax:		[int]	The number of tissue slides in the given dataset

	Returns:
		The maximum tile probabilities for each slide. 

	'''
	out = np.empty(nmax)
	out[:] = np.nan
	order = np.lexsort((data, groups))
	groups = groups[order]
	data = data[order]
	index = np.empty(len(groups), 'bool')
	index[-1] = True
	index[:-1] = groups[1:] != groups[:-1]
	out[groups[index]] = data[index]
	return (out)



def calculateError (pred,real):
	'''
	Function edited from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

	Calculates the error between the predicted and real values.

	Parameters:
		pred:		[list]	The predicted class labels after performing an inference pass over the validation set
		real:		[list]	The target class labels of the validation set

	Returns:
		The error, false positive rate, and false negative rate.
	'''
	real = [1 if x[1] >= 0.5 else 0 for x in real]
	pred = np.array(pred)
	real = np.array(real)
	eq = np.equal(pred, real)
	accuracy = float(eq.sum()) / pred.shape[0]
	real_pos_count = (real == 1).sum()
	real_neg_count = (real == 0).sum()
	neq = np.not_equal(pred, real)

	# False Positives: Predicted 1, but was 0
	fp = float(np.logical_and(pred == 1, neq).sum())
	# False Negatives: Predicted 0, but was 1
	fn = float(np.logical_and(pred == 0, neq).sum())

	# Calculate rates, handling division by zero
	fpr = fp / real_neg_count if real_neg_count > 0 else 0.0
	fnr = fn / real_pos_count if real_pos_count > 0 else 0.0
	return (accuracy, fpr, fnr)


from collections import defaultdict
import random

import torch.utils.data as data
import torch
import numpy as np
import random
from collections import defaultdict
from PIL import Image # Make sure PIL is imported

# --- This helper function is needed by __getitem__ ---
def safe_open(path):
	"""
    Tries to open an image path. Returns a blank white image on failure.
    """
	try:
		# Assuming path is a path-like object
		return Image.open(path)
	except Exception as e:
		print(f"Error opening image {path}: {e}")
		# Return a blank white image on failure
		return Image.new('RGB', (224, 224), (255, 255, 255))

	# --- REPLACE YOUR ENTIRE MILdataset CLASS WITH THIS ---

class MILdataset(data.Dataset):
	'''
    Class edited used from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).
    Instantiates a MIL dataset.
    '''

	def __init__(self, libraryfile, transform=None):
		'''
        Initializes all class atributes.
        *** This is your original __init__ logic, which is CORRECT for your data. ***
        '''
		lib = torch.load(libraryfile, map_location='cpu')

		grid = []
		slideIDX = []

		# This checks if 'tiles' key exists, which is what your .pt file has
		if 'tiles' not in lib:
			print(f"[FATAL ERROR in MILdataset] Your .pt file '{libraryfile}' is missing the required 'tiles' key.")
			self.grid = []
			self.slideIDX = []
			self.targets = []
			self.subtype = []
			self.slidenames = []
		else:
			print(f"[INFO MILdataset] Loading tiles from '{libraryfile}'...")
			# This logic is correct for your 'list of lists' structure
			for i, g in enumerate(lib['tiles']):
				grid.extend(g)
				slideIDX.extend([i] * len(g))

			self.slidenames = lib['slides']
			self.targets = lib['targets']
			self.subtype = lib ["subtype"]# This is your list of torch.Tensors
			self.grid = grid
			self.slideIDX = slideIDX
			print(f"[INFO MILdataset] Loaded {len(self.grid)} total tiles from {len(self.slidenames)} slides.")

		self.transform = transform
		self.mode = 1  # Default to inference mode
		self.t_data = [] # Initialize t_data

	def modelState(self, mode):
		'''Changes the current mode either to inference or training'''
		self.mode = mode
	def setTransforms(self, transforms):
		self.transform = transforms

	# --- This is the NEW, CORRECT maketraindata function ---
	def maketraindata(self, idxs):
		'''
        Generates the training dataset by performing WEIGHTED SAMPLING of slides
        (based on subtype) and then flattening the tile list.
        'idxs' are the Top-K tile indices *for this epoch*.
        '''

		# 1. Group all top-k tile data by their slideID
		slide_to_tiles = defaultdict(list)
		for x in idxs:
			if x >= len(self.slideIDX):
				print(f"[WARN] maketraindata: Index {x} out of bounds. Skipping.")
				continue

			slide_id = self.slideIDX[x]

			# --- [THIS IS THE MISSING LINE] ---
			# This line defines the 'tile_data' variable before it's used.
			tile_data = (slide_id, self.grid[x], self.targets[slide_id])
			slide_id = self.slideIDX[x]
			slide_to_tiles[slide_id].append(tile_data)

		# 2. Get a list of all unique slide IDs that have tiles *in this epoch*
		#    This is the "available samples" list you're talking about.
		unique_slide_ids = list(slide_to_tiles.keys())

		if not unique_slide_ids:
			print("[ERROR] maketraindata: No slide IDs found. t_data will be empty.")
			self.t_data = []
			return

		# --- [THIS IS THE DYNAMIC WEIGHTING LOGIC] ---
		try:
				# 3. Get the subtype for *only these available slides*
				slide_subtypes = [self.subtype[slide_id] for slide_id in unique_slide_ids]

				# 4. Calculate weights based *only* on the counts from this epoch
				from collections import Counter
				subtype_counts = Counter(slide_subtypes)

				subtype_weights = {
					subtype: 1.0 / count
					for subtype, count in subtype_counts.items()
				}

				# 5. Create a list of weights *in the same order* as unique_slide_ids
				slide_weights = [subtype_weights[subtype] for subtype in slide_subtypes]

				# 6. Perform a weighted shuffle
				shuffled_slide_ids = random.choices(
					population=unique_slide_ids,
					weights=slide_weights,
					k=len(unique_slide_ids)
				)
				# This print statement will prove it's working (e.g., counts will change each epoch)
				print(f"[INFO] maketraindata: Calculated epoch-specific subtype counts: {subtype_counts}")

		except Exception as e:
				print(f"[ERROR] maketraindata: Error during weighting: {e}. Falling back to random.shuffle.")
				shuffled_slide_ids = unique_slide_ids
				random.shuffle(shuffled_slide_ids)

			# 7. Build the new t_data (still bag-contiguous)
		self.t_data = []
		for slide_id in shuffled_slide_ids:
				self.t_data.extend(slide_to_tiles[slide_id])

		print(f"[INFO] maketraindata: Created new weighted training set with {len(self.t_data)} tiles from {len(unique_slide_ids)} unique slides.")
		sampled_subtypes = [self.subtype[sid] for sid in shuffled_slide_ids]
		from collections import Counter
		print(f"[DEBUG] Sampled subtype distribution this epoch: {Counter(sampled_subtypes)}")
	def __getitem__(self, index):
			'''
			Accesses a tile based upon the preset mode (inference or training)
			'''
			if self.mode == 1:
				slideIDX = self.slideIDX[index]
				target = self.targets[slideIDX] # Gets the tensor for this slide
				tile_path = self.grid[index]
				img = safe_open(tile_path) # Use the safe_open function

			# Mode 2: Training - use the subset t_data
			elif self.mode == 2:
				slideIDX, tile_path, target = self.t_data[index] # target is already the tensor
				img = safe_open(tile_path) # Use the safe_open function

			else:
				# Handle case where mode is not set
				raise IndexError(f"Dataset mode not set to 1 or 2. Current mode: {self.mode}")

			# Apply transformations
			if self.transform is not None:
				img = self.transform(img)

			# target is already a tensor, so it's ready for the model
			return (img, target, slideIDX)
	def __len__(self):
		'''
        Returns the length of the given dataset, whether it's the training or inference sets
        '''
		if self.mode == 1:
			return len(self.grid)
		elif self.mode == 2:
			return len(self.t_data)
		else:
			return 0 # Return 0 if mode is not set
