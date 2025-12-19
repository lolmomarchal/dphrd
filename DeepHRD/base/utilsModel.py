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
from collections import defaultdict, Counter # Import Counter
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image #

def runMultiGpuTraining (i, iModels, pythonVersion, outputPath, batch_size, dropoutRate, resolution, workers, epochs, checkpointModel=None, validation_interval=1, k=100, weights=0.5, patience=40, sampling_mode='dampened_combined', lambda_sup=0.3, loss_fn='ce', focal_gamma=2.0, focal_alpha=None, k_sup=10, train_inference_dropout_enabled=False, train_inference_transforms_enabled=False):
	for currentModel in iModels:

		# Base command with all common parameters
		base_command = (
			f"{pythonVersion} base/train_mp.py "
			f"--output {os.path.join(outputPath, 'training_m' + str(currentModel+1))} "
			f"--batch_size {str(batch_size)} --gpu {str(i)} "
			f"--dropoutRate {str(dropoutRate)} --resolution {resolution} "
			f"--workers {str(workers)} --epochs {str(epochs)} "
			f"--validation_interval {str(validation_interval)} "
			f"--k {str(k)} --weights {str(weights)} "
			f"--patience {str(patience)} --sampling_mode {sampling_mode} "
			f"--lambda_sup {str(lambda_sup)} --loss_fn {loss_fn} "
			f"--focal_gamma {str(focal_gamma)} --k_sup {str(k_sup)}"
		)

		if checkpointModel:
			base_command += f" --checkpoint {checkpointModel}"
		if focal_alpha is not None:
			base_command += f" --focal_alpha {str(focal_alpha)}"
		if train_inference_dropout_enabled:
			base_command += f" --train_inference_dropout_enabled"
		if train_inference_transforms_enabled:
			base_command += f" --train_inference_transforms_enabled"
		if resolution == "5x":
			trainCommand = (
				f"--train_lib {os.path.join(outputPath, 'trainData.pt')} "
				f"--val_lib {os.path.join(outputPath, 'valData.pt')} "
			)
			testCommand = f"{base_command} {trainCommand}"
		elif resolution == "20x":
			trainCommand = (
				f"--train_lib {os.path.join(outputPath, 'training_20x_m' + str(currentModel+1), 'trainData20x.pt')} "
				f"--val_lib {os.path.join(outputPath, 'training_20x_m' + str(currentModel+1), 'valData20x.pt')} "
			)
			testCommand = f"{base_command} {trainCommand}"
		else:
			print("Resolution " + resolution + " is not currently supported.")
			sys.exit()
		try:
			print(testCommand)
			os.system(testCommand)
		except Exception as e:
			print(e)
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
		prefix = f"checkpoint_best_{resolution}_epoch_"
		all_files = os.listdir(modelPath)

		existingCheckpointModels = [f for f in all_files if f.startswith(prefix) and f.endswith(".pth")]
		existingCheckpointModelNumbers = []
		for f in existingCheckpointModels:
			try:
				# "checkpoint_best_5x_epoch_10.pth" -> "10.pth" -> "10"
				epoch_str = f.split(prefix)[1].split(".")[0]
				existingCheckpointModelNumbers.append(int(epoch_str))
			except Exception:
				print(f"[Warning] Skipping file, could not parse epoch number: {f}")

		if not existingCheckpointModelNumbers:
			print(f"[ERROR] No valid checkpoint models found in {modelPath} matching prefix '{prefix}'.")
			print("Please check your training logs. Skipping feature generation for this model.")
			# Continue to the next model in the loop
			continue

		if bestModels[l] != None:
			# Assumes bestModels[currentModel] holds the epoch number
			best_epoch_num = bestModels[currentModel]
			bestModel_filename = f"{prefix}{best_epoch_num}.pth"
			bestModel = os.path.join(modelPath, bestModel_filename)
			if not os.path.exists(bestModel):
				print(f"[Warning] Specified best model epoch {best_epoch_num} not found.")
				print("Falling back to highest epoch number.")
				best_epoch_num = max(existingCheckpointModelNumbers)
				bestModel_filename = f"{prefix}{best_epoch_num}.pth"
				bestModel = os.path.join(modelPath, bestModel_filename)
		else:
			best_epoch_num = max(existingCheckpointModelNumbers)
			bestModel_filename = f"{prefix}{best_epoch_num}.pth"
			bestModel = os.path.join(modelPath, bestModel_filename)

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
	predictions5x = p2d.read_csv(predictionsPath, header=0, index_col=0)
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
	samples = list(set.intersection(set(mat5a.index), set(mat20x.index)))
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
	Function edited from (https_://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).

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


def safe_open(path): #
		return Image.open(path)
class MILdataset(data.Dataset):
	'''
    Class edited used from (https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019).
    Instantiates a MIL dataset.
    '''

	def __init__(self, libraryfile, transform=None):
		'''
		 Initializes all class atributes and filters out missing images.
		 '''
		lib = torch.load(libraryfile, map_location='cpu')

		grid = []
		slideIDX = []
		invalid_count = 0  # Counter for missing files

		if 'tiles' not in lib:
			print(f"[FATAL ERROR in MILdataset] Your .pt file '{libraryfile}' is missing the required 'tiles' key.")
			self.grid = []
			self.slideIDX = []
			self.targets = []
			self.subtype = []
			self.slidenames = []
		else:
			print(f"[INFO MILdataset] Validating tile paths in '{libraryfile}'...")

			# Iterate through each slide's tiles
			for i, slide_tiles in enumerate(lib['tiles']):
				valid_tiles_for_this_slide = []
				for tile_path in slide_tiles:
					# Check if the file actually exists on the system
					if os.path.exists(tile_path):
						valid_tiles_for_this_slide.append(tile_path)
					else:
						invalid_count += 1

				# Only add valid tiles to the global grid
				grid.extend(valid_tiles_for_this_slide)
				slideIDX.extend([i] * len(valid_tiles_for_this_slide))

			self.slidenames = lib['slides']
			self.targets = lib['targets']
			self.subtype = lib["subtype"]
			self.grid = grid
			self.slideIDX = slideIDX

			print(f"--- Dataset Loading Summary ---")
			print(f"[INFO] Total valid tiles loaded: {len(self.grid)}")
			print(f"[INFO] Total invalid/missing tiles skipped: {invalid_count}")
			print(f"[INFO] From {len(self.slidenames)} slides.")
			print(f"-------------------------------")

		self.transform = transform
		self.mode = 1
		self.t_data = []

		self.epoch_tile_info = None
		self.epoch_slide_id_map = None
		self.epoch_target_map = None
		self.epoch_subtype_map = None


	def modelState(self, mode):
		'''Changes the current mode either to inference or training'''
		self.mode = mode
		if mode == 2:
			self.epoch_tile_info = None
	def setTransforms(self, transforms):
		self.transform = transforms

	def preselect_epoch_slides(self, sampling_mode='dampened_combined'):
		unique_original_slide_ids = list(range(len(self.slidenames)))
		num_slides_to_sample = len(unique_original_slide_ids)

		shuffled_original_slide_ids = []
		weights = None
		counts_to_log = None

		try:


			if 'subtype' in sampling_mode:
				all_subtypes = self.subtype
				subtype_counts = Counter(all_subtypes)
				counts_to_log = subtype_counts
				if sampling_mode == 'dampened_subtype':
					weights = [(1.0 / subtype_counts[s])**0.5 for s in all_subtypes]
				elif sampling_mode == 'balanced_subtype':
					weights = [1.0 / subtype_counts[s] for s in all_subtypes]
				shuffled_original_slide_ids = random.choices(
					population=unique_original_slide_ids,
					weights=weights, #
					k=num_slides_to_sample
				)

			elif 'target' in sampling_mode:
				all_targets_hard = [1 if t[1] >= 0.5 else 0 for t in self.targets]
				target_counts = Counter(all_targets_hard)
				counts_to_log = target_counts
				if sampling_mode == 'dampened_target':
					weights = [(1.0 / target_counts[t])**0.5 for t in all_targets_hard]
				elif sampling_mode == 'balanced_target':
					weights = [1.0 / target_counts[t] for t in all_targets_hard]
				shuffled_original_slide_ids = random.choices(
					population=unique_original_slide_ids,
					weights=weights, #
					k=num_slides_to_sample
				)

			elif 'combined' in sampling_mode:
				all_targets_hard = [1 if t[1] >= 0.5 else 0 for t in self.targets]
				all_subtypes = self.subtype
				all_combined_keys = [f"{s}_{t}" for s, t in zip(all_subtypes, all_targets_hard)]
				combined_counts = Counter(all_combined_keys)
				counts_to_log = combined_counts
				if sampling_mode == 'dampened_combined':
					weights = [(1.0 / combined_counts[key])**0.5 for key in all_combined_keys]
				elif sampling_mode == 'balanced_combined':
					weights = [1.0 / combined_counts[key] for key in all_combined_keys]
				shuffled_original_slide_ids = random.choices(
					population=unique_original_slide_ids,
					weights=weights, #
					k=num_slides_to_sample
				)
			elif "none" in sampling_mode:
				unique_original_slide_ids = list(range(len(self.slidenames)))
				num_slides_to_sample = len(unique_original_slide_ids)
				shuffled_original_slide_ids = unique_original_slide_ids.copy()
				# random.shuffle(shuffled_original_slide_ids)


			if counts_to_log:
				print(f"[INFO] Sampling from base counts: {counts_to_log}")
			sampled_subtypes = [self.subtype[i] for i in shuffled_original_slide_ids]
			print(f"[INFO] Post-sampling subtype distribution: {Counter(sampled_subtypes)}")
			sampled_targets = [1 if self.targets[i][1] >= 0.5 else 0 for i in shuffled_original_slide_ids]
			print(f"[INFO] Post-sampling target distribution: {Counter(sampled_targets)}")


		except Exception as e:
			print(f"[ERROR] preselect_epoch_slides: Error during weighting: {e}. Falling back to uniform sampling.")
			# shuffled_original_slide_ids = random.choices(
			# 	population=unique_original_slide_ids,
			# 	k=num_slides_to_sample
			# )

		selected_slide_set = set(shuffled_original_slide_ids)
		print(f"[INFO] Pre-selection resulted in {len(selected_slide_set)} unique slides for this epoch's inference.")

		self.epoch_tile_info = []
		self.epoch_slide_id_map = {}
		self.epoch_target_map = {}
		self.epoch_subtype_map = {}
		new_slide_idx_counter = 0

		for i in range(len(self.grid)):
			original_slide_id = self.slideIDX[i]

			if original_slide_id in selected_slide_set:
				if original_slide_id not in self.epoch_slide_id_map:
					new_id = new_slide_idx_counter
					self.epoch_slide_id_map[original_slide_id] = new_id
					self.epoch_target_map[new_id] = self.targets[original_slide_id]
					self.epoch_subtype_map[new_id] = self.subtype[original_slide_id]
					new_slide_idx_counter += 1

				new_epoch_slide_id = self.epoch_slide_id_map[original_slide_id]
				self.epoch_tile_info.append((i, new_epoch_slide_id))

		print(f"[INFO] Created new inference set with {len(self.epoch_tile_info)} tiles from {len(self.epoch_slide_id_map)} unique slides.")
	def maket_data(self, all_tile_probs, k):
		slide_to_all_tiles = defaultdict(list)
		if len(self.epoch_tile_info) != len(all_tile_probs):
			print(f"[ERROR] maket_data: Mismatch! Have {len(self.epoch_tile_info)} tiles in epoch info but {len(all_tile_probs)} probs. Aborting.")
			self.t_data = []
			return
		for i in range(len(all_tile_probs)):
			original_grid_index, new_epoch_slide_id = self.epoch_tile_info[i]

			slide_id = new_epoch_slide_id # Use the new ID

			tile_data = (
				self.grid[original_grid_index],   # 0: tile_path
				self.epoch_target_map[slide_id],  # 1: target
				all_tile_probs[i],                # 2: prob
				self.epoch_subtype_map[slide_id]  # 3: subtype
			)
			slide_to_all_tiles[slide_id].append(tile_data)

		unique_slide_ids = list(slide_to_all_tiles.keys())
		shuffled_slide_ids = unique_slide_ids
		random.shuffle(shuffled_slide_ids)
		self.t_data =[]
		for slide_id in shuffled_slide_ids:
			all_tiles_for_this_slide = slide_to_all_tiles[slide_id]
			sorted_tiles = sorted(all_tiles_for_this_slide, key=lambda t: t[2], reverse=True)
			num_to_sample = min(k, len(sorted_tiles))
			if num_to_sample ==0:
				continue
			top_k_tiles = sorted_tiles[:num_to_sample]
			for tile_to_add in top_k_tiles:
				self.t_data.append(
					(
						slide_id,       # slide_id (this is the new epoch ID)
						tile_to_add[0], # tile_path
						tile_to_add[1]  # target
					)

				)

	def maketraindata(self, idxs):
		slide_to_tiles = defaultdict(list)
		for x in idxs:
			if x >= len(self.slideIDX):
				print(f"[WARN] maketraindata: Index {x} out of bounds. Skipping.")
				continue

			slide_id = self.slideIDX[x]
			tile_data = (slide_id, self.grid[x], self.targets[slide_id])
			slide_id = self.slideIDX[x]
			slide_to_tiles[slide_id].append(tile_data)
		unique_slide_ids = list(slide_to_tiles.keys())

		if not unique_slide_ids:
			print("[ERROR] maketraindata: No slide IDs found. t_data will be empty.")
			self.t_data = []
			return

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

		except Exception as e:
			shuffled_slide_ids = unique_slide_ids
			random.shuffle(shuffled_slide_ids)

		self.t_data = []
		for slide_id in shuffled_slide_ids:
			self.t_data.extend(slide_to_tiles[slide_id])

		sampled_subtypes = [self.subtype[sid] for sid in shuffled_slide_ids]

	def __getitem__(self, index):
		'''
        Accesses a tile based upon the preset mode (inference or training)
        '''
		if self.mode == 1:
			# If we are in a pre-selected epoch, use the epoch data
			if self.epoch_tile_info is not None:
				original_grid_index, new_epoch_slide_id = self.epoch_tile_info[index]
				slideIDX = new_epoch_slide_id
				target = self.epoch_target_map[new_epoch_slide_id]
				tile_path = self.grid[original_grid_index]
			else:
				slideIDX = self.slideIDX[index]
				target = self.targets[slideIDX]
				tile_path = self.grid[index]

			img = safe_open(tile_path)

		elif self.mode == 2:
			slideIDX, tile_path, target = self.t_data[index] # target is already the tensor
			img = safe_open(tile_path)

		else:
			raise IndexError(f"Dataset mode not set to 1 or 2. Current mode: {self.mode}")

		if self.transform is not None:
			img = self.transform(img)

		return (img, target, slideIDX)

	def __len__(self):
		'''
        Returns the length of the given dataset, whether it's the training or inference sets
        '''
		if self.mode == 1:
			if self.epoch_tile_info is not None:
				return len(self.epoch_tile_info)
			else:
				return len(self.grid)
		elif self.mode == 2:
			return len(self.t_data)
		else:
			return 0