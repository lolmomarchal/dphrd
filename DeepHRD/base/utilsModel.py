#!/usr/bin/env python3

# Author: Erik N. Bergstrom

# Contact: ebergstr@eng.ucsd.edu

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
from collections import defaultdict, Counter  # Import Counter
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image  #


def runMultiGpuTraining(i, iModels, pythonVersion, outputPath, batch_size, dropoutRate, resolution, workers, epochs,
                        checkpointModel=None, validation_interval=1, k=100, weights=0.5, patience=40,
                        sampling_mode='dampened_combined', lambda_sup=0.3, loss_fn='ce', focal_gamma=2.0,
                        focal_alpha=None, k_sup=10, train_inference_dropout_enabled=False,
                        train_inference_transforms_enabled=False):
    for currentModel in iModels:

        # Base command with all common parameters
        base_command = (
            f"{pythonVersion} base/train_mp2.py "
            f"--output {os.path.join(outputPath, 'training_m' + str(currentModel + 1))} "
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
                f"--train_lib {os.path.join(outputPath, 'training_20x_m' + str(currentModel + 1), 'trainData20x.pt')} "
                f"--val_lib {os.path.join(outputPath, 'training_20x_m' + str(currentModel + 1), 'valData20x.pt')} "
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


def runMultiGpuInference(i, iModels, pythonVersion, outputPath, modelPath, batch_size, dropoutRate, resolution, workers,
                         BN_reps):
    print("IMODELS")
    print(iModels)
    for currentModel in iModels:
        print(f"currentModel:{currentModel}")
        if resolution == '5x':
            # Non-dropout inference for extracting features of each tile.
            testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath,
                                                                                      "testData.pt") + " --output " + os.path.join(
                outputPath, "m" + str(currentModel + 1)) + " --model " + os.path.join(modelPath,
                                                                                      resolution + "_m" + str(
                                                                                          currentModel + 1) + ".pth") + " --batch_size " + str(
                batch_size) + " --BN_reps 1 --gpu " + str(
                i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers)
            testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel + 1),
                                                "feature_vectors.tsv") + " " + os.path.join(outputPath,
                                                                                            "m" + str(currentModel + 1),
                                                                                            "feature_vectors_test_" + resolution + ".tsv")

            # time.sleep(random.randrange(0, 4))
            os.system(testCommand)
            os.system(testCommand2)

            # Additional inference for all BN-reps with the specified dropout rate (default 0.2).
            if dropoutRate > 0:
                testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath,
                                                                                          "testData.pt") + " --output " + os.path.join(
                    outputPath, "m" + str(currentModel + 1)) + " --model " + os.path.join(modelPath,
                                                                                          resolution + "_m" + str(
                                                                                              currentModel + 1) + ".pth") + " --batch_size " + str(
                    batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(
                    dropoutRate) + " --resolution " + resolution + " --workers " + str(workers)
                testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel + 1),
                                                    "predictions.csv") + " " + os.path.join(outputPath,
                                                                                            "m" + str(currentModel + 1),
                                                                                            "predictions_" + resolution + ".csv")
                os.system(testCommand)
                os.system(testCommand3)
                torch.cuda.empty_cache()

        else:
            # Non-dropout inference for extracting features of each tile.
            testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i + 1),
                                                                                      "ROI",
                                                                                      "testData20x.pt") + " --output " + os.path.join(
                outputPath, "m" + str(currentModel + 1)) + " --model " + os.path.join(modelPath,
                                                                                      resolution + "_m" + str(
                                                                                          currentModel + 1) + ".pth") + " --batch_size " + str(
                batch_size) + " --BN_reps 1 --gpu " + str(
                i) + " --dropoutRate 0.0 --resolution " + resolution + " --workers " + str(workers)
            testCommand2 = "mv " + os.path.join(outputPath, "m" + str(currentModel + 1),
                                                "feature_vectors.tsv") + " " + os.path.join(outputPath,
                                                                                            "m" + str(currentModel + 1),
                                                                                            "feature_vectors_test_" + resolution + ".tsv")
            # time.sleep(random.randrange(0, 4))
            os.system(testCommand)
            os.system(testCommand2)

            # Additional inference for all BN-reps with the specified dropout rate (default 0.2).
            if dropoutRate > 0:
                testCommand = pythonVersion + " base/test_final.py --lib " + os.path.join(outputPath, "m" + str(i + 1),
                                                                                          "ROI",
                                                                                          "testData20x.pt") + " --output " + os.path.join(
                    outputPath, "m" + str(currentModel + 1)) + " --model " + os.path.join(modelPath,
                                                                                          resolution + "_m" + str(
                                                                                              i + 1) + ".pth") + " --batch_size " + str(
                    batch_size) + " --BN_reps " + str(BN_reps) + " --gpu " + str(i) + " --dropoutRate " + str(
                    dropoutRate) + " --resolution " + resolution + " --workers " + str(workers)
                testCommand3 = "mv " + os.path.join(outputPath, "m" + str(currentModel + 1),
                                                    "predictions.csv") + " " + os.path.join(outputPath,
                                                                                            "m" + str(currentModel + 1),
                                                                                            "predictions_" + resolution + ".csv")
                os.system(testCommand)
                os.system(testCommand3)
                torch.cuda.empty_cache()


def generateFeatureVectorsUsingBestModels(i, iModels, project, projectPath, pythonVersion, outputPath, batch_size,
                                          dropoutRate, resolution, bestModels, checkpointModel=None):
    for l, currentModel in enumerate(iModels):
        modelPath = os.path.join(outputPath, "training_m" + str(currentModel + 1))
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
        if not os.path.exists(
                os.path.join(outputPath, "training_m" + str(currentModel + 1), "feature_vectors_train.tsv")):
            testCommand = pythonVersion + " base/test_final2.py --lib " + os.path.join(outputPath,
                                                                                      "trainData.pt") + " --output " + os.path.join(
                outputPath, "training_m" + str(currentModel + 1)) + " --model " + bestModel + " --batch_size " + str(
                batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
            testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(
                currentModel + 1), "predictions_train.csv")
            testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "feature_vectors.tsv") + " " + os.path.join(outputPath,
                                                                                            "training_m" + str(
                                                                                                currentModel + 1),
                                                                                       "feature_vectors_train.tsv")
            print(testCommand)
            os.system(testCommand)
            os.system(testCommand2)
            os.system(testCommand3)
        if not os.path.exists(
                os.path.join(outputPath, "training_m" + str(currentModel + 1), "feature_vectors_val.tsv")):
            testCommand = pythonVersion + " base/test_final2.py --lib " + os.path.join(outputPath,
                                                                                      "valData.pt") + " --output " + os.path.join(
                outputPath, "training_m" + str(currentModel + 1)) + " --model " + bestModel + " --batch_size " + str(
                batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
            testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(
                currentModel + 1), "predictions_val.csv")
            testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "feature_vectors.tsv") + " " + os.path.join(outputPath,
                                                                                            "training_m" + str(
                                                                                                currentModel + 1),
                                                                                            "feature_vectors_val.tsv")
            print(testCommand)
            os.system(testCommand)
            os.system(testCommand2)
            os.system(testCommand3)
        if not os.path.exists(
                os.path.join(outputPath, "training_m" + str(currentModel + 1), "feature_vectors_test.tsv")):
            testCommand = pythonVersion + " base/test_final2.py --lib " + os.path.join(outputPath,
                                                                                      "testData.pt") + " --output " + os.path.join(
                outputPath, "training_m" + str(currentModel + 1)) + " --model " + bestModel + " --batch_size " + str(
                batch_size) + " --BN_reps 1 --gpu " + str(i) + " --dropoutRate 0.0 --resolution " + resolution
            testCommand2 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "predictions.csv") + " " + os.path.join(outputPath, "training_m" + str(
                currentModel + 1), "predictions_test.csv")
            testCommand3 = "mv " + os.path.join(outputPath, "training_m" + str(currentModel + 1),
                                                "feature_vectors.tsv") + " " + os.path.join(outputPath,
                                                                                            "training_m" + str(
                                                                                                currentModel + 1),
                                                                                            "feature_vectors_test.tsv")
            os.system(testCommand)
            os.system(testCommand2)
            os.system(testCommand3)
        torch.cuda.empty_cache()


def runMultiGpuROIs(i, iModels, project, projectPath, pythonVersion, outputPath, maxROI, max_cpu, stain_norm,
                    removeBlurry, predict=False):
    for currentModel in iModels:
        if predict:

            roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " + os.path.join(
                outputPath, "m" + str(currentModel + 1), "ROI") + " --objectiveFile " + \
                         os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath,
                                                                                                         project) + " --tileConv " + \
                         os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(
                outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath,
                                                                                       "m" + str(currentModel + 1),
                                                                                       "feature_vectors_test_5x.tsv") + \
                         " --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu) + " --predict"


        else:
            roiCommand = pythonVersion + " base/pullROIs.py --project " + project + " --projectPath " + outputPath + " --output " + os.path.join(
                outputPath, "training_20x_m" + str(currentModel + 1)) + " --objectiveFile " + \
                         os.path.join(projectPath, "objectiveInfo.txt") + " --slidePath " + os.path.join(projectPath,
                                                                                                         project) + " --tileConv " + \
                         os.path.join(projectPath, "slideNumberToSampleName.txt") + " --test_lib " + os.path.join(
                outputPath, "testData.pt") + " --feature_vectors_test " + os.path.join(outputPath, "training_m" + str(
                currentModel + 1), "feature_vectors_test.tsv") + \
                         " --train_lib " + os.path.join(outputPath,
                                                        "trainData.pt") + " --feature_vectors_train " + os.path.join(
                outputPath, "training_m" + str(currentModel + 1), "feature_vectors_train.tsv") + \
                         " --val_lib " + os.path.join(outputPath,
                                                      "valData.pt") + " --feature_vectors_val " + os.path.join(
                outputPath, "training_m" + str(currentModel + 1), "feature_vectors_val.tsv") + \
                         " --maxROI " + str(maxROI) + " --max_cpu " + str(max_cpu)
        if removeBlurry:
            roiCommand += " --removeBlurry"
        if stain_norm:
            roiCommand += " --stain_norm"

        print(roiCommand)
        os.system(roiCommand)
        torch.cuda.empty_cache()


def selectBestModel(predictionsPath):
    predictions5x = p2d.read_csv(predictionsPath, header=0, index_col=0)
    avgPredictions = predictions5x.loc[:, predictions5x.columns.str.endswith("AverageProb")]
    bestModels = pd.DataFrame(index=predictions5x.index, columns=avgPredictions.columns)

    for sample in predictions5x.index:
        bestModels.loc[sample] = abs(
            avgPredictions.loc[sample] - float(predictions5x.loc[sample, 'Ensemble-Probability']))

    bestModel = bestModels.mean(axis=0).astype(float).idxmin().split("-")[0]

    return (bestModel)


def collectSampleIndex(file, tileConversionMatrix):
    sample = file.split("/")[-1].split(".")[0]
    sampleIndex = int(tileConversionMatrix.loc[sample].iloc[0])
    if sampleIndex < 10:
        sampleIndex = "00" + str(sampleIndex)
    elif 100 > sampleIndex >= 10:
        sampleIndex = "0" + str(sampleIndex)
    else:
        sampleIndex = str(sampleIndex)
    return (sampleIndex)


def z_test(x, mu, sigma):
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
    z = (x - mu) / sigma
    p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    return (z, p)


def multiResolution(outputPath, nModels, dropoutRate, threshold):
    mat5x = pd.read_csv(
        os.path.join(outputPath, "predictions_5x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"),
        index_col=0, header=0)
    mat20x = pd.read_csv(
        os.path.join(outputPath, "predictions_20x_n" + str(nModels) + "_models_" + str(dropoutRate) + ".csv"),
        index_col=0, header=0)
    # samples = list(set(mat5x.index & mat20x.index))
    samples = list(set.intersection(set(mat5a.index), set(mat20x.index)))
    mat5x = mat5x.loc[samples]
    mat20x = mat20x.loc[samples]

    finalMat = pd.DataFrame(
        columns=["file", "target", "HRD-prediction", "Multi-Res-prediction", "LowerCI", "UpperCI", "p-value"])
    finalMat['file'] = samples
    finalMat = finalMat.set_index('file')
    finalMat['target'] = list(mat20x['target'])
    finalMat['Multi-Res-prediction'] = list((mat5x['Ensemble-Probability'] + mat20x['Ensemble-Probability']) / 2)
    finalMat['HRD-prediction'] = (finalMat['Multi-Res-prediction'] > threshold).astype(int)

    ensemblePredictions = pd.concat(
        [mat5x.loc[:, mat5x.columns.str.endswith("Prob")], mat20x.loc[:, mat20x.columns.str.endswith("Prob")]], axis=1)

    finalMat['LowerCI'] = list(ensemblePredictions.quantile(0.025, axis=1))
    finalMat['UpperCI'] = list(ensemblePredictions.quantile(0.975, axis=1))
    for sample in finalMat.index:
        predictions = ensemblePredictions.loc[sample]
        z, p = z_test(threshold, np.mean(predictions), np.std(predictions))
        finalMat.loc[sample, 'p-value'] = p

    finalMat.to_csv(
        os.path.join(outputPath, "DeepHRD_report_5x_20x_n" + str(nModels) + "_dropout" + str(dropoutRate) + ".csv"))


def combinePredictions(resolution, outputPath, nModels, dropoutRate):
    stdev = 1
    # bestModel = None
    for i in range(nModels):
        # modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions_" + resolution + "_m" + str(i+1) + "_" + str(dropoutRate) + "_temp.csv")
        # modelPath = os.path.join(outputPath, "m" + str(i+1), "predictions.csv")
        modelPath = os.path.join(outputPath, "m" + str(i + 1), "predictions_" + resolution + ".csv")
        newPredictions = pd.read_csv(modelPath, header=0, index_col=0)
        if i == 0:
            finalPredictions = newPredictions
            finalPredictions['Ensemble-LowerCI'] = 0
            finalPredictions['Ensemble-UpperCI'] = 0
            finalPredictions = finalPredictions.rename(columns={'probability': 'Ensemble-Probability'})

        else:
            finalPredictions['Ensemble-Probability'] += newPredictions['probability']
            finalPredictions = pd.concat(
                [finalPredictions, newPredictions.loc[:, newPredictions.columns.str.startswith("BN_rep")]], axis=1)
        finalPredictions["m" + str(i + 1) + "-AverageProb"] = newPredictions['probability']
        finalPredictions["m" + str(i + 1) + "-LowerCI"] = newPredictions.loc[:,
                                                          newPredictions.columns.str.startswith("BN_rep")].quantile(
            0.025, axis=1)
        finalPredictions["m" + str(i + 1) + "-UpperCI"] = newPredictions.loc[:,
                                                          newPredictions.columns.str.startswith("BN_rep")].quantile(
            0.975, axis=1)

    finalPredictions['Ensemble-Probability'] = finalPredictions['Ensemble-Probability'] / nModels
    finalPredictions['Ensemble-LowerCI'] = finalPredictions.loc[:,
                                           finalPredictions.columns.str.startswith("BN_rep")].quantile(0.025, axis=1)
    finalPredictions['Ensemble-UpperCI'] = finalPredictions.loc[:,
                                           finalPredictions.columns.str.startswith("BN_rep")].quantile(0.975, axis=1)
    finalPredictions = finalPredictions.loc[:, ~finalPredictions.columns.str.startswith("BN_rep")]
    finalPredictions.to_csv(os.path.join(outputPath,
                                         "predictions_" + resolution + "_n" + str(nModels) + "_models_" + str(
                                             dropoutRate) + ".csv"))


def groupTopKtilesTesting(groups, data, k):
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


def groupTopKtiles(groups, data, k=1):
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

def groupTopKtilesAverage(groups, data, nmax, percentile=0.05, min_k=5, max_k=15):
    """
    Calculates an adaptive Top-K mean.
    Ideal for 5x where tile area is large.
    """
    out = np.empty(nmax)
    out[:] = np.nan

    unique_groups = np.unique(groups)

    for g in unique_groups:
        # Get all probabilities for this specific slide
        slide_probs = data[groups == g]
        n_tiles = len(slide_probs)

        k = int(n_tiles * percentile)

        k = max(min_k, min(k, max_k))

        k = min(k, n_tiles)

        top_k = np.sort(slide_probs)[-k:]
        out[g] = np.mean(top_k)

    return out
def groupTopKtilesProbabilities(groups, data, nmax):
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


def calculateError(pred, real):
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


import PIL
from sklearn.cluster import MiniBatchKMeans, KMeans


def safe_open(path):
    try:
        if not os.path.exists(path):
            return None
        # We add .convert('RGB') to ensure we don't have grayscale/alpha issues later
        return Image.open(path).convert('RGB')
    except (PIL.UnidentifiedImageError, IOError, OSError) as e:
        print(f"[Warning] Skipping corrupted image: {path} | Error: {e}")
        return None


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
            self.softLabels = []
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
                if len(valid_tiles_for_this_slide) < 10:
                    print(f"[INFO] Skipping slide {lib['slides'][i]} (only {len(valid_tiles_for_this_slide)} valid tiles)")
                    continue

                # Only add valid tiles to the global grid
                grid.extend(valid_tiles_for_this_slide)
                slideIDX.extend([i] * len(valid_tiles_for_this_slide))

            self.slidenames = lib['slides']
            raw_scores = np.array(lib['targets'])

            self.targets = (raw_scores / 100.0).tolist()
            # print(self.targets)
            self.subtype = lib["subtype"]
            self.softLabels = lib["softLabels"]
            self.grid = grid
            self.slideIDX = slideIDX
            #
            # print(f"--- Dataset Loading Summary ---")
            # print(f"[INFO] Total valid tiles loaded: {len(self.grid)}")
            # print(f"[INFO] Total invalid/missing tiles skipped: {invalid_count}")
            # print(f"[INFO] From {len(self.slidenames)} slides.")
            # print(f"-------------------------------")

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
    def preselect_epoch_slides(self, sampling_mode='balanced_combined'):
        all_targets_hard = [1 if t[1] >= 0.5 else 0 for t in self.softLabels]
        buckets = defaultdict(list)
        for i in range(len(self.slidenames)):
            if 'combined' in sampling_mode:
                key = f"{self.subtype[i]}_{all_targets_hard[i]}"
            elif 'subtype' in sampling_mode:
                key = f"{self.subtype[i]}"
            elif 'target' in sampling_mode:
                key = f"{all_targets_hard[i]}"
            else:
                key = "all"
            buckets[key].append(i)
        print("\n" + "="*30)
        print(f"[INFO] ORIGINAL DATA DISTRIBUTION:")
        for b_key, b_indices in buckets.items():
            print(f"  - {b_key}: {len(b_indices)} slides")
        print("="*30)
        # 3. Determine sampling depth with Soft-Cap Lenience
        counts = [len(v) for v in buckets.values()]
        if not counts: return
        median_size = int(np.median(counts)) # Around 40 in your case

        shuffled_original_slide_ids = []
        for key, indices in buckets.items():
            n_available = len(indices)

            if n_available >= median_size:
                # SOFT-CAP LENIENCE for Majority Classes:
                # Instead of a hard cap at 40, we take: median + 30% of the excess.
                # Example (175 slides): 40 + (135 * 0.30) = ~80 slides.
                # This retains 45% of the unique slides rather than only 22%.
                excess = n_available - median_size
                num_to_sample = median_size + int(excess * 0.30)
            else:
                # DAMPENED BOOST for Minority Classes:
                # We bring them halfway to the median to ensure signal.
                # Example (5 slides): 5 + (35 * 0.5) = ~22 instances.
                gap = median_size - n_available
                num_to_sample = n_available + int(gap * 0.5)

            # 4. Standard Sampling Execution
            if n_available >= num_to_sample:
                # Sub-sample without replacement (all unique slides)
                shuffled_original_slide_ids.extend(random.sample(indices, num_to_sample))
            else:
                # Take all unique + duplicate some for the boost
                shuffled_original_slide_ids.extend(indices)
                remaining = num_to_sample - n_available
                if remaining > 0:
                    shuffled_original_slide_ids.extend(random.choices(indices, k=remaining))

        random.shuffle(shuffled_original_slide_ids)
        # PRINT NEW EPOCH DISTRIBUTION
        new_strata = []
        for i in shuffled_original_slide_ids:
            new_strata.append(f"{self.subtype[i]}_Target{all_targets_hard[i]}")

        epoch_counts = Counter(new_strata)
        print(f"[INFO] NEW EPOCH DISTRIBUTION (by Stratum):")
        for b_key in sorted(epoch_counts.keys()):
            print(f"  - {b_key:25} : {epoch_counts[b_key]:4} instances")

        # Also print a high-level Target balance
        new_targets = [all_targets_hard[i] for i in shuffled_original_slide_ids]
        print(f"[INFO] NEW EPOCH TARGET TOTALS: {Counter(new_targets)}")
        print("="*50 + "\n")


        self.epoch_tile_info = []
        self.epoch_slide_id_map = {}
        self.epoch_target_map = {}
        self.epoch_subtype_map = {}
        self.epoch_softlabel_map = {}

        original_slide_to_tiles = defaultdict(list)
        for i, orig_id in enumerate(self.slideIDX):
            original_slide_to_tiles[orig_id].append(i)
        for new_instance_id, original_slide_id in enumerate(shuffled_original_slide_ids):
            self.epoch_slide_id_map[new_instance_id] = original_slide_id
            self.epoch_target_map[new_instance_id] = self.targets[original_slide_id]
            self.epoch_subtype_map[new_instance_id] = self.subtype[original_slide_id]
            self.epoch_softlabel_map[new_instance_id] = self.softLabels[original_slide_id]

            tile_indices = original_slide_to_tiles[original_slide_id]
            for t_idx in tile_indices:
                self.epoch_tile_info.append((t_idx, new_instance_id))

        print(f"[INFO] Created epoch with {len(shuffled_original_slide_ids)} slide instances.")
    def maket_data(self, all_tile_probs, percentile=0.20, min_k=5, max_k=15, pool_factor=2.0):
        """
        pool_factor: controls how much larger the candidate pool is than k_target.
        - 2.0 (Default): Very stochastic, good for earlier stages.
        - 1.2: Very focused, good for final unfreezing stages.
        """
        instance_to_tiles = defaultdict(list)
        for i in range(len(self.epoch_tile_info)):
            grid_idx, inst_id = self.epoch_tile_info[i]
            instance_to_tiles[inst_id].append({'grid_idx': grid_idx, 'prob': all_tile_probs[i]})

        self.t_data = []
        for inst_id, tiles in instance_to_tiles.items():
            sorted_tiles = sorted(tiles, key=lambda x: x['prob'], reverse=True)
            num_avail = len(sorted_tiles)

            # 1. Target number of tiles (5% of slide)
            k_target = max(min_k, min(int(num_avail * 0.05), max_k))
            k_target = min(k_target, num_avail)

            # 2. Candidate Pool size based on pool_factor
            # Earlier: pool_factor 2.0 -> sample k from top 2k tiles
            # Later: pool_factor 1.1 -> sample k from top 1.1k tiles
            pool_size = max(int(k_target * pool_factor), k_target)
            pool_size = min(pool_size, num_avail)

            candidate_pool = sorted_tiles[:pool_size]
            selected_tiles = random.sample(candidate_pool, k_target)

            target = self.epoch_target_map[inst_id]
            soft_label = self.epoch_softlabel_map[inst_id]

            for t in selected_tiles:
                self.t_data.append((inst_id, self.grid[t['grid_idx']], target, soft_label))

        random.shuffle(self.t_data)
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
        try:

            if self.mode == 1:
                if self.epoch_tile_info is not None:
                    original_grid_index, new_epoch_slide_id = self.epoch_tile_info[index]
                    slideIDX = new_epoch_slide_id
                    target = self.epoch_target_map[new_epoch_slide_id]
                    tile_path = self.grid[original_grid_index]
                    softLabel = self.epoch_softlabel_map[new_epoch_slide_id]

                else:
                    slideIDX = self.slideIDX[index]
                    target = self.targets[slideIDX]
                    tile_path = self.grid[index]
                    softLabel = self.softLabels[slideIDX]

                img = safe_open(tile_path)

            elif self.mode == 2:
                slideIDX, tile_path, target, softLabel = self.t_data[index]
                img = safe_open(tile_path)

            # make sure file is not corrupted -> from old bug
            if img is None:
                print(f"Error: {self.grid[index]}")
                return self.__getitem__((index + 1) % self.__len__())

            if self.transform is not None:
                img = self.transform(img)
            assert softLabel is not None
            assert torch.isfinite(softLabel).all()
            assert softLabel.sum().item() > 0


            return (img, target, softLabel, slideIDX)

        except Exception as e:
            # Fallback for any other unexpected errors
            print(f"[Runtime Error] Index {index} failed: {e}")
            return self.__getitem__((index + 1) % self.__len__())

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
    # In utilsModel.py -> inside MILdataset class
    def make_smart_warmup_data(self, all_tile_probs, epoch, explore_thresh=5, uncertain_thresh=15, percentile=0.15, min_k=5, max_k=50):
        """
        Curriculum sampling that scales based on training thresholds.
        """
        instance_to_tiles = defaultdict(list)
        for i in range(len(self.epoch_tile_info)):
            grid_idx, inst_id = self.epoch_tile_info[i]
            instance_to_tiles[inst_id].append({'grid_idx': grid_idx, 'prob': all_tile_probs[i]})

        self.t_data = []
        for inst_id, tiles in instance_to_tiles.items():
            target = self.epoch_target_map[inst_id]
            soft_label = self.epoch_softlabel_map[inst_id]
            sorted_tiles = sorted(tiles, key=lambda x: x['prob'])
            n_tiles = len(sorted_tiles)

            # Calculate k based on slide size
            k = max(min_k, min(int(n_tiles * percentile), max_k))
            k = min(k, n_tiles)

            # --- CURRICULUM PHASES ---
            if epoch < explore_thresh:
                # Phase 1: Pure Exploration
                selected = random.sample(range(n_tiles), k)

            elif epoch < uncertain_thresh:
                # Phase 2: Focus on the "Confused" middle (25th to 75th percentile)
                mid_start, mid_end = n_tiles // 4, 3 * n_tiles // 4
                available = list(range(mid_start, mid_end))
                if len(available) < k:
                    selected = random.sample(range(n_tiles), k)
                else:
                    selected = random.sample(available, k)

            else:
                # Phase 3: Transition (70% Top-K + 30% Random diversity)
                top_k_count = int(k * 0.8)
                random_k_count = k - top_k_count
                selected = list(range(n_tiles - top_k_count, n_tiles))
                remaining = list(range(n_tiles - top_k_count))
                if remaining:
                    selected.extend(random.sample(remaining, min(random_k_count, len(remaining))))

            for idx in selected:
                tile = sorted_tiles[idx]
                self.t_data.append((inst_id, self.grid[tile['grid_idx']], target, soft_label))

        random.shuffle(self.t_data)
    import random
    from collections import defaultdict
    from sklearn.cluster import MiniBatchKMeans
    # to start off -> force model to learn from diverse features (which includes making it learn from clusters rather than simply random)
    def make_clustered_warmup_data(self, all_tile_probs, all_features, percentile=0.1, n_clusters=8, min_k=1, max_k=20):
        """
        Dynamically selects the top percentage of tiles from each cluster based on probability.
        Ensures morphological diversity while keeping sampling representative of cluster size.
        """
        slide_to_data = defaultdict(lambda: {'indices': [], 'features': [], 'probs': []})
        for i in range(len(self.epoch_tile_info)):
            grid_idx, slide_id = self.epoch_tile_info[i]
            slide_to_data[slide_id]['indices'].append(grid_idx)
            slide_to_data[slide_id]['features'].append(all_features[i])
            slide_to_data[slide_id]['probs'].append(all_tile_probs[i])

        self.t_data = []
        for slide_id, data_dict in slide_to_data.items():
            feats = np.array(data_dict['features'])
            indices = np.array(data_dict['indices'])
            probs = np.array(data_dict['probs'])

            # Phenotype Clustering
            n_c = min(n_clusters, len(indices))
            kmeans = MiniBatchKMeans(n_clusters=n_c, n_init=1, batch_size=256)
            labels = kmeans.fit_predict(feats)

            target = self.epoch_target_map[slide_id]
            soft_label = self.epoch_softlabel_map[slide_id]
            # INSIDE make_clustered_warmup_data
            for c_id in range(n_c):
                cluster_mask = (labels == c_id)
                if not np.any(cluster_mask): continue

                c_indices = indices[cluster_mask]

                num_to_pick = max(min_k, min(int(len(c_indices) * percentile), max_k))
                selected_indices = np.random.choice(len(c_indices), num_to_pick, replace=False)
                selected_grid_idxs = c_indices[selected_indices]

                for g_idx in selected_grid_idxs:
                    self.t_data.append((slide_id, self.grid[g_idx], target, soft_label))

        print(f"[INFO] Adaptive Warmup: Created {len(self.t_data)} tiles using top {percentile*100}% per cluster.")