#!/usr/bin/env python3
 
#Author: Erik N. Bergstrom

#Contact: ebergstr@eng.ucsd.edu

# This code represents the testing script for the DeepHRD prediction package. The MIL backbone has been adpated from a previous implementation 
# developed by Campanella et al. 2019:
#		[3] Campanella, G., Hanna, M.G., Geneslaw, L. et al. Clinical-grade computational pathology using weakly supervised deep learning on 
#			whole slide images. Nat Med 25, 1301–1309 (2019). https://doi.org/10.1038/s41591-019-0508-1.



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import openslide
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import tqdm
import os
import sys
import argparse
import numpy as np
from model import ResNet_dropout as RNN
import utilsModel as ut




parser = argparse.ArgumentParser(description='Multi-Resolution biomarker classifier testing script')
parser.add_argument('--lib', type=str, default='', help='Path to the testing datastructure. See README for more details on formatting')
parser.add_argument('--output', type=str, default='.', help='Path to the output where the checkpoints and training files are saved')
parser.add_argument('--model', type=str, default='', help='Path to the pretrained model')
parser.add_argument('--batch_size', type=int, default=64, help='How many tiles to include for each mini-batch (default: 64)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--BN_reps', type=int, default=10, help='Number of MonteCarlo iterations to perform for bayesian network estimation (default is 10: sufficient for a dropout<0.2)')
parser.add_argument('--gpu', default=0, type=int, help='gpu device selection (default: 0)')
parser.add_argument('--dropoutRate', default=0.2, type=float, help='Rate of dropout to be used within the fully connected layers.')
parser.add_argument('--resolution', type=str, default='5x', help='Current magnification resolution')

def save_tile_probs(probs, slideIDX, reps, modelNumber):
	"""
    Save raw tile-level probabilities so aggregation can be rerun later.
    """
	out_file = os.path.join(
		args.output,
		f"tile_probs_{modelNumber}_BNrep{reps+1}.npz"
	)

	np.savez(
		out_file,
		probs=probs,                     # shape: [num_tiles]
		slideIDX=np.array(slideIDX),     # shape: [num_tiles]
		BN_rep=reps
	)

def main ():
	'''
	Multi-resolution testing main function to organize and execute all subsequent commands. This script will test a single model at a single
	resolution. This version of testing performs an iterative inference over each slide (up to BN_reps times) to estimate model uncertainty. 
	NOTE: The set dropout rate is set to a probability=0.2; however, this can be changed within the model_test.py script. Performing at least 10 iterations of
	inference is sufficient to estimate the model's uncertainty when using a dropout rate of less than 0.2. The number of iterations should
	be increased if using a higher dropout rate.

	Parameters:
		Passed via command line arguments (See command "python3 test_final.py -h" for more details)

	Returns:
		None

	Outputs:
		predictions.csv		The final predcitions of the model on each tissue slide. All prediction values across each BN_rep is included along with the final
							average prediction.

	'''
	global args, device
	args = parser.parse_args()
	print(args.lib)


	# Creates the output directory if it does not already exist
	if not os.path.exists(args.output):
		os.makedirs(args.output)

	# If a GPU is available, set the default device to "cuda", otherwise perform training on the "cpu" device. Assuming a GPU is available
	# move the current training to the desired GPU specified by the user (default is 0).
	gpu_available = torch.cuda.is_available()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if gpu_available:
		torch.cuda.set_device(args.gpu)


	# The number of inference iterations for the bayesian approximation
	t = args.BN_reps


	# Import the model from the model_test script. Instantiate model weights using the pretrained model and send to the 
	# relevant device (CPU or GPU)
	model = RNN(args.dropoutRate)
	model=model.to(device)

	if gpu_available:
		ch = torch.load(args.model)
	else:
		print(f"Model path:{args.model}")
		ch = torch.load(args.model, map_location=torch.device('cpu'))
	state_dict = ch['state_dict']


	new_state_dict = {}

	state_dict = ch["state_dict"]
	new_state_dict = {}
	# adjust for some of the possible (older versions)
	for k, v in state_dict.items():
		if k.startswith("classifier."):
			k = k.replace("classifier.", "resnet.fc.")
		elif k.startswith("head."):
			k = k.replace("head.", "resnet.fc.")
		new_state_dict[k] = v

	model.load_state_dict(new_state_dict, strict= False)
	cudnn.benchmark = True

	normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) # fixed transforms to resnet ones :3
	trans = transforms.Compose([transforms.ToTensor(),normalize])

	dset = ut.MILdataset(args.lib, trans)
	loader = torch.utils.data.DataLoader(
		dset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False)

	# Set the mode to inference
	dset.modelState(1)

	# Performs inference across the entire dataset for t number of BN_reps. Each rep will randomly dropout a new set of nodes withint
	# the fully connected layer of the model.
	modelNumber = args.model.split("/")[-1].split(".")[0]
	allProbs = []
	for reps in range(t):
		probs, features = inference(modelNumber, reps, t, loader, model)
		save_tile_probs(
			probs=probs,
			slideIDX=dset.slideIDX,
			reps=reps,
			modelNumber=modelNumber
		)

		if args.dropoutRate == 0.0:
			writeFeatureVectorsToFile(probs, features, modelNumber, dset)
		# Collects the maximum tile probability for each tissue slide and generates the final prediction label
		maxs = ut.groupTopKtilesProbabilities(np.array(dset.slideIDX), probs, len(dset.targets))

		# Collect the indeces for the k top tiles with the maximum predicted probabilites and their corresponding slide indeces and probabilities for each tissue slide
		topk, topgroups, topProbs = ut.groupTopKtilesTesting(
			np.array(dset.slideIDX),
			probs,
			25
		)

		# Collects all tile probabilites for each slide
		newProbs = [[y[0] for y in zip(topProbs, topgroups) if y[1]==x] for x in set(topgroups)]

		# Average the top k tile probabilites and save for the current BN_rep
		finalProbsK = [np.mean(y) for y in newProbs]
		allProbs.append(finalProbsK)

	# Save the final predictions for each BN_rep along with the final average probabilities across all iterations.
	allProbs = list(map(list, zip(*allProbs)))
	# fp = open(os.path.join(args.output, 'predictions_' + modelNumber + "_" + str(args.dropoutRate) + '_temp.csv'), 'w')
	fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
	fp.write('file,target,prediction,probability,' + ",".join(['BN_rep-' + str(x+1) for x in range(t)]) + '\n')
	for name, target, probs in zip(dset.slidenames, dset.targets, allProbs):
		try:
			fp.write('{},{},{},{},{}\n'.format(name, int(target[1]>=0.5), int(np.mean(probs)>=0.5), np.mean(probs), ",".join([str(y) for y in probs])))
		except:
			# Output for non-softlabels
			fp.write('{},{},{},{}\n'.format(name, int(target>=0.5), int(np.mean(probs)>=0.5), np.mean(probs), ",".join([str(y) for y in probs])))
	fp.close()


def enable_dropout (model):
	'''
	Allows dropout to be performed during inference passes of the model. The dropout probability needs to be set withint the model_test.py
	script. The default dropout is 0.2.
	'''
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()



def inference (modelNumber, reps, t, loader, model):
	'''
	Performs an inference pass over the complete dataset.

	Parameters:
		loader:		[iterable over a given dataset] The specified dataset that is loaded in as a PyTorch dataloader object.
		model:		[PyTorch model] The current training model.

	Returns:
		Probabilities for each tile after passing through the model
		The feature vectors for each tile extracted from the penultimate layer of the model.
	'''
	model.eval()
	
	# set dropout layers to train mode
	enable_dropout(model)

	probs = torch.FloatTensor(len(loader.dataset))
	allFeatures = []
	with torch.no_grad():
		instance_features = None
		
		def my_hook(module_, input_, output_):
			'''Generates a hook that allows the feature vectors to be extracted as output.'''
			nonlocal instance_features
			instance_features = output_
		
		for i, (input, target, slide_ids) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TRAINING]"):
			input = input.to(device, non_blocking= True )
			target = target.to(device, non_blocking= True )  
			slide_ids = slide_ids.to(device, non_blocking= True )
			outputList = []
			currentFeatures = []
			currentProbs = []
			sys.stdout.write("\t\t\t\tModel Number: " + str(modelNumber) + "; Bayesian Network iteration: " + str(reps+1) + "/" + str(t) + "; Batch: [" + str(i+1) + "/" + str(len(loader)) + "]\r")

			sys.stdout.flush()

			input = input.to(device)
			target = target.to(device)
			a_hook = model.resnet.avgpool.register_forward_hook(my_hook)
			output = F.softmax(model(input), dim=1)
			a_hook.remove()
			allFeatures += [[x[0][0] for x in y] for y in instance_features.cpu().numpy()]
			probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()

	return (probs.cpu().numpy(), allFeatures)


def writeFeatureVectorsToFile (probs, features, modelNumber, dset):
	with open(os.path.join(args.output, "feature_vectors.tsv"), "w") as fv_out:
		for i, x in enumerate(dset.slidenames):
			print(x, end="\t", file=fv_out)
			firstLine = True
			for l, idx in enumerate(dset.slideIDX):
				grid = dset.grid[l].split("/")[-1].split("-")
				if args.resolution == '20x':
					xGrid = grid[3]
					yGrid = grid[4]
				else:
					xGrid = grid[5]
					yGrid = grid[6]
				slideIDx = dset.slideIDX[l]
				target = dset.targets[slideIDx]
				if idx == i:
					if firstLine:
						print("\t".join([str(idx), xGrid, yGrid, str(probs[l]), "\t".join([str(y) for y in features[l]])]), file=fv_out)
						firstLine = False
					else:
						print("\t".join(['', str(idx), xGrid, yGrid, str(probs[l]), "\t".join([str(y) for y in features[l]])]), file=fv_out)
				else:
					continue



if __name__ == '__main__':
	main()
