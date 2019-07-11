import cv2, os, sys

vidInputPath = sys.argv[1]
frameOutPath = sys.argv[2]

# Following two lines are needed for Break Action Dataset
#vidPaths_Subject = [str(os.path.join(vidInputPath, f) + '/') for f in os.listdir(vidInputPath) if os.path.isdir(os.path.join(vidInputPath, f))]
#vidPaths = [str(os.path.join(vidPath, f) + '/') for vidPath in vidPaths_Subject for f in os.listdir(vidPath) if os.path.isdir(os.path.join(vidPath, f))]

#print(vidPaths_Subject, vidPaths)

# Remove the following line for Breakfast Action Dataset
vidPaths = [vidInputPath]

for camPath in vidPaths:
	#print (camPath)
	vidFilePaths = [os.path.join(camPath, f) for f in os.listdir(camPath) if os.path.isfile(os.path.join(camPath, f)) and f.endswith('.AVI')]
	if not vidFilePaths:
		continue
	for vidFile in vidFilePaths:
		cap = cv2.VideoCapture(vidFile)
		totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		print (vidFile, totalFrame)
		subID = vidFile.split('/')[-2]
		vidName = vidFile.split('/')[-1].split('.')[0].split('_')
		vidName.insert(1,subID)
		vidName = '_'.join(vidName)
		outFilePrefix = os.path.join(frameOutPath, vidName)
		#print (vidName, outFilePrefix, totalFrame)
		if not os.path.exists(outFilePrefix):
			print("Frame out path %s does not exist... Creating..."%outFilePrefix)
			os.makedirs(outFilePrefix)
		currFrame = 0
		while(True):
			try:
				ret,frame = cap.read()
			except:
				continue
			if not ret:
				break

			currFrame += 1

			outFileName = os.path.join(outFilePrefix, "Frame_%06d.jpg"%currFrame)
			#print (outFileName)
			cv2.imwrite(outFileName, frame)
			# break
		cap.release()
		# break

	# break
print ("Finished")