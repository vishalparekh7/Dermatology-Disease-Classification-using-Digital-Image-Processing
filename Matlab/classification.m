outputfolder=fullfile('dataset');
rootfolder=fullfile(outputfolder, 'data');

categories = {'melanoma', 'nevus', 'seborrheic_keratosis'};
imds = imageDatastore(fullfile(rootfolder,categories),'LabelSource','foldernames');

tbl = countEachLabel(imds);
minCount = min(tbl{:,2});

imds = splitEachLabel(imds, minCount, 'randomized');
countEachLabel(imds);

melanoma = find(imds.Labels == 'melanoma' , 1);
nevus = find(imds.Labels == 'nevus' , 1);
seborrheic_keratosis = find(imds.Labels == 'seborrheic_keratosis' , 1);

% figure
% subplot(2,2,1);
% imshow(readimage(imds,melanoma));
% subplot(2,2,2);
% imshow(readimage(imds,nevus));
% subplot(2,2,3);
% imshow(readimage(imds,seborrheic_keratosis));

net = resnet50();
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize,...
    trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize,...
    testSet, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

trainingLables = trainingSet.Labels;
svmclassifier = fitcecoc(trainingFeatures, trainingLables, ...
    'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

 predictLabels = predict(svmclassifier, testFeatures, 'ObservationsIn', 'columns');
 
 testLables = testSet.Labels;
confMat = confusionmat(testLables,predictLabels)
bsxfun(@rdivide, confMat, sum(confMat,2));

newImage = imread(fullfile('image.jpg'));

ds = augmentedImageDatastore(imageSize,...
    newImage, 'ColorPreprocessing', 'gray2rgb');

imageFeatures = activations(net, ...
    ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

 label = predict(svmclassifier, imageFeatures, 'ObservationsIn', 'columns');
 sprintf('The Image Belongs to %s Class', label)
 accuracy = mean(predictLabels == testSet.Labels);

disp(['Mean accuracy = ' num2str(accuracy)])

save('svmclassifier');