function [numK1, numK2, numKeypoints] = main()
    im1 = imread('Input_Left.jpg');
    im2 = imread('Right.jpg'); 
    sigma_orig  = 1;
    num_octaves = 3;
    num_scales = 3;
    r = 5;
    H = [1.5, 0.5, 0; 0, 2.5, 0; 0, 0, 1];
    k1 = [];
    k2 = [];
    
    %gather key points from images via keypoint and SIFT 
    k1 = SIFT(im1, sigma_orig, num_octaves, num_scales, r);
    k2 = SIFT(im2, sigma_orig, num_octaves, num_scales, r);
    
    numK1 = size(k1,1);
    numK2 = size(k2,1);

    k1_points = k1(1:numK1, 3:4);
    k2_points = k2(1:numK2, 3:4);

    %Compute ground truth keypoints from Homography matrix and keypoints in image 1
    groundTruthk2 =[];
    for i = 1:numK1
        point = [k1_points(i,1); k1_points(i,2);1];
        corresponding_point = H * point;
        groundTruthk2 = [groundTruthk2;corresponding_point.'];
    end
    groundTruthk2(:,3) =[];
    
    %find total number of matching keypoints
    numKeypoints = 0;
    for i = 1:numK2
        x1 = k2_points(i,1);
        y1 = k2_points(i,2);
        for j = 1:size(groundTruthk2,1)
            x2 = groundTruthk2(j,1);
            y2 = groundTruthk2(j,2);
            if (CalcDistance(x1, y1, x2, y2) <=3)
                numKeypoints = numKeypoints + 1;
            end
        end
    end
    percentageMatchingKeypoints = numKeypoints/numK2;
    disp("Results for parameters: sigma  = 1, octaves = 3, scales = 3, r = 5")
    fprintf('Number of keypoints in first image: %d\n', numK1);
    fprintf('Number of keypoints in second image: %d\n', numK2);
    fprintf('Number of matching keypoints: %d\n', numKeypoints);
    fprintf('Percentage of matching keypoints: %d\n', percentageMatchingKeypoints);
    
end