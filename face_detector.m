clear all;
clc;
close all;

% create face-detecting object using Viola Jones algorithm
faceDetector = vision.CascadeObjectDetector();
% Point tracker is used for video stabilization, camera motion estimation and target tracking
% Bidirectionally restrict errors, even if the presence of noise can be correctly displayed
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
% create camera
cam = webcam();
% get image's current frame and size
videoFrame = snapshot(cam);
frameSize = size(videoFrame);
% video player
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);        

isTrue = true;  % camera's condition
numPoints = 0; % the number of feature points
numFrame = 0; % the number of frames

% enter loop only when camera is open and frameCount<1000
while isTrue && numFrame < 1000
    % take the current image frame, and each loop returns the current frame
    % frameCount ++ in each loop
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    numFrame = numFrame + 1;    
    
    if numPoints < 10
        % bbox is showed as [x y width height]
        % which is the left-top point's location and size of the window.
        bbox = faceDetector.step(videoFrameGray);
        if ~isempty(bbox)
            % identify feature points in face
            % set the minimum value 1 of the retained feature value
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            % Initializes the trace with the position of original points
            xyPoints = points.Location;
            numPoints = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            % oldpoints perform geometric transformation between previous point and current point
            oldPoints = xyPoints;
            % convert to a series of points:[x1 y1 x2 y2 x3 y3 x4 y4]
            % to identify the face although the face is rotated
            bboxPoints = bbox2points(bbox(1, :));    
            % insert border and mark features
            bboxPolygon = reshape(bboxPoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
    
    % retain the original feature points, detect points' trace and continue to track
    else
        % xyPoints contains every feature points' location
        % isFound check whether the point has track. OK is 1, others 0.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        numPoints = size(visiblePoints, 1);       
        
        if numPoints >= 10
            % geometric transformation between previous point and new point
            % set up the border between
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);            
            
            % perform transforming, and mark feature points inside new border
            bboxPoints = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'green');
            
            % reset feature points
            oldPoints = visiblePoints;
            % set following points for re-detecting old points
            setPoints(pointTracker, oldPoints);
        end
    end
    % Display annotated video frames
    step(videoPlayer, videoFrame);
    % check whether the camera is open
    isTrue = isOpen(videoPlayer);
end

clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);