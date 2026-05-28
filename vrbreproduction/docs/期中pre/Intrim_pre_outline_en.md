# Interim Pre Presentation Outline

## 1. Overall Framing

The core of this presentation is: **how we reproduced the data processing pipeline described in the paper for generating contact points and trajectories, and how we gradually identified problems, analyzed the causes, and explored improvements during reproduction.**

The whole project can be summarized in one sentence:

> The paper provides only a method description, without open-source code or processed intermediate data, so we had to reproduce the entire data processing and visualization pipeline from raw videos and annotations.

Following a general-to-specific structure, the presentation can be organized as:

1. First, introduce the overall goal of the reproduction task and the main difficulties.
2. Then, explain what we have completed in different stages.
3. Next, use Experiment E16 to show the problems in the most paper-faithful route.
4. Then, explain where the main failure points come from.
5. Finally, present the improvement attempts and the next steps.

---

## 2. Main Challenges

The paper describes a complete data processing pipeline, but the main difficulties in reproduction are:

- there is no open-source code for the data processing part;
- there is no processed intermediate data provided;
- many key details are only described roughly in the paper;
- steps such as contact point generation, reference frame selection, and homography projection all had to be implemented from the paper description;
- failure in a single step may cause the whole subaction to be discarded, so the retention rate has a large impact on the final result.

So this is not simply a matter of running existing code. We had to build and verify the whole pipeline from scratch.

---

## Goal Introduction

The final goal of data processing is to draw contact points, a heatmap, and an arrow indicating the later motion direction of the object on a frame that contains only the object and no hand.

The ground truth is a tensor with shape `(5, 2, 2)`. Its meaning is: five points, each with 2D coordinates, together with confidence values of the same dimensionality.

Figure:

---

## 3. Stage-by-Stage Progress

### Stage 1: Study the model code, draw the model pipeline, and understand the data processing flow. The goal is to understand what the ground truth actually is and clarify the target.

Model pipeline figure:

Briefly introduce the model in one or two sentences.

### 3.1 Stage 2: Run object detection successfully

The goal of this stage is to run the basic detection module first, so that object bounding boxes are available for later contact point generation.

Main work completed in this stage:

- read EPIC annotations;
- locate the video segments corresponding to each subaction;
- run object detection on video frames;
- check whether the detections cover the manipulated object.

### 3.2 Stage 3: Run hand detection and the contact point algorithm

The goal of this stage is to connect hand detection, contact point generation, and trajectory visualization into one pipeline.

Main work completed in this stage:

- run hand detection successfully;
- find the hand-object contact frame;
- generate contact points based on hand boxes and object boxes;
- visualize contact points and trajectories;
- make a preliminary check of whether the contact points fall into reasonable regions.

### 3.3 Stage 3: Single-frame projection verification

The goal of this stage is to first choose an ideal single-frame sample and verify whether the projection logic works.

Main work completed in this stage:

- select an ideal contact frame;
- find the corresponding reference frame;
- compute the projection relation from the contact frame to the reference frame;
- project the contact points onto the reference frame;
- run through the complete data processing and visualization pipeline.

### 3.4 Stage 4: Reproduce the full pipeline on a small batch

The goal of this stage is to test the stability of the whole pipeline on a small batch.

At this stage, several clear problems started to appear:

- the paper describes homography projection too roughly;
- many implementation details are not clearly written;
- when strictly following the paper and projecting to a handless frame, the success rate is low;
- too many samples are filtered out, and the final amount of usable data is insufficient.

---

## 4. The Most Paper-Faithful Reproduction Route: E16

In Experiment E16, we strictly followed the paper and tried to project contact points to a handless reference frame using homography.

Results:

- test scope: `100` subactions;
- final retained: `26` subactions;
- success rate: `26%`.

The paper reports around `54k` samples, while the full dataset contains about `67k` subactions. That means the corresponding pass rate in the paper is roughly:

```text
54k / 67k ≈ 81%
```

In contrast, when we strictly reproduced the E16 route, the pass rate on a small batch was only about `26%`, which suggests that some key implementation details may not have been fully described in the paper.

---

## 5. E16 Pass Funnel

Experiment ID: `E16`

| Stage | Remaining | What this stage filters | Intuitive meaning of the filtered-out samples |
| --- | ---: | --- | --- |
| Raw input | 100 | The first 100 EPIC annotations from P01_109 | Original subaction input |
| Has new-contact candidate | 81 | Whether this subaction contains the frame where contact just starts | 19 subactions did not have a clear new-contact starting point |
| Cell2 contact / GMM pass | 59 | Whether stable enough contact points can be extracted on the contact frame | 22 had contact frames, but hand/object detection or contact points were too poor for GMM fitting |
| Reference found | 38 | Whether a valid reference frame can be found before contact | 21 did not have an active-hand-invisible pre-contact reference satisfying E16 |
| Homography available | 29 | Whether a reliable homography chain can be computed between the reference and contact frames | 9 failed because image matching or homography quality was not reliable enough |
| Final keep | 26 | Whether the projected geometry and heatmap are usable | 3 could be projected, but the geometry was unreasonable or the heatmap gate failed |

From this funnel, the main losses happen at three points:

1. contact point generation fails;
2. a suitable reference frame cannot be found;
3. the homography chain is unstable.

---

## 6. Breakdown of Failure Causes

### 6.1 Contact point generation failure

This corresponds to cases where there is no clear new-contact starting point, or where the contact point quality is not good enough.

Common reasons include:

- poor hand box detection;
- poor object box detection;
- insufficient overlap or proximity between hand and object boxes;
- too few boundary points for contact;
- the number of contact points is smaller than the number of components required by GMM;
- the active hand is found, but the corresponding object detection is unstable.

### 6.2 The `reference found` condition is too strict

The E16 reference condition can be understood intuitively as:

> Before the first contact frame, search backward for a frame where the hand that is making the contact is not visible.

More concretely:

1. first find the first contact frame in the subaction;
2. determine whether the first contact is made by the left hand or the right hand;
3. search only before the first contact frame;
4. search backward from the contact frame, up to 360 frames;
5. find the nearest frame where the active hand is not visible.

Here, "active hand not visible" means:

- if the contact is made by the right hand, then the right hand must not be detected by HOA in the reference frame;
- if the contact is made by the left hand, then the left hand must not be detected by HOA in the reference frame.

It is important to note that the E16 reference is not a strictly humanless frame. It does not require:

- that no person appears in the frame;
- that the other hand disappears;
- that there are no objects;
- that there is no contact;
- that the background is clean;
- that it satisfies the clean hand-object gate used in E14.

So a more accurate description of the E16 reference is:

```text
the nearest frame before contact where the active hand has not appeared yet / is not detected
```

The 21 samples filtered out at this stage can be understood as:

> Looking backward from the first contact, the active hand remains visible all the time, so no valid pre-contact frame with the active hand absent can be found.

Possible cases include:

- the hand is already in the frame from the beginning of the subaction;
- contact happens too early, so there are not enough earlier frames;
- the hand is not yet touching the object but remains visible in the frame;
- HOA keeps detecting the active hand;
- the action is a continuous hold, move, or put-down, so there is no phase where the active hand disappears.

### 6.3 `homography available` is unstable

`homography available` can be understood intuitively as:

> Between the reference frame and the contact frame, every pair of adjacent frames must be alignable, and the alignment quality cannot be too poor.

Instead of computing one large transformation directly from the reference frame to the contact frame, the method computes it frame by frame:

```text
contact frame
-> contact - 1
-> contact - 2
-> ...
-> reference frame
```

Each step requires pairwise homography. The process for each adjacent frame pair is roughly:

1. find visual feature points in both frames, such as table corners, cabinet edges, textures, or object boundaries;
2. match feature points between the two frames;
3. estimate a homography matrix from the matched points;
4. check whether the transformation is reliable.

Reliability mainly depends on:

- whether there are enough matched points;
- whether there are enough good matches;
- whether there are enough RANSAC inliers;
- whether the inlier ratio is high enough;
- whether the transformed result is geometrically reasonable;
- whether there is obvious distortion, flipping, or abnormal area change.

If any adjacent pair fails, the whole homography chain fails.

Therefore, `homography available = 29` in E16 means:

> Out of the 38 samples with a valid reference, only 29 can be stably aligned frame by frame back to the reference frame from the contact frame.

The 9 filtered-out samples can be understood as:

> They have a reference frame and contact points, but the image alignment in the middle is not reliable enough, so the projected location cannot be trusted.

Common reasons include:

- the reference frame is too far from the contact frame, so the scene changes too much;
- camera motion is too fast;
- the hand or object blocks important background texture;
- motion blur;
- the kitchen counter or wall has too little texture, so not enough feature points can be found;
- many matched points are wrong;
- some intermediate frame changes abruptly, such as large hand occlusion, exposure changes, or motion blur;
- the estimated homography projects points to unrealistic locations, so it is rejected by the quality gate.

In short:

```text
reference found asks: is there a valid pre-contact reference frame?
homography available asks: can the points from the contact frame be reliably aligned back to that reference frame?
```

---

## 7. Accuracy Definition and the Current Main Conflict

Our definition of accuracy is:

> whether the projected heatmap / contact points land on the correct manipulated region of the manipulated object.

For example:

- if the action is grasping a pan, the ideal result should be on the pan handle;
- a bad case is when the projection drifts to the body of the pan, or even outside the pan.

The main conflict at the moment is:

1. if we strictly use handless reference frames, the data retention rate is too low;
2. if we relax the reference condition, the amount of data can increase, but the reference frame may still contain hands, which introduces occlusion issues;
3. the projection accuracy of homography is still unstable;
4. the detected contact object is sometimes semantically inconsistent with the annotation.

So the core objective of later improvement is:

> improve data retention without clearly lowering quality, and improve the accuracy of projecting onto the correct region of the target object.

---

## 8. Improvement Attempts So Far

### 8.1 Relax the reference frame condition

One later idea is to relax the handless constraint:

- the target does not have to be a strictly handless frame;
- the projection can go to a few frames before the contact frame;
- then the hand can be masked out.

The expected benefits are:

- the reference frame is closer to contact;
- the homography chain is shorter;
- the scene variation is smaller;
- the success rate may increase.

In E17, for example, using frames 5-10 frames before contact as the reference mainly aims to reduce the difficulty of the homography chain:

> the frame distance is shorter, the scene change is smaller, and the homography chain is shorter, so it is easier to make it stable.

### 8.2 Optical flow route: E7

We also tried optical flow to solve the projection problem.

Conclusion:

- the precision improved slightly;
- but the overall quality was still not good enough;
- projection drift still happened easily.

So optical flow alone still cannot solve the projection problem robustly. Also, the computation time is too long. Pass.

### 8.3 SAM2 + Tracking + Weighted Affine

After that, we tried a SAM2-based solution. The overall workflow is:

1. **SAM**  
   Generate the target object mask and identify which pixels belong to the object.

2. **CoTracker / LK**  
   Track points on the object mask and determine where these object points move from the contact frame to the reference frame.

3. **Weighted affine**  
   Estimate a local transformation from the tracked points, then project the five contact GMM centers to the reference frame.

4. **Mask gate**  
   Use the SAM / reference mask to check whether the projected points actually fall onto the object.

Current effect:

- both precision and quality improved clearly;
- E12 performs relatively well;
- the projection problem has improved significantly.

However, this solution still has some problems:

- computational cost increases a lot;
- semantic mismatch may still happen;
- many reference frames still contain hands;
- some samples still do not have enough contact point precision;
- the data amount still needs to be increased further.

---

## 9. Points Still Needing Optimization

### 9.1 Semantic mismatch

Sometimes the detected contact object does not match the object labeled in the annotation.

For example:

- the annotation says the target is a pan;
- the current method may detect a knife instead;
- the heatmap may appear on the moving hand;
- or it may appear near a moving object, but not on the target object labeled by the annotation.

Possible follow-up solutions:

1. use open-vocabulary detection with semantic awareness;
2. first detect the target object specified by the annotation;
3. then perform hand-object related detection around that target object;
4. then continue with the contact point / trajectory pipeline.

### 9.2 Hands still appear in the reference frame

At the moment, many reference frames still contain hands.

This leads to several follow-up needs:

- how to detect hands in the reference frame;
- how to mask them out;
- how to prevent hand regions from affecting the heatmap or object mask;
- how to ensure that the result remains natural and usable after hand removal.

### 9.3 Some samples still lack enough precision

Some samples still have inaccurate contact point locations.

For example:

> In S45, the action is contact with a pan. Ideally, the contact point should be on the handle, but the current result lands near the center of the pan. The contact point shape looks like a handle-like line, but it is still not projected to the correct region on the object.

This suggests that although the current method is more stable than the original homography route, we still need stronger constraints on the consistency between contact point location and the semantic target region.

### 9.4 Data retention still has room for improvement

According to the paper, there are 54k usable samples out of about 67k total. At the moment, we can reach about one-third to one-half of the amount reported in the paper, which is within the same order of magnitude.

---

## 10. Next Steps

The follow-up work on data processing can be divided into three directions:

### 10.1 Improve data retention

- relax the handless reference frame constraint;
- prioritize reference frames that are closer to the contact frame;
- shorten the projection path;
- reduce sample loss caused by homography / tracking failure.

### 10.2 Improve projection accuracy

- use the SAM + tracking + weighted affine solution as a reference baseline, and look for a better tradeoff between accuracy and computational cost.

### 10.3 Resolve semantic target mismatch

- introduce open-vocabulary detection;
- first lock onto the target object according to the annotation;
- then perform contact point detection around that target object;
- reduce cases where the heatmap is projected onto the wrong object or onto the hand.

### 10.4 Find a way to mask out the hand

Overall follow-up work:

- full-scale data processing;
- adjust the model structure;
- start training.

---

## 11. Presentation Summary

This part can be used as the ending of the talk:

> Overall, we have reproduced the data processing flow described in the paper, from object detection, hand detection, contact point generation, reference frame selection, and homography projection to visualization.  
> However, when strictly following the paper route, only 26% of the subactions in a small batch can finally be retained, and the main bottlenecks are reference frame selection and homography stability.  
> Therefore, we tried alternatives including optical flow, SAM, tracking, and weighted affine. At present, the SAM + tracking route shows a clear improvement in projection quality, but because the computational cost is still too high, it is currently used only as a reference solution.  
> The next priority is to improve data retention while maintaining quality, and make the contact points land more consistently on the annotated target object and the correct operational region.
