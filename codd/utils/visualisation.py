import numpy as np
import torch
from mayavi import mlab

def plotPCL(pcl, color, bgcolor=(0,0,0), fig=None):
    '''Plot a pointcloud. pcl [N,3]'''

    if not fig:
        fig = mlab.figure(bgcolor=bgcolor, size=(640,480))

    if type(pcl) == torch.Tensor:
        pcl = pcl.detach().cpu().numpy()

    assert len(pcl.shape) == 2, 'pcl must have two dimensions'
    assert pcl.shape[1] == 3  , 'pcl must have shape [N,3]'

    pointsp = mlab.points3d(pcl[:,0], pcl[:,1], pcl[:,2], color=color, mode='point', figure=fig)
    return fig, pointsp

def plotLines(pts1, pts2, color=(0.8,0.8,0), radius=0.05, fig=None):
    '''Given pts1 and pts2, plot lines between pts1[i] and pts2[i] for i in 0...N. pts1,pts2 [N,3]'''

    assert pts1.dim() == 2, 'pts1 must have two dimensions'
    assert pts2.dim() == 2, 'pts2 must have two dimensions'
    assert pts1.shape[0] == pts2.shape[0], 'pts1 and pts2 must have the same number of points'
    assert pts1.shape[1] == pts2.shape[1] == 3, 'pts1 and pts2 second dimension must have size of 3'

    npts = pts1.shape[0]

    #concatenate points
    ptscat = torch.cat([pts1,pts2], dim=0).detach().cpu().numpy()

    #plot points in invisible mode (0 opacity)
    points_plot = mlab.points3d(ptscat[:,0], ptscat[:,1], ptscat[:,2], opacity=0, mode='point', figure=fig)

    #create connections object (representing lines)
    connections = torch.cat([torch.arange(npts).reshape(-1,1), torch.arange(npts, 2*npts).reshape(-1,1)], dim=1).cpu().numpy()
    points_plot.mlab_source.dataset.lines = connections

    #create tube between lines
    tube = mlab.pipeline.tube(points_plot, tube_radius=radius)
    tube.filter.radius_factor = 1
    tubesurf = mlab.pipeline.surface(tube, color=color)

def plotLinesWithWeights(pts1, pts2, weights, radius=0.05, fig=None):
    '''Given pts1 and pts2, plot lines between pts1[i] and pts2[i] for i in 0...N. pts1,pts2 [N,3]. The weights [N,1] represent the color between the lines.'''

    assert pts1.dim() == 2, 'pts1 must have two dimensions'
    assert pts2.dim() == 2, 'pts2 must have two dimensions'
    assert pts1.shape[0] == pts2.shape[0], 'pts1 and pts2 must have the same number of points'
    assert pts1.shape[1] == pts2.shape[1] == 3, 'pts1 and pts2 second dimension must have size of 3'

    npts = pts1.shape[0]

    #concatenate points
    ptscat = torch.cat([pts1,pts2], dim=0).detach().cpu().numpy()
    weights = weights.view(-1).detach().cpu().repeat(2).numpy()

    #plot points in invisible mode (0 opacity)
    points_plot = mlab.points3d(ptscat[:,0], ptscat[:,1], ptscat[:,2], weights, opacity=0, mode='point', figure=fig)

    #create connections object (representing lines)
    connections = torch.cat([torch.arange(npts).reshape(-1,1), torch.arange(npts, 2*npts).reshape(-1,1)], dim=1).cpu().numpy()
    points_plot.mlab_source.dataset.lines = connections

    #create tube between lines
    tube = mlab.pipeline.tube(points_plot, tube_radius=radius)
    tube.filter.radius_factor = 1
    tubesurf = mlab.pipeline.surface(tube)

    #show color bar
    mlab.colorbar(points_plot, orientation='vertical')

def plotCorrespondences(pts1, pts2, pdist, pidx, fdist, fidx, inlier_thresh, inlier_feature_thresh, plotGT=False):
    '''Plot (detected) correspondences between two point clouds. Correspondences are represented by green if are correct or red if wrong
    Inputs: pts1, pts2 [1,npts,3] points from each point cloud
    pdist, pidx indicating the distance and index from each point in pts1 to the three closest points in pts2. [1,npts,3]
    fdist, fidx, same but in feature space
    inlier_thresh: min distance to be considered a correspondence in R^3
    inlier_feature_thresh: min distance to be considered a matching correspondence in feature space
    plotGT: bool indicating whether or not to plot GT correspondences.

    TP are plotted in green. FP in red. GT (if plotGT) in yellow.
    '''

    #Plot both point clouds
    fig, _ = plotPCL(pts1[0], (1,0,0), None)
    fig, _ = plotPCL(pts2[0], (0,0,1), fig)

    #get only nearest neighbor (first)
    pdist, pidx = pdist[0,:,0], pidx[0,:,0]
    fdist, fidx = fdist[0,:,0], fidx[0,:,0]

    #Compute masks and indexes of gt correspondences and detected correspondences
    mcor = pdist < inlier_thresh #mask of correspondences [N,]
    icor = pidx[mcor].long()     #idx of correspondences  [Ncor,]

    mdetcor = fdist < inlier_feature_thresh    #mask of detected correspondences [N,]
    idetcor = fidx[mdetcor].long()             #idx of detected correspondences  [Ndetcor,]
    correct = (mcor * (pidx == fidx))[mdetcor] #bool representing if detected correspondence is correct [Ndetcor,]

    #Plot GT correspondences
    if plotGT:
        plotLines(pts1[0,mcor], pts2[0,icor], color=(0.5,0.5,0), fig=fig)

    #Plot TP correspondences
    if correct.sum() >0 :
        plotLines(pts1[0,mdetcor][correct], pts2[0,idetcor][correct], color=(0,1,0), fig=fig)

    #Plot FP correspondences
    if (~correct).sum() > 0:
        plotLines(pts1[0,mdetcor][~correct], pts2[0,idetcor][~correct], color=(1,0,0), fig=fig)

    mlab.show()
    return

