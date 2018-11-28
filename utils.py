from science_rcn.preproc import Preproc, get_gabor_filters
from science_rcn.learning import make_adjacency_graph
from science_rcn.inference import forward_pass, LoopyBPInference, dilate_2d, get_tree_schedule
from science_rcn.learning import sparsify, learn_laterals, make_adjacency_graph, \
    add_underconstraint_edges, adjust_edge_perturb_radii
    
import numpy as np
from numpy.random import rand, randint
from scipy.misc import imresize, imshow
from scipy.ndimage import imread
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
import copy


class Preproc_v2(Preproc):
    def __init__(self, size=9, filter_scale=1.3, cross_channel_pooling=False):
        Preproc.__init__(self, filter_scale=filter_scale, cross_channel_pooling=cross_channel_pooling)
        self.size = size
    @property
    def filters(self):
        return get_gabor_filters(size=self.size,
            filter_scale=self.filter_scale, num_orients=self.num_orients, weights=False)
    
    @property
    def pos_filters(self):
        return get_gabor_filters(size=self.size,
            filter_scale=self.filter_scale, num_orients=self.num_orients, weights=True)
    
    def fwd_infer(self, img, brightness_diff_threshold=30):
        return super(Preproc_v2, self).fwd_infer(img, brightness_diff_threshold=brightness_diff_threshold)


class LoopyBPInference_v2(LoopyBPInference):
    def __init__(self, bu_msg, frcs, edge_factors, pool_shape, preproc_layer,
                 n_iters=300, damping=1.0, tol=1e-5):
        self.n_feats, self.n_rows, self.n_cols = bu_msg.shape
        self.n_pools, self.n_factors = frcs.shape[0], edge_factors.shape[0]
        self.vps, self.hps = pool_shape
        self.frcs = frcs
        self.bu_msg = bu_msg
        self.edge_factors = edge_factors
        self.preproc_layer = preproc_layer
        self.n_iters = n_iters
        self.damping = damping
        self.tol = tol
        
        # Check inputs
        if (np.array([0, self.vps // 2, self.hps // 2]) > frcs.min(0)).any():
            raise RCNInferenceError("Some frcs are too small for the provided pool shape")
        if (frcs.max(0) >= np.array([self.n_feats,
                                    self.n_rows - ((self.vps - 1) // 2),
                                    self.n_cols - ((self.hps - 1) // 2)])).any():
            raise RCNInferenceError("Some frcs are too big for the provided pool "
                                    "shape and/or `bu_msg`")
        if (edge_factors[:, :2].min(0) < np.array([0, 0])).any():
            raise RCNInferenceError("Some variable index in `edge_factors` is negative")
        if (edge_factors[:, :2].max(0) >= np.array([self.n_pools, self.n_pools])).any():
            raise RCNInferenceError("Some index in `edge_factors` exceeds the number of vars")
        if (edge_factors[:, 0] == edge_factors[:, 1]).any():
            raise RCNInferenceError("Some factor connects a variable to itself")
        if not issubclass(edge_factors.dtype.type, np.integer):
            raise RCNInferenceError("Factors should be an integer numpy array")
        
        # Initialize message
        self.unary_messages = np.zeros((self.n_pools, self.vps, self.hps))
        for i, (f, r, c) in enumerate(self.frcs):
            rstart = r - self.vps // 2
            cstart = c - self.hps // 2
            self.unary_messages[i] = self.bu_msg[f,
                                                 rstart:rstart + self.vps,
                                                 cstart:cstart + self.hps] + 0.01 * (2 * rand(*self.unary_messages[i].shape) - 1)


def get_image(filepath, size=(512, 512), padding=40):
    image_arr = imresize(imread(filepath, flatten=True), size)
    img = np.pad(image_arr,
                 pad_width=tuple([(p, p) for p in (padding, padding)]),
                 mode='constant', constant_values=0)
    return img

def grid_plot(imgs, ncols=4, size=3, cmap='gray', axis=False, cbar=False, vmin=None, vmax=None, titles=None):
    nrows = len(imgs) // ncols
    nrows = nrows if nrows*ncols == len(imgs) else nrows + 1
    fig = plt.figure(figsize=(ncols*size, nrows*size))
    for i, img in enumerate(imgs):
        fig.add_subplot(nrows, ncols, i + 1)
        ax = plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        if not axis:
            plt.axis('off')
        if titles:
            plt.title(titles[i])
        if cbar:
            plt.colorbar(ax, fraction=0.046, pad=0.04, format='%.1f')
    
def sparse_repr(sparse):
    frcs, bu_msg = sparse
    sparsified = -np.ones(bu_msg.shape[1:], dtype=int)
    for f, i, j in frcs:
        sparsified[i, j] = f
    return sparsified

def feat_repr(feature, shape=(21,21), child_features=None, filter_size=9, filter_scale=1.5):
    sparsified = -np.ones(shape, dtype=int)
    frcs, _, _ = feature
    fbound = frcs[:, -2:].max(0) + 1
    trans_vec = (shape[0] - fbound[0]) // 2, (shape[1] - fbound[1]) // 2
    
    for f, i, j in frcs:
        if not child_features:
            ic, jc = i + trans_vec[0], j + trans_vec[1]
            sparsified[ic, jc] = f
        else:
            child_frcs, _, _ = child_features[f]
            child_fbound = child_frcs[:, -2:].max(0) + 1
            
            for ff, ii, jj in child_frcs:
                ic = trans_vec[0] + i - child_fbound[0] // 2 + ii
                jc = trans_vec[1] + j - child_fbound[1] // 2 + jj
                sparsified[ic, jc] = ff
    return gabor_repr(sparsified, filter_size=filter_size, filter_scale=filter_scale)

def gabor_repr(sparsified, filter_size=7, filter_scale=1.3):
    f1_feats = [(sparsified[i,j], i, j) for i, j in np.argwhere(sparsified>=0)]
    filters = Preproc_v2(size=filter_size, filter_scale=filter_scale).filters
    r = filter_size // 2
    
    views = np.zeros((len(f1_feats), ) + sparsified.shape, dtype=np.float32)
    for it, (f, i, j) in enumerate(f1_feats):
        view = views[it]
        view[i-r:i+r+1, j-r:j+r+1] = filters[f]
    gabor_val_min = np.mean([filt.min() for filt in filters])
    gabor_val_max = np.mean([filt.max() for filt in filters])
    gabor_map = views.sum(0)
    gabor_map = gabor_map.clip(gabor_val_min, gabor_val_max)
    return gabor_map

def draw_graph(frcs, graph, shape=(200,200), fig_size=6, node_size=3, with_labels=False):
    fig = plt.figure(figsize=(fig_size, fig_size))
    nx.draw_networkx(graph,
                     pos={node: (frcs[node, 2], shape[0] - frcs[node, 1]) for node in range(frcs.shape[0])},
                     node_size=node_size,
                     with_labels=with_labels)
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[0])


#########################################################
def learn_feature_laterals(frcs, perturb_factor):
    graph = nx.Graph()
    graph.add_nodes_from(range(frcs.shape[0]))

#     graph = add_underconstraint_edges(frcs, graph, perturb_factor=perturb_factor, tolerance=0)
    graph = add_underconstraint_edges(frcs, graph, perturb_factor=perturb_factor)
    graph = adjust_edge_perturb_radii(frcs, graph, perturb_factor=perturb_factor)

    edge_factors = np.array(
        [(edge_source, edge_target, edge_attrs['perturb_radius'])
         for edge_source, edge_target, edge_attrs in graph.edges_iter(data=True)])
    return graph, edge_factors

def suppress_explained_locs(bu_msg, explained_locs, suppress_radius=5):
    for f, r, c in explained_locs:
        bu_msg[:,
               max(0, r-suppress_radius):r+suppress_radius+1,
               max(0, c-suppress_radius):c+suppress_radius+1] = -1
    return
    
def get_slide_pos_from_contour(bu_msg, dilate_shape, stride):
    slide_pos = []
    suppress_radius = max(0, stride - 1)
    img = dilate_2d(bu_msg.max(0), dilate_shape)
    img = img > 0
    while True:
        r, c = np.unravel_index(img.argmax(), img.shape)
        if not img[r, c]:
            break
        slide_pos.append((r, c))
        img[max(0, r - suppress_radius):r + suppress_radius + 1, 
            max(0, c - suppress_radius):c + suppress_radius + 1] = False
    return slide_pos

def get_slide_info(bu_msg, feat_factors, pool_shape, stride, contour_hint=None):
    frcs, edge_factors, graph = feat_factors
    img_shape = bu_msg.shape[-2:]
    fbound = frcs[:, -2:].max(0) + 1
    
    i_start, j_start = pool_shape
    i_end = img_shape[0] - pool_shape[0] - fbound[0] + 1
    j_end = img_shape[1] - pool_shape[1] - fbound[1] + 1
    
    slide_info = []
    if contour_hint is not None:
        dilate_shape = contour_hint
        for ic, jc in get_slide_pos_from_contour(bu_msg, dilate_shape, stride):
            i = ic - fbound[0] // 2
            j = jc - fbound[1] // 2
            if not (i_start <= i < i_end) or not (j_start <= j < j_end):
                continue
            
            trans_frcs = frcs.copy()
            trans_frcs[:, 1] += i
            trans_frcs[:, 2] += j
            topleft_corner = (i, j)
            feat_center = (ic, jc)
            slide_info.append((trans_frcs, topleft_corner, feat_center, fbound))
    else:
        for i in range(i_start, i_end, stride):
            for j in range(j_start, j_end, stride):
                if not (i_start <= i < i_end) or not (j_start <= j < j_end):
                    continue
                
                trans_frcs = frcs.copy()
                trans_frcs[:, 1] += i
                trans_frcs[:, 2] += j
                ic = i + fbound[0] // 2
                jc = j + fbound[1] // 2
                topleft_corner = (i, j)
                feat_center = (ic, jc)
                slide_info.append((trans_frcs, topleft_corner, feat_center, fbound))
    return slide_info

def _trans_frcs(frcs, feat_center):
    ic, jc = feat_center
    fbound = frcs[:, -2:].max(0) + 1
    i = ic - fbound[0] // 2
    j = jc - fbound[1] // 2
    trans_frcs = frcs.copy()
    trans_frcs[:, 1] += i
    trans_frcs[:, 2] += j
    return trans_frcs

def calculate_reconstruction_error(imgs, sparses, hierarchy_below):
    rec_error = []
    preproc_layer = Preproc_v2()
    suppress_radius = hierarchy_below[0]['suppress_radius']
    
    for img, sparse in zip(imgs, sparses):
        bu_msg = preproc_layer.fwd_infer(img)
        total_points = len(np.argwhere(bu_msg.max(0) > 0))
        
        explained_locs = [loc for _, backtrace_positions in sparse for loc in backtrace_positions]
        suppress_explained_locs(bu_msg, explained_locs, suppress_radius=suppress_radius)
        unexplained_points = len(np.argwhere(bu_msg.max(0) > 0))
        
        rec_error.append(unexplained_points)
    return sum(rec_error) / float(len(imgs))

def render_sparsification(img, sparse, suppress_radius=5):
    preproc_layer = Preproc_v2(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(img)
    canvases = []
    
    canvas = bu_msg.max(0)
    for (fid, ic, jc), backtrace_positions in sparse:
        for f, r, c in backtrace_positions:
            canvas[r-1:r+2, c-1:c+2] = 2
        
        canvas_ = canvas.copy()
        canvas_[ic-1:ic+2, jc-1:jc+2] = 3
        
#         size = 35
#         gabor_feat = feat_repr(features[fid][0], (size, size))
#         canvas_[:size, -size:] = gabor_feat
        canvases.append(canvas_)
    return canvases

def get_bp_info(bu_msg, trans_frcs, edge_factors, pool_shape, n_iters):
    preproc_layer = Preproc_v2(cross_channel_pooling=True)
    rcn_inf = LoopyBPInference_v2(bu_msg, trans_frcs, edge_factors, pool_shape, preproc_layer, n_iters=n_iters)
    
    # belief propagation
    rcn_inf._reset_messages()
    rcn_inf.infer_pbp()
    
    # sum all messages as score
    beliefs = rcn_inf.unary_messages.copy()
    for f, (var_i, var_j, pert_radius) in enumerate(rcn_inf.edge_factors):
        beliefs[var_j] += rcn_inf.lat_messages[0, f]
        beliefs[var_i] += rcn_inf.lat_messages[1, f]
    
    assignments = np.zeros((rcn_inf.n_pools, 2), dtype=np.int)
    backtrace = []
    for i, (f, r, c) in enumerate(rcn_inf.frcs):
        r_max, c_max = np.where(beliefs[i] == beliefs[i].max())
        choice = randint(len(r_max))
        assignments[i] = np.array([r_max[choice], c_max[choice]])
        rstart = r - rcn_inf.vps // 2
        cstart = c - rcn_inf.hps // 2
        backtrace.append((f, rstart + assignments[i, 0], cstart + assignments[i, 1]))
    backtrace_positions = np.array(backtrace)
    
    score = 0
    for i, (f, r, c) in enumerate(backtrace_positions):
        score += bu_msg[f, r, c]
    return score, backtrace_positions

def forward_pass_v2(frcs, bu_msg, graph, pool_shape, tree_schedule):
    height, width = bu_msg.shape[-2:]
    # Vertical and horizontal pool shapes
    vps, hps = pool_shape

    def _pool_slice(f, r, c):
        assert (r - vps // 2 >= 0 and r + vps - vps // 2 < height and
                c - hps // 2 >= 0 and c + hps - hps // 2 < width), \
            "Some pools are out of the image boundaries. "\
            "Consider increase image padding or reduce pool shapes."
        return np.s_[f,
                     r - vps // 2: r + vps - vps // 2,
                     c - hps // 2: c + hps - hps // 2]

    # Find a schedule to compute the max marginal for the most constrained tree
#     tree_schedule = get_tree_schedule(frcs, graph)
    
    # If we're sending a message out from x to y, it means x has received all
    # incoming messages
    incoming_msgs = {}
    for source, target, perturb_radius in tree_schedule:
        msg_in = bu_msg[_pool_slice(*frcs[source])]
        if source in incoming_msgs:
            msg_in = msg_in + incoming_msgs[source]
            del incoming_msgs[source]
#         msg_in = dilate_2d(msg_in, (2 * perturb_radius + 1, 2 * perturb_radius + 1)) # bug here
        msg_in = dilate_2d(msg_in.copy(), (2 * perturb_radius + 1, 2 * perturb_radius + 1))
        if target in incoming_msgs:
            incoming_msgs[target] += msg_in
        else:
            incoming_msgs[target] = msg_in
    fp_score = np.max(incoming_msgs[tree_schedule[-1, 1]] +
                      bu_msg[_pool_slice(*frcs[tree_schedule[-1, 1]])])
    return fp_score


####################################################################################
color_names = ['gold', 'indianred', 'mistyrose', 'olive', 'pink',
               'navajowhite', 'palegreen', 'burlywood', 'dimgray', 'aquamarine',
               'orange', 'lightsalmon', 'dodgerblue', 'lightseagreen', 'cyan',
               'mediumorchid', 'slateblue']
np.random.seed(0)
feat_colors = np.random.randint(len(color_names), size=1000)
feat_colors = [color_names[i] for i in feat_colors]

def render_sparse(img, sparse, features, r=2):
    background_color = 'darkblue'
    border_color = 'maroon'
    
    im = Image.new('RGB', img.shape, color=background_color)
    draw = ImageDraw.Draw(im)
    
    bu_msg = Preproc_v2().fwd_infer(img)
    border = [(j, i) for i, j in np.argwhere(bu_msg.max(0)>0)]
    draw.point(border, fill=border_color)

    for (f, i, j), frcs in sparse:
        color = feat_colors[f]
        for ff, ii, jj in frcs:
            x, y = jj, ii
            draw.ellipse((x-r, y-r, x+r, y+r), outline=color)
    return im

def animate_belief_progation(img, feature, center_pos, hierarchy_below,
                             pool_shape=(5,5), n_iters=10, size=2, cbar=True, cmap='RdBu_r'):
    frcs, edge_factors, graph = copy.deepcopy(feature)
    trans_frcs = _trans_frcs(frcs, center_pos)
    bu_msg = get_bu_msgs(img, [], hierarchy_below)[-1]
    
    preproc_layer = Preproc_v2(cross_channel_pooling=True)
    rcn_inf = LoopyBPInference(bu_msg, trans_frcs, edge_factors, pool_shape, preproc_layer, n_iters=n_iters)
    
    rcn_inf._reset_messages()
    
    for it in xrange(rcn_inf.n_iters):
         # Compute beliefs
        beliefs = rcn_inf.unary_messages.copy()
        for f, (var_i, var_j, pert_radius) in enumerate(rcn_inf.edge_factors):
            beliefs[var_j] += rcn_inf.lat_messages[0, f]
            beliefs[var_i] += rcn_inf.lat_messages[1, f]

        # Compute outgoing messages
        new_lat_messages = np.zeros_like(rcn_inf.lat_messages)
        for f, (var_i, var_j, pert_radius) in enumerate(rcn_inf.edge_factors):
            new_lat_messages[0, f] = rcn_inf.compute_1pl_message(
                beliefs[var_i] - rcn_inf.lat_messages[1, f], pert_radius)
            new_lat_messages[1, f] = rcn_inf.compute_1pl_message(
                beliefs[var_j] - rcn_inf.lat_messages[0, f], pert_radius)
        
        messages = []
        messages.append([new_lat_messages[d, i] for i in range(len(rcn_inf.edge_factors)) for d in range(2)])
        messages.append([copy.deepcopy(rcn_inf.lat_messages[d, i])
                         for i in range(len(rcn_inf.edge_factors)) for d in range(2)])
        
        delta = new_lat_messages - rcn_inf.lat_messages
        rcn_inf.lat_messages += rcn_inf.damping * delta
        
        messages.append([delta[d, i] for i in range(len(rcn_inf.edge_factors)) for d in range(2)])
        messages.append([copy.deepcopy(rcn_inf.lat_messages[d, i])
                         for i in range(len(rcn_inf.edge_factors)) for d in range(2)])
        
        print 'beliefs'
        vmin, vmax = (beliefs.min(), beliefs.max()) if cbar else (None, None)
        titles = range(len(frcs))
        grid_plot([beliefs[i] for i in range(len(beliefs))],
                  size=size, cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax, titles=titles)
        plt.show()
        
        print 'new old delta update'
        for im, message in enumerate(messages):
            vmin = min([m.min() for m in message]) if cbar else None
            vmax = max([m.max() for m in message]) if cbar else None
            if im == 0:
                titles = []
                for i in range(len(edge_factors)):
                    for d in range(2):
                        if d == 0:
                            source = edge_factors[i,0]
                            target = edge_factors[i,1]
                        else:
                            source = edge_factors[i,1]
                            target = edge_factors[i,0]
                        perturb_radi = edge_factors[i, 2]
                        titles.append('{} - {} [{}]'.format(source, target, perturb_radi))
            else:
                titles = None
            
            grid_plot(message,
                      ncols=len(message), size=size, cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax, titles=titles)
            plt.show()
        
        if np.abs(delta).max() < rcn_inf.tol:
            print ("Parallel loopy BP converged in {} iterations".format(it))
            return
    print ("Parallel loopy BP didn't converge in {} iterations".format(rcn_inf.n_iters))