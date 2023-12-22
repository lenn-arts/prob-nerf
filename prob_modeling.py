import numpy as np
from scipy.stats import dirichlet, norm
from scipy.special import logsumexp
import os
import imageio
import json
import cv2
import shutil

def save_img(img, fn, W=100, H=100, F=3, W_target=100, H_target=100, draw_coords=False):
    # img: [w,h,3] in (0,1)
    print("save_img 1: shape", img.shape)
    if F == 5 and draw_coords is True: # coordinates provided
        out = np.zeros((W_target,H_target,3))
        img_save = img.reshape(-1,F)
        print("in coords draw:", img_save.shape)
        c_size = W_target // W
        for cluster in img_save:
            print(cluster.round(3))
            x = int(np.round(cluster[3]))
            y = int(np.round(cluster[4]))
            x_start = max(0,x-c_size//2)
            x_end = min(H_target,x+c_size//2)
            y_start = max(0,y-c_size//2)
            y_end = min(W_target,y+c_size//2)
            print(x_start, x_end, y_start, y_end)
            out[x_start:x_end, y_start:y_end] += cluster[:3]*0.3
        img_save = out
    else: # coordinates fixed
        img_save = img.reshape(W,H,F)
        if img_save.shape[1] < H_target:
            img_save = cv2.resize(img_save[...,:3], (W_target,H_target), interpolation=cv2.INTER_NEAREST)
        print("save_img 2: shape", img.shape)
    img_save = (np.clip(img_save,0.,1.)[...,:3]*255).astype(np.uint8)
    imageio.imsave(f"{fn}.png", img_save)


def preprocess_imgs(imgs, c_size=1, W=100, H=100, F=3):
    ### input modification
    n_clusters_row = n_clusters_col = W//c_size
    xs_new = np.zeros((imgs.shape[0], n_clusters_row, n_clusters_col, F))
    for i_img in range(imgs.shape[0]):
        for j_row in range(n_clusters_row):
            for k_col in range(n_clusters_col):
                xs_new[i_img, j_row, k_col, :3] = np.mean(imgs[i_img, j_row*c_size:(j_row+1)*c_size, k_col*c_size:(k_col+1)*c_size], axis=(-3,-2))
                if F>3:
                    x_coord = (j_row*c_size + (j_row+1)*c_size)/2.
                    y_coord = (k_col*c_size + (k_col+1)*c_size)/2.
                    xs_new[i_img, j_row, k_col, 3:] = [x_coord,y_coord]
    imgs = xs_new
    return imgs

"""
input: images
output: N < |images| image names selected iteratively by uncertainty
"""
def get_indices(img_fps, N=5):
    shutil.rmtree("run") # ! CAUTION
    os.makedirs("run", exist_ok=False)
    #### vars
    imgs = []
    W = H = 100
    F = 5 # 3 for rbg, 5 for rgb + pos
    K = 16
    c_size = 25
    variable_pos = True

    #### get imgs
    for img_fp in img_fps:
        img_tmp = imageio.imread(img_fp)
        img_tmp = cv2.resize(img_tmp, (W,H), interpolation=cv2.INTER_AREA)
        imgs.append(img_tmp)
    imgs = np.asarray(imgs)
    imgs = imgs.astype(np.float32)
    imgs /= 255.
    print(imgs[:,:,:,3].min(), imgs[:,:,:,3].max())
    imgs[imgs[...,3] < 1.] = 1. # overwrite transparency
    print(imgs[:,:,:,3].min(), imgs[:,:,:,3].max())
    imgs = imgs[...,:3] # drop transparency
    imgs_org = imgs # full size
    imgs = preprocess_imgs(imgs, c_size=c_size, F=F, W=W, H=H) # aggregated
    imgs = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]) # [N, w*h, 3]
    if not variable_pos: # a1, a2, b1, b2
        imgs = imgs.reshape(imgs.shape[0], -1) # [N, wh3] -> N_img samples, F=wh3
    #else:
    #    imgs = imgs.reshape(-1, imgs.shape[-1]) # [Nwh, 3+] -> N_img*w*h samples, F=F
    print(len(imgs), imgs.shape)

    #### algorithm
    start_img_id = np.random.choice(len(imgs), size=1).squeeze()
    #fit_imgs = imgs
    fit_imgs_org = imgs_org[start_img_id:start_img_id+1]
    fit_imgs = imgs[start_img_id:start_img_id+1]
    fit_fps = img_fps[start_img_id:start_img_id+1]
    imgs_org = np.delete(imgs_org, start_img_id, axis=0)
    imgs = np.delete(imgs, start_img_id, axis=0)
    img_fps.pop(start_img_id)
    for l in range(N-1):
        print("\nFIT_IMGS", fit_imgs.shape)
        save_img(fit_imgs[-1], f"run/{l}_lastref"+fit_fps[-1].split("/")[-1], F=F, W=W//c_size, H=H//c_size)
        save_img(fit_imgs_org[-1], f"run/{l}_lastref"+fit_fps[-1].split("/")[-1]+"_org", F=3, W=W, H=H)
        params = fit_gmm(fit_imgs, F=F, W=W//c_size, H=H//c_size, variable_pos=variable_pos, K=K, xs_out=imgs)
        smallest_prob_img_ids, smallest_probs = select_least_prob_img(imgs, params, variable_pos=variable_pos)
        print(smallest_prob_img_ids)
        for m,mth_id in enumerate(smallest_prob_img_ids):
            print(img_fps[mth_id])
            save_img(imgs[mth_id], f"run/{l}_{m}_"+img_fps[mth_id].split("/")[-1]+f"_{smallest_probs[m]}", F=F, W=W//c_size, H=H//c_size)
            save_img(imgs_org[mth_id], f"run/{l}_{m}_"+img_fps[mth_id].split("/")[-1]+f"_{smallest_probs[m]}_org", F=3, W=W, H=H)
        new_img_id = smallest_prob_img_ids[0]
        #fit_imgs.append(imgs[new_img_id])
        #print(fit_imgs.shape, new_img_id, imgs[new_img_id].shape)
        fit_imgs = np.append(fit_imgs, imgs[new_img_id][np.newaxis,...], axis=0)
        fit_imgs_org = np.append(fit_imgs_org, imgs_org[new_img_id][np.newaxis,...], axis=0)
        fit_fps.append(img_fps[new_img_id])
        imgs = np.delete(imgs, new_img_id, axis=0)
        imgs_org = np.delete(imgs_org, new_img_id, axis=0)
        img_fps.pop(new_img_id)
    print([int(fit_fp.split("_")[-1].split(".png")[0]) for fit_fp in fit_fps])

def fit_gmm(imgs, W=100,H=100,F=3, K=1, variable_pos=False, xs_out=None):
    ### reshaping
    #imgs = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]) # [N, w*h, 3]
    #imgs = imgs.reshape(imgs.shape[0], -1)
    #print("fit_gmm:", len(imgs), imgs.shape)
    if variable_pos:
        imgs = imgs.reshape(-1, imgs.shape[-1]) # [Nwh, 3+] -> N_img*w*h samples, F=F

    #### hyperparams
    NUM_COMPONENTS = K # K   x=img, K=1  (all images)
    NUM_ITERATIONS = 1000
    N_IMGS = len(imgs)
    if xs_out is None:
        xs_out = imgs[-10:] # for debugging during execution, in addition to run on xs-out
    #imgs = imgs[:-10]
    print("new", len(imgs), imgs.shape)
    #save_img(imgs[0], "tmp", W=W, H=H, F=F)
    N_FEAT = imgs.shape[-1]
    # to reconstruct image exactly: hyper_eta = 1000, hyper_sigma = 0.0001
    # to have one image per component eta=1, sigma=1
    # too large (both equal): all noise
    # too small (both equal): reconstruction but dark because doesnt get over 0.5
    hyper_alpha = 1.0 # !!!
    hyper_eta = np.ones(N_FEAT)*100 # std of beta prior # SAME FOR ALL COMPONENTS
    hyper_sigma = np.ones(N_FEAT)*0.1 # std of likelihood / data-generating distribution # SAME FOR ALL COMPONENTS
    print(f"Using: K={NUM_COMPONENTS}, N_FEAT={N_FEAT}")

    ### Gibbs sampling: initial draws
    theta_sample = np.random.rand(NUM_COMPONENTS) # weights # dist # [N_COMP]
    theta_sample /= np.sum(theta_sample)
    theta_sample = np.log(theta_sample)
    betas_sample = np.random.rand(NUM_COMPONENTS, N_FEAT) # dist # [N_COMP, N_FEAT]
    print("betas_sample", betas_sample.shape, betas_sample[0,500:503])
    zs_sample = np.random.choice(NUM_COMPONENTS, replace=True, size=imgs.shape[0]) # var # [N_SAMP]

    ### Gibbs sampling
    pred_log_probs = []
    zs_sample_history = []
    log_p_history = []
    for i in range(NUM_ITERATIONS):
        # PROPORTIONS THETA (CAT. DIST OVER COMPS)
        theta_dist_param = hyper_alpha + np.bincount(zs_sample, minlength=NUM_COMPONENTS) 
        theta_sample = np.log(dirichlet.rvs(alpha=theta_dist_param)[0]) # [N_COMPS]
        #print("theta_sample", theta_sample.shape, theta_sample)

        # COMPONENTS BETA (EACH: GAUS. DIST OVER SAMPLE SPACE)
        for k in range(NUM_COMPONENTS):
            xs_select = imgs[zs_sample==k]
            if len(xs_select) == 0: 
                continue
            component_mean = np.sum(xs_select, axis=0) / xs_select.shape[0] # [N_FEAT]
            #print(component_mean[15000:15005])
            #print(((len(xs_select)/hyper_sigma**2)))
            #print((len(xs_select)/hyper_sigma**2 + 1/hyper_eta**2) )
            #exit()
            posterior_mean = ( (len(xs_select)/hyper_sigma**2) / (len(xs_select)/hyper_sigma**2 + 1/hyper_eta**2) )*component_mean # [N_FEAT]
            posterior_variance = 1. / (len(xs_select)/hyper_sigma**2 + 1/hyper_eta**2) # [N_FEAT]
            betas_sample[k, :] = norm.rvs(loc=posterior_mean, scale=posterior_variance)

        if not variable_pos:
            print(i,"betas_sample", betas_sample.shape, betas_sample[0,500:503].round(4), betas_sample.reshape((-1,W,H,F))[...,:3].min().round(3), betas_sample.reshape((-1,W,H,F))[...,:3].max().round(3), betas_sample.reshape((-1,W,H,F))[...,3:].min().round(3), betas_sample.reshape((-1,W,H,F))[...,3:].max().round(3))
        else:
            print(i,"betas_sample", betas_sample.shape, betas_sample[...,:3].min().round(3), betas_sample[...,:3].max().round(3), betas_sample[...,3:].min().round(3), betas_sample[...,3:].max().round(3))
        

        # ASSIGNMENTS Z (EACH: ONE-HOT VECTOR OVER N_COMPS)
        for j in range(len(imgs)):
            # gauss:
            zs_sample_param = np.zeros_like(theta_sample)
            for k in range(NUM_COMPONENTS):
                log_densities_k = norm.logpdf(imgs[j], loc=betas_sample[k], scale=hyper_sigma)
                log_densities_k_sum = theta_sample[k]+np.sum(log_densities_k)
                zs_sample_param[k] = log_densities_k_sum
            
            zs_sample_param_exp = np.exp(zs_sample_param - logsumexp(zs_sample_param))
            zs_sample[j] = np.random.choice(NUM_COMPONENTS, p=zs_sample_param_exp) 

        # LOGGING: LOG JOINT
        log_p_theta = dirichlet.logpdf(np.exp(theta_sample), alpha=np.array([hyper_alpha]*NUM_COMPONENTS)) 
        
        log_p_beta = []
        for k in range(NUM_COMPONENTS):
            log_p_beta_k = norm.logpdf(betas_sample[k], loc=0, scale=hyper_eta) 
            log_p_beta.append(np.sum(log_p_beta_k)) 
        log_p_beta = logsumexp(log_p_beta) 
        
        log_p_zs = []
        log_p_xs = []
        for j in range(len(imgs)):
            log_p_zs.append(theta_sample[zs_sample[j]]) # prob of cateogrical # [1]
            log_p_xs_components = norm.logpdf(imgs[j], loc=betas_sample[zs_sample[j]], scale=hyper_sigma) # [N_FEAT]
            log_p_xs.append(np.sum(log_p_xs_components)) # [1]
        log_p_xs = logsumexp(log_p_xs) 
        #log_p = log_p_theta + log_p_beta + log_p_zs + log_p_xs
        log_p = log_p_xs # !!! can be positive since logpdf can be positive -> use actual probability instead
        #print("log_p", log_p.round(3))
        log_p_history.append(log_p)
        if len(log_p_history)>50 and np.mean(np.diff(log_p_history[-10:])) < 0.:
            print("converged")
            break

        pred_log_prob = get_pred_log_prob(xs_out, theta_sample, betas_sample, hyper_sigma)
        print("pred_log_prob", pred_log_prob.round(3))
        pred_log_probs.append(pred_log_prob)
        

        # CONVERGENCE: CHECK ASSIGNMENT CHANGE
        """print("zs_sample", zs_sample)
        print("log_p", log_p, log_p.shape)
        if len(zs_sample_history) > 0 and np.all(zs_sample == zs_sample_history[-1]):
            same_counter += 1
        else:
            same_counter = 0
            zs_sample_history = []
        if same_counter == 20:
            print("converged")
        """

    for k in range(NUM_COMPONENTS):
        if variable_pos:
            save_img(betas_sample[k], f"prints/beta_{k}", W=W, H=H, F=F, draw_coords=True)
        if not variable_pos:
            save_img(betas_sample[k], f"prints/beta_{k}_fixed", W=W, H=H, F=F, draw_coords=False)
    if variable_pos:
        save_img(betas_sample.reshape(-1,betas_sample.shape[-1]), f"prints/beta_all", W=W, H=H, F=F, draw_coords=True)
    np.save("pred_log_probs", pred_log_probs)

    return (theta_sample, betas_sample, hyper_sigma)
        
    
def get_pred_log_prob_sample(x_new, theta, beta, hyper_sigma):
    # theta assumed to be log
    k_terms = []
    for k in range(len(theta)):
        log_densities_k = norm.logpdf(x_new, loc=beta[k], scale=hyper_sigma)
        log_densities_k_sum = theta[k]+np.sum(log_densities_k)
        k_terms.append(log_densities_k_sum)
    x_prob = logsumexp(k_terms)
    out = x_prob
    return out # log space

def get_pred_log_prob(xs_new, *args):
    # theta assumed to be log
    xs_probs = []
    for j in range(len(xs_new)):
        x_prob = get_pred_log_prob_sample(xs_new[j], *args)
        xs_probs.append(x_prob)

    out = logsumexp(xs_probs)/len(xs_new)
    return out # log space

def select_least_prob_img(imgs, params, variable_pos=False):
    # x can either be img or region or pixel
    imgs_probs = []
    samples_per_img = 1
    if variable_pos:
        print(imgs.shape)
        samples_per_img = imgs.shape[1]
        imgs = imgs.reshape(-1, imgs.shape[-1]) # [Nimgs*Ncluster, F]
    img_accum_prob = [] # xs_probs
    for j in range(len(imgs)): # go over samples xs (not necessarily imgs)
        x_prob = get_pred_log_prob_sample(imgs[j], *params)
        img_accum_prob.append(x_prob) # log space
        if j%samples_per_img == (samples_per_img-1): # if image: every iteration
            imgs_probs.append(logsumexp(img_accum_prob)/len(img_accum_prob))
            img_accum_prob = []
    imgs_probs = np.array(imgs_probs)
    m=5
    m_smallest_ids = np.argsort(imgs_probs)[:m]
    print("all candidate pred. prob. mean:", np.mean(imgs_probs).round(3))
    print(m_smallest_ids[:m])
    print(imgs_probs[m_smallest_ids[:m]])
    print(np.argmin(imgs_probs))
    return m_smallest_ids, imgs_probs[m_smallest_ids[:m]]


def scratch(): #!!! DELETE
    names = sorted(os.listdir("/Users/lennartschulze/Downloads/flowers/png"))
    names = {i : names[i] for i in range(len(names))}

    # import text
    xs = np.load("/Users/lennartschulze/Downloads/flowers/flowers_features.npy")[:600]
    xs_out_len = int(0.2*len(xs))
    np.random.seed(1)
    xs_out_ind = np.random.choice(len(xs), size=xs_out_len, replace=False)
    xs_out = xs[xs_out_ind]
    xs_in_ind = [i for i in range(len(xs)) if i not in xs_out_ind]
    print(xs.shape, xs.max(), xs.min())
    xs = xs[xs_in_ind]
    print(xs_out_ind, xs_in_ind)

    # hyperparams
    NUM_COMPONENTS = 6 # K
    NUM_ITERATIONS = 100000
    N_IMGS = len(xs)
    N_FEAT = xs.shape[-1]

    hyper_alpha = 1.
    hyper_eta = np.ones(N_FEAT)*0.5 # std of beta prior # SAME FOR ALL COMPONENTS
    hyper_sigma = np.ones(N_FEAT)*0.5 # std of likelihood / data-generating distribution # SAME FOR ALL COMPONENTS

    def get_pred_log_prob(xs_new, theta, beta):
        # theta assumed to be log
        k_terms = []
        xs_probs = []
        for j in range(len(xs_new)):
            for k in range(len(theta)):
                log_densities_k = norm.logpdf(xs_new[j], loc=beta[k], scale=hyper_sigma)
                log_densities_k_sum = theta[k]+np.sum(log_densities_k)
                k_terms.append(log_densities_k_sum)
            x_prob = logsumexp(k_terms)
            xs_probs.append(x_prob)

        out = logsumexp(xs_probs)/len(xs_new)
        print("avg", out)
        return out

    theta_sample = np.random.rand(NUM_COMPONENTS) # weights # dist # [N_COMP]
    theta_sample /= np.sum(theta_sample)
    theta_sample = np.log(theta_sample)
    betas_sample = np.random.rand(NUM_COMPONENTS, N_FEAT) # dist # [N_COMP, N_FEAT]
    zs_sample = np.random.choice(NUM_COMPONENTS, replace=True, size=xs.shape[0]) # var # [N_SAMP]
    zs_sample_history = []
    log_joints = []
    log_p_history = []
    for i in range(NUM_ITERATIONS):
        print(i)
        # THETA
        theta_dist_param = hyper_alpha + np.bincount(zs_sample, minlength=NUM_COMPONENTS) 
        theta_sample = np.log(dirichlet.rvs(alpha=theta_dist_param)[0]) 

        # COMPONENTS BETA
        for k in range(NUM_COMPONENTS):
            xs_select = xs[zs_sample==k]
            if len(xs_select) == 0: 
                continue
            component_mean = np.sum(xs_select, axis=0) / xs_select.shape[0] # [N_FEAT]
            posterior_mean = ( (len(xs_select)/hyper_sigma**2) / (len(xs_select)/hyper_sigma**2 + 1/hyper_eta**2) )*component_mean # [N_FEAT]
            posterior_variance = 1. / (len(xs_select)/hyper_sigma**2 + 1/hyper_eta**2) # [N_FEAT]
            betas_sample[k, :] = norm.rvs(loc=posterior_mean, scale=posterior_variance)

        # ASSIGNMENTS Z
        for j in range(len(xs)):
            # gauss:
            zs_sample_param = np.zeros_like(theta_sample)
            for k in range(NUM_COMPONENTS):
                log_densities_k = norm.logpdf(xs[j], loc=betas_sample[k], scale=hyper_sigma)
                log_densities_k_sum = theta_sample[k]+np.sum(log_densities_k)
                zs_sample_param[k] = log_densities_k_sum
            
            zs_sample_param_exp = np.exp(zs_sample_param - logsumexp(zs_sample_param))
            zs_sample[j] = np.random.choice(NUM_COMPONENTS, p=zs_sample_param_exp) 
        
        # LOGGING: LOG JOINT
        log_p_theta = dirichlet.logpdf(np.exp(theta_sample), alpha=np.array([hyper_alpha]*NUM_COMPONENTS)) 
        
        log_p_beta = []
        for k in range(NUM_COMPONENTS):
            log_p_beta_k = norm.logpdf(betas_sample[k], loc=0, scale=hyper_eta) 
            log_p_beta.append(np.sum(log_p_beta_k)) 
        log_p_beta = logsumexp(log_p_beta) 
        
        log_p_zs = []
        log_p_xs = []
        for j in range(len(xs)):
            log_p_zs.append(theta_sample[zs_sample[j]]) # prob of cateogrical # [1]
            log_p_xs_components = norm.logpdf(xs[j], loc=betas_sample[zs_sample[j]], scale=hyper_sigma) # [N_FEAT]
            log_p_xs.append(np.sum(log_p_xs_components)) # [1]
        log_p_xs = logsumexp(log_p_xs) 
        #log_p = log_p_theta + log_p_beta + log_p_zs + log_p_xs
        log_p = log_p_xs
        log_p_history.append(log_p)
        

        print("zs_sample", zs_sample)
        print("log_p", log_p, log_p.shape)
        if len(zs_sample_history) > 0 and np.all(zs_sample == zs_sample_history[-1]):
            same_counter += 1
        else:
            same_counter = 0
            zs_sample_history = []
        if same_counter == 20:
            print("converged")
            get_pred_log_prob(xs_out, theta_sample, betas_sample)
            break
        zs_sample_history.append(zs_sample.copy())
        
    np.save("log_p_hist.npy",log_p_history)

# MAIN
base_dir = "../data/nerf_synthetic/lego"
split = "train"
with open(os.path.join(base_dir, f'transforms_{split}.json'), 'r') as fp:
    json_obj = json.load(fp)
ds = [os.path.join(base_dir, frame['file_path']+".png") for frame in json_obj['frames']]
get_indices(ds)