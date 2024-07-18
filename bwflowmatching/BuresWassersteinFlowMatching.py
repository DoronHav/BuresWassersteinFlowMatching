from functools import partial

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np  # type: ignore
import optax # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange, tqdm # type: ignore
from flax.training import train_state # type: ignore
import tensorflow_probability.substrates.jax.math as jax_prob # type: ignore

import bwflowmatching.utils_OT as utils_OT # type: ignore
import bwflowmatching.utils_Noise as utils_Noise # type: ignore
import bwflowmatching.utils_Pointclouds as utils_Pointclouds # type: ignore
from bwflowmatching._utils_Neural import BuresWassersteinNN # type: ignore
from bwflowmatching.DefaultConfig import DefaultConfig # type: ignore






class BuresWassersteinFlowMatching:
    """
    Initializes BW Flow Matching model and processes input point clouds


    :param point_clouds: (list of np.array) list of train-set point clouds to calculates means and covariances from
    :param means: (list of np.array) list of train-set point clouds to calculates means and covariances from
    :param config: (flax struct.dataclass) object with parameters

    :return: initialized WassersteinFlowMatching model
    """

    def __init__(
        self,
        point_clouds: list = None,
        means: np.array = None,
        covariances: np.array = None,
        labels: np.array =  None,
        noise_type: str =  'gaussian',
        config = DefaultConfig,
    ):


        self.config = config
        self.point_clouds = point_clouds


        if(point_clouds is not None):
            self.means, self.covariances = utils_Pointclouds.calc_mean_and_cov(point_clouds, minval = -1, maxval = 1)
        else:
            self.means, self.covariances = means, covariances

        self.mean_scale =jnp.abs(self.means).mean()
        self.cov_scale = jnp.diagonal(self.covariances, axis1 = 1, axis2 = 2).mean()

        self.space_dim = self.means.shape[-1]


        self.monge_map_jit = jax.jit(jax.vmap(utils_OT.gaussian_monge_map, (0, 0), 0))
        self.mccann_interpolation_jit = jax.jit(jax.vmap(utils_OT.mccann_interpolation, (0, 0, 0), 0))
        self.mccann_derivative_jit = jax.jit(jax.vmap(utils_OT.mccann_derivative, (0, 0, 0), 0))

        self.psd_project_jit = jax.jit(jax.vmap(utils_OT.project_to_psd, 0, 0))

        self.noise_type = noise_type
        self.noise_func = getattr(utils_Noise, self.noise_type)

        self.mini_batch_ot_mode = config.mini_batch_ot_mode

        if(self.mini_batch_ot_mode):
            self.frechet_dist_jit = jax.jit(
                jax.vmap(utils_OT.frechet_distance, (0, 0), 0),
            )
            self.minibatch_ot_eps = config.minibatch_ot_eps
            self.minibatch_ot_lse = config.minibatch_ot_lse

        if(labels is not None):
            self.label_to_num = {label: i for i, label in enumerate(np.unique(labels))}
            self.num_to_label = {i: label for i, label in enumerate(np.unique(labels))}
            self.labels = jnp.array([self.label_to_num[label] for label in labels])
            self.label_dim = len(np.unique(labels))
            self.config.label_dim = self.label_dim 
        else:
            self.labels = None
            self.label_dim = -1


    def create_train_state(self, model, learning_rate, decay_steps = 10000, key = random.key(0)):
        """
        :meta private:
        """

        subkey, key = random.split(key)
        means_noise, covariances_noise = self.noise_func(size = 10, dimention = self.space_dim, mean_scale = self.mean_scale, cov_scale = self.cov_scale, key = subkey)
        
        subkey, key = random.split(key)
        
        if(self.labels is not None):
            params = model.init(rngs={"params": subkey}, 
                                means = means_noise, 
                                covariances = covariances_noise,
                                t = jnp.ones((means_noise.shape[0])), 
                                labels =  jnp.ones((means_noise.shape[0])),
                                deterministic = True)['params']
        else:
            params = model.init(rngs={"params": subkey}, 
                                means = means_noise, 
                                covariances = covariances_noise,
                                t = jnp.ones((means_noise.shape[0])), 
                                deterministic = True)['params']

        lr_sched = optax.exponential_decay(
            learning_rate, decay_steps, 0.6, staircase = True,
        )
        #lr_sched = learning_rate
        tx = optax.adam(lr_sched)  #

        return train_state.TrainState.create(apply_fn= model.apply, params=params, tx=tx)


    @partial(jit, static_argnums=(0,))
    def minibatch_ot(self, means_batch, covariances_batch, means_noise, covariances_noise, key = random.key(0)):

        """
        :meta private:
        """
            
        tri_u_ind = jnp.stack(jnp.triu_indices(means_batch.shape[0]), axis=1)

        # compute pairwise ot between point clouds and noise:

        ot_matrix = jax_prob.fill_triangular(self.frechet_dist_jit(
                                            [means_batch[tri_u_ind[:, 0]], covariances_batch[tri_u_ind[:, 0]]],
                                            [means_noise[tri_u_ind[:, 1]], covariances_noise[tri_u_ind[:, 1]]],
                                            ))
        ot_matrix = ot_matrix + ot_matrix.T - jnp.diag(jnp.diag(ot_matrix))

        pairing_matrix = utils_OT.sinkhorn_from_distance(ot_matrix, self.minibatch_ot_eps, self.minibatch_ot_lse)
        pairing_matrix = pairing_matrix/pairing_matrix.sum(axis = 1)

        subkey, key = random.split(key)
        noise_ind = random.categorical(subkey, logits = jnp.log(pairing_matrix + 0.000001))
        return(noise_ind)
    


    @partial(jit, static_argnums=(0,))
    def train_step(self, state, means_batch, covariances_batch, labels_batch = None, key=random.key(0)):
        """
        :meta private:
        """
        subkey_t, subkey_noise, key = random.split(key, 3)
        
        means_noise, covariances_noise = self.noise_func(size = means_batch.shape[0], dimention = self.space_dim, mean_scale = self.mean_scale, cov_scale = self.cov_scale, key = subkey_noise)

        if(self.mini_batch_ot_mode):
            subkey_resample, key = random.split(key)
            noise_ind = self.minibatch_ot(means_batch, covariances_batch, means_noise, covariances_noise, key = subkey_resample)
            means_noise = means_noise[noise_ind]
            covariances_noise = covariances_noise[noise_ind]


        interpolates_time = random.uniform(subkey_t, (means_batch.shape[0],), minval=0.0, maxval=1.0)
        
        A_flow, b_flow = self.monge_map_jit([means_noise, covariances_noise], 
                                            [means_batch, covariances_batch])
        
        interpolates_means, interpolates_covariances  = self.mccann_interpolation_jit([means_noise, covariances_noise], [A_flow, b_flow], 1 - interpolates_time)
        interpolates_means_dot, interpolates_covariances_dot = self.mccann_derivative_jit([means_noise, covariances_noise], [A_flow, b_flow], 1 - interpolates_time)

        subkey, key = random.split(key)
        def loss_fn(params):       
            predicted_mean_dot, predicted_cov_dot = state.apply_fn({"params": params},  
                                            means = interpolates_means, 
                                            covariances = interpolates_covariances,
                                            t = interpolates_time, 
                                            labels = labels_batch,
                                            deterministic = False, 
                                            dropout_rng = subkey)
            mean_error = jnp.mean(jnp.square(predicted_mean_dot - interpolates_means_dot))
            cov_error = jnp.mean(jnp.square(predicted_cov_dot - interpolates_covariances_dot))
            loss = mean_error/self.space_dim + cov_error 
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, loss)

    def train(
        self,
        training_steps=32000,
        batch_size=16,
        verbose=8,
        init_lr=0.0001,
        decay_num=4,
        key=random.key(0),
    ):
        """
        Set up optimization parameters and train the ENVI moodel


        :param training_steps: (int) number of gradient descent steps to train ENVI (default 10000)
        :param batch_size: (int) size of train-set point clouds sampled for each training step  (default 16)
        :param verbose: (int) amount of steps between each loss print statement (default 8)
        :param init_lr: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_num: (int) number of times of learning rate decay during training (default 4)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """



        subkey, key = random.split(key)

        self.FlowMatchingModel = BuresWassersteinNN(config = self.config)
        self.state = self.create_train_state(model = self.FlowMatchingModel,
                                             learning_rate=init_lr, 
                                             decay_steps = int(training_steps / decay_num), 
                                             key = subkey)


        tq = trange(training_steps, leave=True, desc="")
        self.losses = []
        for training_step in tq:

            subkey, key = random.split(key, 2)
            batch_ind = random.choice(
                key=subkey,
                a = self.means.shape[0],
                shape=[batch_size])
            
            means_batch, covariances_batch = self.means[batch_ind],  self.covariances[batch_ind]

            subkey, key = random.split(key, 2)
            if(self.labels is not None):
                labels_batch = self.labels[batch_ind]
            else:
                labels_batch = None
            self.state, loss = self.train_step(self.state, means_batch, covariances_batch, labels_batch, key = subkey)
            self.losses.append(loss) 

            if(training_step % verbose == 0):
                tq.set_description(": {:.3e}".format(loss))

    @partial(jit, static_argnums=(0,))
    def get_flow(self, means, covariances, t, labels = None):

        if(means.ndim == 1):
            means = means[None, :]
            covariances = covariances[None, :]  

        flow_mean, flow_cov = self.FlowMatchingModel.apply({"params": self.state.params},
                    means = means, 
                    covariances = covariances,
                    t = t * jnp.ones(covariances.shape[0]), 
                    labels = labels,
                    deterministic = True)
        flow_mean,flow_cov = jnp.squeeze(flow_mean), jnp.squeeze(flow_cov)

        return([flow_mean, flow_cov])
        

    def generate_samples(self, num_samples = 10, timesteps = 100, generate_labels = None, init_noise = None, key = random.key(0)): 
        """
        Generate samples from the learned flow


        :param num_samples: (int) number of samples to generate (default 10)
        :param timesteps: (int) number of timesteps to generate samples (default 100)

        :return: generated samples
        """ 

        if(self.labels is None):
            generate_labels = None
        else:
            if(generate_labels is None):
                generate_labels = random.choice(key, self.label_dim, [num_samples], replace = True)
            elif(isinstance(generate_labels, (str, int))):
                generate_labels = jnp.array([self.label_to_num[generate_labels]] * num_samples)
            else:
                generate_labels = jnp.array([self.label_to_num[label] for label in generate_labels])
            
        subkey, key = random.split(key)

        if(init_noise is not None):
            if(init_noise.ndim == 2):
                init_noise = init_noise[None, :, :]
            generated_samples = [init_noise]
        else:
            generated_samples =  [self.noise_func(size = num_samples, dimention = self.space_dim, mean_scale = self.mean_scale, cov_scale = self.cov_scale, key = subkey)]

        

        dt = 1/timesteps

        for t in tqdm(jnp.linspace(1, 0, timesteps)):
            grad_fn = self.get_flow(generated_samples[-1][0], generated_samples[-1][1], t, generate_labels)
            generated_samples.append([generated_samples[-1][0] + dt * grad_fn[0], self.psd_project_jit(generated_samples[-1][1] + dt * grad_fn[1])])
        if(generate_labels is None):
            return generated_samples
        return generated_samples, [self.num_to_label[label] for label in np.array(generate_labels)]