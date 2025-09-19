import optax

# Given 
# variance schedule: β_t

# x_t is sampled from: N(mean = sqrt(1-b)*x_{t-1}, var = B_t*I (diagonal))

# alpha_t = 1-Beta_t
# bar alpha_t = pi_0^t alpha_s

# epsilon ~ N(0, 1)
# sampled noise = sqrt(alpha_t) x_0 + epsilon * sqrt(1-alpha_bar),



def loss(pred, true_noise):
    return optax.losses.squared_error(pred, true_noise).mean()