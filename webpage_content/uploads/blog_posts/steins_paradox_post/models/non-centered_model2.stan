data {
  int<lower=0> N;              // Number of players
  array[N] int<lower=0> Tr;     // Trials for each player
  array[N] int<lower=0> y;      // Successes for each player
}

parameters {
  vector[N] alpha_raw;          // Raw parameters to be scaled
  real mu;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] alpha;
  alpha = mu + sigma * alpha_raw; // Scaling alpha_raw by sigma
}

model {
  // Priors
  sigma ~ normal(0, 1);
  mu ~ normal(0.3, 1);
  alpha_raw ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
    target += binomial_logit_lpmf(y[n] | Tr[n], alpha[n]);
  }
}

generated quantities {
  array[N] int y_pred;          // Predicted number of successes
  array[N] real p_hat_pred;     // Mean predicted probability of success

  for (n in 1:N) {
    p_hat_pred[n] = inv_logit(alpha[n]);
    y_pred[n] = binomial_rng(Tr[n], p_hat_pred[n]);
  }
}